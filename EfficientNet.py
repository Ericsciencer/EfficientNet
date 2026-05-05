import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn import init

# ----------------------
# 1. 复现 EfficientNet 核心代码（适配CIFAR10 32x32输入）
# ----------------------
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_connect(x, drop_p=0.2, training=True):
    if not training or drop_p == 0:
        return x
    keep_prob = 1 - drop_p
    batch_size = x.shape[0]
    random_tensor = keep_prob + torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_tensor = torch.floor(random_tensor)
    x = x / keep_prob * binary_tensor
    return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        reduced_channels = make_divisible(in_channels * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_p=0.2
    ):
        super(MBConv, self).__init__()
        self.expand_ratio = expand_ratio
        hidden_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        self.drop_p = drop_p

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True)
            )

        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size, stride,
                padding=kernel_size//2, groups=hidden_channels, bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )

        self.se = SEBlock(hidden_channels, se_ratio)

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = drop_connect(x, self.drop_p, self.training)
            x = x + residual
        return x

# EfficientNet-B0 基础配置（适配32x32 CIFAR10）
BASE_BLOCKS = [
    [3, 1, 1, 32, 16, 1],
    [3, 2, 6, 16, 24, 2],
    [5, 2, 6, 24, 40, 2],
    [3, 2, 6, 40, 80, 3],
    [5, 1, 6, 80, 112, 3],
    [5, 2, 6, 112, 192, 4],
    [3, 1, 6, 192, 320, 1],
]

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_coef=1.0, depth_coef=1.0, se_ratio=0.25):
        super(EfficientNet, self).__init__()
        self.se_ratio = se_ratio

        # Stem 输入层
        in_channels = make_divisible(32 * width_coef)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1, bias=False),  # stride=1 适配32x32
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # 堆叠MBConv模块
        self.blocks = nn.Sequential()
        block_idx = 0
        total_blocks = sum(make_divisible(layer[5] * depth_coef) for layer in BASE_BLOCKS)

        for layer_cfg in BASE_BLOCKS:
            k, s, e, in_c, out_c, n = layer_cfg
            in_c = make_divisible(in_c * width_coef)
            out_c = make_divisible(out_c * width_coef)
            num_layers = make_divisible(n * depth_coef)

            for i in range(num_layers):
                stride = s if i == 0 else 1
                drop_p = 0.2 * block_idx / total_blocks
                self.blocks.add_module(
                    f"mbconv_{block_idx}",
                    MBConv(in_c, out_c, k, stride, e, se_ratio, drop_p)
                )
                in_c = out_c
                block_idx += 1

        # 输出层
        last_channels = make_divisible(1280 * width_coef)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.SiLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(last_channels, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ----------------------
# 2. 数据加载
# ----------------------
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ----------------------
# 3. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item() * images.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc

# ----------------------
# 4. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ----------------------
# 5. 主程序
# ----------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    lr = 0.01
    num_epochs = 20

    # 初始化 EfficientNet 模型
    model = EfficientNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_loader, test_loader = get_data_loaders(batch_size)

    # 指标存储
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 训练
    print(f"Training EfficientNet on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'efficientnet_cifar10.pth')
    print("Model saved as efficientnet_cifar10.pth")

    # 可视化
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 7))

    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')

    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, 11, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('EfficientNet Training Metrics', fontsize=16)

    plt.savefig('efficientnet_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()