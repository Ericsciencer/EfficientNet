import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# -------------------------- 工具函数 --------------------------
def make_divisible(v, divisor=8, min_value=None):
    """
    保证通道数是8的倍数（EfficientNet官方设计）
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 防止通道数缩减超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_connect(x, drop_p=0.2, training=True):
    """
    随机深度（Drop Connect）：残差连接的正则化方法
    """
    if not training or drop_p == 0:
        return x
    keep_prob = 1 - drop_p
    batch_size = x.shape[0]
    # 生成随机掩码
    random_tensor = keep_prob + torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_tensor = torch.floor(random_tensor)
    x = x / keep_prob * binary_tensor
    return x

# -------------------------- SE通道注意力模块 --------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        # 压缩通道
        reduced_channels = make_divisible(in_channels * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=True),
            nn.SiLU(inplace=True),  # SiLU = Swish
            nn.Linear(reduced_channels, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 压缩 -> 激励 -> 通道加权
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -------------------------- MBConv 核心模块 --------------------------
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio=0.25,
        drop_p=0.2
    ):
        super(MBConv, self).__init__()
        self.expand_ratio = expand_ratio
        hidden_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        self.drop_p = drop_p

        # 1x1 升维卷积
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True)
            )

        # 深度卷积（DWConv）
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size, stride,
                padding=kernel_size//2, groups=hidden_channels, bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )

        # SE注意力
        self.se = SEBlock(hidden_channels, se_ratio)

        # 1x1 降维卷积
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        # 升维
        if self.expand_ratio != 1:
            x = self.expand_conv(x)

        # 深度卷积 + SE
        x = self.dwconv(x)
        x = self.se(x)

        # 降维
        x = self.project_conv(x)

        # 残差连接 + 随机深度
        if self.use_residual:
            x = drop_connect(x, self.drop_p, self.training)
            x = x + residual
        return x

# -------------------------- EfficientNet 主网络 --------------------------
# 论文官方：各版本缩放参数 (宽度系数, 深度系数, 输入分辨率, Dropout率)
EFFICIENTNET_CONFIG = {
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
}

# EfficientNet-B0 基础模块配置（所有版本基于此缩放）
BASE_BLOCKS = [
    # kernel_size, stride, expand_ratio, in_c, out_c, num_layers
    [3, 1, 1, 32, 16, 1],
    [3, 2, 6, 16, 24, 2],
    [5, 2, 6, 24, 40, 2],
    [3, 2, 6, 40, 80, 3],
    [5, 1, 6, 80, 112, 3],
    [5, 2, 6, 112, 192, 4],
    [3, 1, 6, 192, 320, 1],
]

class EfficientNet(nn.Module):
    def __init__(self, model_name="b0", num_classes=1000, se_ratio=0.25):
        super(EfficientNet, self).__init__()
        # 加载对应版本的缩放参数
        width_coef, depth_coef, self.image_size, dropout_rate = EFFICIENTNET_CONFIG[model_name]
        self.se_ratio = se_ratio

        # ==================== Stem 输入层 ====================
        in_channels = make_divisible(32 * width_coef)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # ==================== 堆叠 MBConv 模块 ====================
        self.blocks = nn.Sequential()
        block_idx = 0
        # 计算总模块数（用于动态调整Drop Connect概率）
        total_blocks = sum(make_divisible(layer[5] * depth_coef) for layer in BASE_BLOCKS)

        for layer_cfg in BASE_BLOCKS:
            k, s, e, in_c, out_c, n = layer_cfg
            # 复合缩放：调整通道数 + 层数
            in_c = make_divisible(in_c * width_coef)
            out_c = make_divisible(out_c * width_coef)
            num_layers = make_divisible(n * depth_coef)

            for i in range(num_layers):
                stride = s if i == 0 else 1
                # 动态Drop Connect概率
                drop_p = 0.2 * block_idx / total_blocks
                # 添加MBConv模块
                self.blocks.add_module(
                    f"mbconv_{block_idx}",
                    MBConv(in_c, out_c, k, stride, e, se_ratio, drop_p)
                )
                in_c = out_c
                block_idx += 1

        # ==================== Head 输出层 ====================
        last_channels = make_divisible(1280 * width_coef)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.SiLU(inplace=True)
        )

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(last_channels, num_classes)

        # 权重初始化
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

# -------------------------- 快速创建模型函数 --------------------------
def efficientnet_b0(num_classes=1000): return EfficientNet("b0", num_classes)
def efficientnet_b1(num_classes=1000): return EfficientNet("b1", num_classes)
def efficientnet_b2(num_classes=1000): return EfficientNet("b2", num_classes)
def efficientnet_b3(num_classes=1000): return EfficientNet("b3", num_classes)
def efficientnet_b4(num_classes=1000): return EfficientNet("b4", num_classes)
def efficientnet_b5(num_classes=1000): return EfficientNet("b5", num_classes)
def efficientnet_b6(num_classes=1000): return EfficientNet("b6", num_classes)
def efficientnet_b7(num_classes=1000): return EfficientNet("b7", num_classes)

# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 1. 创建模型（以B0为例，修改分类数为10）
    model = efficientnet_b0(num_classes=10)
    model.eval()

    # 2. 构造随机输入（batch=2, 3通道, 对应版本分辨率）
    dummy_input = torch.randn(2, 3, 224, 224)  # B0对应224

    # 3. 前向推理
    with torch.no_grad():
        output = model(dummy_input)

    print(f"模型输出形状: {output.shape}")  # 预期: torch.Size([2, 10])
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")