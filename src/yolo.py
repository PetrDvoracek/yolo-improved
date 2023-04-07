import torch
import torch.nn as nn

from dataclasses import dataclass

INPUT_SIZE = 416
N_CLASSES = 0


@dataclass
class CNNBlockConfig:
    stride: int
    out_channels: int
    kernel_size: int


@dataclass
class ResidualSEBlockConfig:
    num_repeats: int
    skip2end: bool = False


@dataclass
class HCStairstepUpscaleBlockConfig:
    in_channels_list: list
    out_channels: int


config = [
    CNNBlockConfig(stride=1, out_channels=32, kernel_size=3),
    CNNBlockConfig(stride=2, out_channels=64, kernel_size=3),
    ResidualSEBlockConfig(num_repeats=1),
    CNNBlockConfig(stride=2, out_channels=128, kernel_size=3),
    ResidualSEBlockConfig(num_repeats=2, skip2end=True),
    CNNBlockConfig(stride=2, out_channels=256, kernel_size=3),
    ResidualSEBlockConfig(num_repeats=8, skip2end=True),
    CNNBlockConfig(stride=2, out_channels=512, kernel_size=3),
    ResidualSEBlockConfig(num_repeats=8, skip2end=True),
    CNNBlockConfig(stride=2, out_channels=1024, kernel_size=3),
    ResidualSEBlockConfig(num_repeats=4, skip2end=True),
    HCStairstepUpscaleBlockConfig(
        in_channels_list=[128, 256, 512, 1024], out_channels=1024
    ),
    # CNNBlockConfig(stride=1, out_channels=(4 + 1 + N_CLASSES), kernel_size=3),
    CNNBlockConfig(
        stride=2, out_channels=1024, kernel_size=3
    ),  # 3: p, w, h of the center
    CNNBlockConfig(
        stride=2, out_channels=1024, kernel_size=3
    ),  # 3: p, w, h of the center
    CNNBlockConfig(
        stride=2, out_channels=1024, kernel_size=3
    ),  # 3: p, w, h of the center
    CNNBlockConfig(
        stride=2, out_channels=1024, kernel_size=3
    ),  # 3: p, w, h of the center
    CNNBlockConfig(
        stride=1, out_channels=25*5, kernel_size=3
    ),  # 3: p, w, h of the center
]


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualSEBlock(torch.nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1, skip2end=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                    SqueezeExciteBlock(channels),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats
        self.skip2end = skip2end

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class SqueezeExciteBlock(torch.nn.Module):
    "https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class HCStairstepUpscaleBlock(torch.nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.upscale = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.pointwise_convs = nn.ModuleList()
        for in_channels in sorted(in_channels_list):
            self.pointwise_convs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,  # TODO hardcoded
                        kernel_size=1,
                    ),
                    torch.nn.UpsamplingNearest2d(scale_factor=2),
                )
            )

    def forward(self, *fmaps):  # TODO generalize
        fmaps_sorted = [
            x[1]
            for x in sorted(
                [(x.shape[1], x) for x in fmaps], key=lambda i: i[0]
            )  # sort by n_channels
        ]

        convoluted_f_maps = [
            conv(x) for conv, x in zip(self.pointwise_convs, fmaps_sorted)
        ]
        x = self.upscale(convoluted_f_maps[-1]) + convoluted_f_maps[-2]
        x = self.upscale(x) + convoluted_f_maps[-3]
        x = self.upscale(x) + convoluted_f_maps[-4]
        return x


class YOLOv3(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.layers = self._create_conv_layers()

    def forward(self, x):
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, CNNBlock):
                x = layer(x)
            if isinstance(layer, ResidualSEBlock):
                x = layer(x)
                if layer.skip2end:
                    route_connections.append(x)
            if isinstance(layer, HCStairstepUpscaleBlock):
                x = layer(x, *route_connections)
        return x

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, CNNBlockConfig):
                layers.append(
                    CNNBlock(
                        in_channels,
                        module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=1,
                    )
                )
                in_channels = module.out_channels
            elif isinstance(module, ResidualSEBlockConfig):
                layers.append(
                    ResidualSEBlock(
                        in_channels,
                        num_repeats=module.num_repeats,
                        skip2end=module.skip2end,
                    )
                )
            elif isinstance(module, HCStairstepUpscaleBlockConfig):
                layers.append(
                    HCStairstepUpscaleBlock(
                        in_channels_list=module.in_channels_list,
                        out_channels=module.out_channels,
                    )
                )

        return layers


if __name__ == "__main__":
    inp = torch.rand((1, 3, 416, 416))
    model = YOLOv3()
    out = model(inp)
    print(model)
    print(out.shape)
