# The code is adapted from MONAI: https://docs.monai.io/en/stable/_modules/monai/networks/nets/resnet.html
# The original code is licensed under the Apache License, Version 2.0 (the "License"):
# http://www.apache.org/licenses/LICENSE-2.0
# 

from collections.abc import Callable
from functools import partial
from typing import Any
import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.
        feature_vector_length: length of the vector of extra features such as sex + race
        MLP_hidden_layer_sizes: number of neurons in the hidden layer (e.g. [64, 32]), default to None

    """

    def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 1,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
        feature_vector_length: int = 9,  # sex 2, race 7, one-hot encoding
        MLP_hidden_layer_sizes: list[int] | None = None,  # number of neurons in the hidden layers, [64, 32] will be input->64->32->num_classes

    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])

        # feedforward network
        self.feature_vector_length = feature_vector_length
        MLP_num_neuron_in = block_inplanes[3] * block.expansion + feature_vector_length  # Will be overwritten later if MLP_hidden_layer_sizes is not None
        
        if (MLP_hidden_layer_sizes == None) or (MLP_hidden_layer_sizes == []):
            self.mlp = nn.Linear(MLP_num_neuron_in, num_classes)
        else:
            MLP_layers = []
            for num_neuron in MLP_hidden_layer_sizes:
                MLP_layers.append(nn.Linear(MLP_num_neuron_in, num_neuron))
                MLP_layers.append(nn.ReLU())
                MLP_num_neuron_in = num_neuron
            MLP_layers.append(nn.Linear(MLP_num_neuron_in, num_classes))
            self.mlp = nn.Sequential(*MLP_layers)

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: type[ResNetBlock | ResNetBottleneck],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: nn.Module | partial | None = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, x_vec: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        # feedforward network
        if (x_vec == None) and (self.feature_vector_length == 0):
            x = self.mlp(x)
        else:
            x_vec = x_vec.view(x_vec.size(0), -1)
            if x_vec.size(1) != self.feature_vector_length:
                raise ValueError("Feature vector length is different from specified in feature_vector_length.\nThe feedforward network will not work.")
            else:
                x = torch.concat((x, x_vec), dim=1)
                x = self.mlp(x)

        return x


def get_the_resnet_model(model_name, feature_vector_length, MLP_hidden_layer_sizes, n_input_channels=2):
    if model_name == 'resnet10':
        model = ResNet(block=ResNetBlock,
                       layers=[1, 1, 1, 1],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
    
    elif model_name == 'resnet18':    
        model = ResNet(block=ResNetBlock,
                       layers=[2, 2, 2, 2],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
        
    elif model_name == 'resnet34':
        model = ResNet(block=ResNetBlock,
                       layers=[3, 4, 6, 3],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
        
    elif model_name == 'resnet50':
        model = ResNet(block=ResNetBottleneck,
                       layers=[3, 4, 6, 3],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
    
    elif model_name == 'resnet101':
        model = ResNet(block=ResNetBottleneck,
                       layers=[3, 4, 23, 3],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
        
    elif model_name == 'resnet152':
        model = ResNet(block=ResNetBottleneck,
                       layers=[3, 8, 36, 3],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
    
    elif model_name == 'resnet200':
        model = ResNet(block=ResNetBottleneck,
                       layers=[3, 24, 36, 3],
                       block_inplanes=get_inplanes(),
                       n_input_channels=n_input_channels,
                       feature_vector_length=feature_vector_length,
                       MLP_hidden_layer_sizes=MLP_hidden_layer_sizes)
    
    else:
        raise ValueError(f"Unknown model name {model_name}, use resnet10/18/34/50/101/152/200")
        
    return model

