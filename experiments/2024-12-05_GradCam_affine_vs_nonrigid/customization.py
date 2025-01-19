# The original forward() function takes two tensors as input:
# x (FA and MD image concatenated as two channels) and 
# x_vec (vector for sex and race info).
# Since pytorch-grad-cam doesn't seem to natively support multiple
# input arguments, some customization is required to run it.
# It can be either customizing the code of pytorch-grad-cam, or 
# customizing the code of our model.
# I chose the latter, as it seems more straightforward.
# The basic idea is to pass in the x_vec when defining the model.


import torch
import torch.nn as nn
from pathlib import Path, PosixPath
from collections.abc import Callable
from functools import partial
from typing import Any
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ToTensord,
    ConcatItemsd,
)


def vectorize_label(label_name, label_value):
    
    if label_name == 'sex':
        lut = {
            0: [1, 0],
            1: [0, 1],
        }
        if label_value in lut.keys():
            return lut[label_value]
        else:
            return [0.5, 0.5]

    elif label_name == 'race':
        lut = {
            1: [1, 0, 0, 0], 
            2: [0, 1, 0, 0], 
            3: [0, 0, 1, 0],
            4: [0, 0, 0, 1],
            0: [0.25, 0.25, 0.25, 0.25],
        }
        if label_value in lut.keys():
            return lut[label_value]
        else:
            return [0.25, 0.25, 0.25, 0.25]
    
    else:
        raise ValueError(f'Unknown label name: {label_name}')


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
        sex: int | None = None,
        race: int | None = None,
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
        
        # Feature vector for sex and race
        sex_vec = vectorize_label(label_name='sex', label_value=sex)
        sex_vec = torch.tensor(sex_vec, dtype=torch.float32)
        race_vec = vectorize_label(label_name='race', label_value=race)
        race_vec = torch.tensor(race_vec, dtype=torch.float32)
        label_feature_vec = torch.cat((sex_vec, race_vec), dim=0)
        label_feature_vec = label_feature_vec.unsqueeze(0)
        self.label_feature_vec = label_feature_vec.to('cuda')  # this "parameter" won't go to cuda with model.to('cuda')


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_vec = self.label_feature_vec

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


def get_the_resnet_model(
        model_name, feature_vector_length, MLP_hidden_layer_sizes, n_input_channels,
        sex, race,
        ):
    
    if model_name != 'resnet101':
        raise Exception("This function is customized and only supports resnet101.")
    
    model = ResNet(
        block=ResNetBottleneck,
        layers=[3, 4, 23, 3],
        block_inplanes=get_inplanes(),
        n_input_channels=n_input_channels,
        feature_vector_length=feature_vector_length,
        MLP_hidden_layer_sizes=MLP_hidden_layer_sizes,
        sex=sex,
        race=race,
        )
        
    return model


def load_trained_model(
        model_name, mlp_hidden_layer_sizes, feature_vector_length, n_input_channels, 
        sex, race, 
        path_pth, device=None
        ):
    """
    Load a trained model with specified parameters and weights.
    
    Args:
        model_name (str): The name of the ResNet model to use, e.g. 'resnet101'.
        mlp_hidden_layer_sizes (list[int]): List of hidden layer sizes for the MLP.
        feature_vector_length (int): The length of the feature vector.
        n_input_channels (int): Number of input channels for the model.
        sex (int): 0 for female, 1 for male.
        race (int): 1 for "White", 2 for "Asian", 3 for "Black or African American", 4 for "American Indian or Alaska Native", and 0 for "Some Other Race".
        path_pth (str): Path to the `.pth` file containing model weights.
        device (str | torch.device | None): Device to load the model on ('cpu', 'cuda', or torch.device).
    
    Returns:
        torch.nn.Module: The loaded and initialized model.
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise ValueError(f"{device} not supported. Use None, 'cpu', 'cuda', 'cuda:1', torch.device, etc.")
    
    # model configuration
    model = get_the_resnet_model(
        model_name = model_name,
        feature_vector_length = feature_vector_length,
        MLP_hidden_layer_sizes = mlp_hidden_layer_sizes,
        n_input_channels = n_input_channels,
        sex = sex,
        race = race,
    )
    model = model.to(device)
    
    # load model weights
    if not Path(path_pth).is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {path_pth}")
    try:
        checkpoint = torch.load(path_pth, map_location=device)
        model.load_state_dict(checkpoint)
    except:
        raise ValueError(f"Error loading checkpoint file: {path_pth}")
        
    print(f"Trained model loaded on {device}")
    return model


def load_images(path_fa=None, path_md=None):
    """Load an FA-MD pair as tensor.

    Args:
        path_fa (str | PosixPath): Path to the FA image in the MNI152 space.
        path_md (str | PosixPath): Path to the MD image in the MNI152 space.

    Returns:
        torch.Tensor: A tensor of shape (1, C, D, H, W) containing the image(s) with a batch dimension.
    """
    
    transform = Compose([
        LoadImaged(keys=['fa', 'md'], image_only=False),
        EnsureChannelFirstd(keys=['fa', 'md']),
        Orientationd(keys=['fa', 'md'], axcodes="RAS"),
        ToTensord(keys=['fa', 'md']),
        ConcatItemsd(keys=['fa', 'md'], name='images')
    ])
    data_dict = {'fa': path_fa, 'md': path_md}
    data_dict = transform(data_dict)
    img = data_dict['images']
    img = img.unsqueeze(0)
    
    return img
