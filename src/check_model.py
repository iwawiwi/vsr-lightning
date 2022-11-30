from torchinfo import summary
from torchvision.models.densenet import densenet121
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.resnet import resnet18
from torchvision.models.shufflenetv2 import shufflenet_v2_x2_0

from models.components.densenet import DenseNetVideoEncoder
from models.components.efficientnet import EfficientNetVideoEncoder, _efficientnet_conf
from models.components.mobilenet_v3 import MobileNetVideoEncoder, _mobilenet_v3_conf
from models.components.resnet import ResNetVideoEncoder
from models.components.shufflenet_v2 import ShuffleNetVideoEncoder


def get_resnet_video_encoder():
    model = ResNetVideoEncoder(inplanes=1, outplanes=512)  # using roi 112x112
    return model


def get_efficientnet_video_encoder():
    model = EfficientNetVideoEncoder(mode="efficientnet_b0")
    return model


def get_mobilenet_video_encoder():
    model = MobileNetVideoEncoder(mode="mobilenet_v3_large")
    return model


def get_shufflenet_video_encoder():
    model = ShuffleNetVideoEncoder(mode="shufflenet_v2_x1_5")
    return model


def get_densenet_video_encoder():
    model = DenseNetVideoEncoder(32, (6, 12, 24, 16), 64)
    return model


if __name__ == "__main__":
    # model = get_efficientnet_video_encoder()
    # model = get_resnet_video_encoder()
    # model = get_mobilenet_video_encoder()
    # model = get_shufflenet_video_encoder()
    model = get_densenet_video_encoder()
    model.eval()
    # print(model)
    # summary(model, input_size=(1, 3, 112, 112)) # original vision model
    # summary(model, input_size=(1, 1, 1, 112, 112))  # `(timestep, batch, channel, height, width)` >> input for VGG repo
    summary(
        model, input_size=(1, 29, 1, 88, 88)
    )  # `(batch, timestep, channel, height, width)` >> input for VIPL repo
