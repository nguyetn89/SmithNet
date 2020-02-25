""" Network supporting semantic estimation
    Can be used with any pretrained network
"""

import numpy as np

import torch
import torchvision
import torch.nn as nn
# from torchsummary import summary
# from torchvision import transforms
from utils import summary

from utils import freeze_all_layers
# from utils import tensor_normalize, freeze_all_layers, CV_to_PIL


class NetInfo():
    def __init__(self, reduce_dict, pretrained_net):
        self.reduce_dict = reduce_dict
        self.pretrainted_net = pretrained_net

# Net_dict = {"AlexNet":      NetInfo({2:  0, 4:  2, 8:   5, 16:  12, 32:  14}, torchvision.models.alexnet(pretrained=True)),
#            "VGG11":        NetInfo({2:  2, 4:  5, 8:  10, 16:  15, 32:  20}, torchvision.models.vgg11(pretrained=True)),
#            "VGG11_bn":     NetInfo({2:  7, 4: 14, 8:  21, 16:  28, 32:  30}, torchvision.models.vgg11_bn(pretrained=True)),
#            "VGG13":        NetInfo({2:  9, 4: 14, 8:  29, 16:  24, 32:  26}, torchvision.models.vgg13(pretrained=True)),
#            "VGG13_bn":     NetInfo({2: 13, 4: 20, 8:  27, 16:  34, 32:  36}, torchvision.models.vgg13_bn(pretrained=True)),
#            "VGG16":        NetInfo({2:  9, 4: 16, 8:  23, 16:  30, 32:  32}, torchvision.models.vgg16(pretrained=True)),
#            "VGG16_bn":     NetInfo({2: 13, 4: 23, 8:  33, 16:  43, 32:  45}, torchvision.models.vgg16_bn(pretrained=True)),
#            "VGG19":        NetInfo({2:  9, 4: 18, 8:  27, 16:  36, 32:  38}, torchvision.models.vgg19(pretrained=True)),
#            "VGG19_bn":     NetInfo({2: 13, 4: 26, 8:  39, 16:  52, 32:  54}, torchvision.models.vgg19_bn(pretrained=True)),
#            "ResNet18":     NetInfo({2:  3, 4: 18, 8:  34, 16:  50, 32:  66}, torchvision.models.resnet18(pretrained=True)),
#            "ResNet34":     NetInfo({2:  3, 4: 25, 8:  55, 16:  99, 32: 122}, torchvision.models.resnet34(pretrained=True)),
#            "ResNet50":     NetInfo({2:  3, 4: 39, 8:  81, 16: 143, 32: 172}, torchvision.models.resnet50(pretrained=True)),
#            "ResNet101":    NetInfo({2:  3, 4: 39, 8:  81, 16: 313, 32: 342}, torchvision.models.resnet101(pretrained=True)),
#            "ResNet152":    NetInfo({2:  3, 4: 39, 8: 121, 16: 483, 32: 512}, torchvision.models.resnet152(pretrained=True)),
#            "GoogleNet":    NetInfo({2:  3, 4: 10, 8:  51, 16: 152, 32: 193}, torchvision.models.googlenet(pretrained=True)),
#            "MobileNet_v2": NetInfo({2: 12, 4: 30, 8:  57, 16: 120, 32: 156}, torchvision.models.mobilenet_v2(pretrained=True)),
#            "ResNeXt50":    NetInfo({2:  3, 4: 39, 8:  81, 16: 143, 32: 172}, torchvision.models.resnext50_32x4d(pretrained=True)),
#            "ResNeXt101":   NetInfo({2:  3, 4: 39, 8:  81, 16: 313, 32: 342}, torchvision.models.resnext101_32x8d(pretrained=True))}


Net_dict = {"VGG19": NetInfo({2:  9, 4: 18, 8:  27, 16:  36, 32:  38},
            torchvision.models.vgg19(pretrained=True))}


class SemanticNet(nn.Module):
    def __init__(self, size, backbone="VGG19", reduction=8, n_layer_desire=-1, device=None):
        assert backbone in Net_dict
        assert reduction in (2, 4, 8, 16, 32)
        assert isinstance(size, (list, tuple)) and len(size) == 2
        super(SemanticNet, self).__init__()
        # resolution reduction from input to output
        self.IM_SIZE = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.data_mean = torch.tensor([0.485, 0.456, 0.406]).type(torch.float32).to(self.device)
        self.data_std = torch.tensor([0.229, 0.224, 0.225]).type(torch.float32).to(self.device)
        self.size_reduce = reduction
        net_info = Net_dict[backbone]
        n_layer = net_info.reduce_dict[reduction]
        if n_layer_desire > 0 and n_layer_desire < n_layer:
            n_layer = n_layer_desire
        if n_layer > 0:
            self.model = net_info.pretrainted_net.features[:n_layer]
            freeze_all_layers(self.model)
            self.model.to(self.device)
        else:
            raise Exception("""Cannot init network with backbone='%s'
                            and reduction=%d""" % (backbone, reduction))

    def is_frozen(self):
        for param in self.parameters():
            if param.requires_grad:
                return False
        return True

    def forward(self, x):
        return self.model(x)

    def get_summary(self, print_details=True):
        _, _, layer_shapes = summary(self, (3, self.IM_SIZE[0], self.IM_SIZE[1]),
                                     print_details=print_details)
        return layer_shapes

    # upscale: None -> do nothing; 1: original image size
    # (h, w): specified size
    def _evaluate_batch(self, imgs, upscale):
        assert upscale is None or upscale == 1 \
                or (isinstance(upscale, (list, tuple)) and len(upscale) == 2)
        # normalize if needed
        # if isinstance(imgs, (list, tuple)):
        #     imgs = np.array(imgs)
        assert isinstance(imgs, (np.ndarray, torch.Tensor))
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.from_numpy(imgs.astype(np.float32))
        assert torch.max(imgs).item() <= 1. and torch.min(imgs).item() >= 0.
        # convert image to channel_last, then normalize and convert to channel_first
        imgs = torch.stack([((img.permute(1, 2, 0) - self.data_mean) / self.data_std).permute(2, 0, 1) for img in imgs])
        # set model to evaluation mode and execute
        self.model.eval()
        with torch.no_grad():
            features = self.model(imgs.to(self.device))
            if upscale:
                sz = self.IM_SIZE if upscale == 1 else upscale
                scaled_features = nn.functional.interpolate(
                                  features, sz,
                                  mode="bilinear",
                                  align_corners=True)
                return features, scaled_features.to(self.device)

        return features, None

    def estimate(self, imgs, upscale=None, print_shape=False):
        features, scaled_features = self._evaluate_batch(imgs, upscale)
        if print_shape:
            print("feature shape:", features.shape)
            if scaled_features is not None:
                print("scaled shape:", scaled_features.shape)
        return features, scaled_features


# def simple_test():
#     SCALE = 0.09
#     img_path = './../test-images/office/15.png'
#     image = cv2.resize(cv2.imread(img_path), (0, 0), fx=SCALE, fy=SCALE)
#     in_shape = image.shape[:2]
#     estimator = SemanticNet(in_shape, backbone="VGG19", reduction=8)
#     estimator.get_summary(print_details=True)
#     estimator.estimate([image, cv2.flip(image, 1)], upscale=1)
#
#
# if __name__ == "__main__":
#     simple_test()
