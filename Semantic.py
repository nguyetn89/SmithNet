""" Network supporting semantic estimation
    Can be used with any pretrained network
"""

import cv2

import torch
import torchvision
import torch.nn as nn
# from torchsummary import summary
from utils import summary

from utils import tensor_normalize, freeze_all_layers, CV_to_PIL


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
    def __init__(self, size, backbone="VGG19", reduction=8, n_layer_desire=-1):
        assert backbone in Net_dict
        assert reduction in (2, 4, 8, 16, 32)
        assert isinstance(size, (list, tuple)) and len(size) == 2
        super(SemanticNet, self).__init__()
        # resolution reduction from input to output
        self.IM_SIZE = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
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
    def _evaluate_batch(self, imgs, is_cv_img, need_normalize, upscale):
        assert upscale is None or upscale == 1 \
                or (isinstance(upscale, (list, tuple)) and len(upscale) == 2)
        # convert opencv data to PIL data
        if isinstance(imgs, (list, tuple)):
            if is_cv_img:
                PIL_imgs = [CV_to_PIL(img) for img in imgs]
            else:
                PIL_imgs = imgs
            if need_normalize:
                imgs_tensor = torch.stack([tensor_normalize(img).float() for img in PIL_imgs])
            else:
                imgs_tensor = torch.stack(PIL_imgs)
        else:
            imgs_tensor = imgs

        # set model to evaluation mode and execute
        self.model.eval()
        with torch.no_grad():
            features = self.model(imgs_tensor.to(self.device))
            if upscale:
                sz = self.IM_SIZE if upscale == 1 else upscale
                scaled_features = nn.functional.interpolate(
                                  features, sz,
                                  mode="bilinear",
                                  align_corners=True)
                return features, scaled_features.to(self.device)

        return features, None

    def estimate(self, imgs, is_cv_img=True, need_normalize=True,
                 upscale=None, print_shape=False):
        features, scaled_features = self._evaluate_batch(
                                    imgs, is_cv_img,
                                    need_normalize,
                                    upscale)
        if print_shape:
            print("feature shape:", features.shape)
            if scaled_features is not None:
                print("scaled shape:", scaled_features.shape)
        return features, scaled_features


def simple_test():
    SCALE = 0.09
    img_path = './../test-images/office/15.png'
    image = cv2.resize(cv2.imread(img_path), (0, 0), fx=SCALE, fy=SCALE)
    in_shape = image.shape[:2]
    estimator = SemanticNet(in_shape, backbone="VGG19", reduction=8)
    estimator.get_summary(print_details=True)
    estimator.estimate([image, cv2.flip(image, 1)], upscale=1)


if __name__ == "__main__":
    simple_test()
