import cv2
import glob
import torch
import os
import sys
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from collections import OrderedDict

from CONFIG import data_info

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

tensor_normalize = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=mean,
                                                            std=std)])


def tensor_restore(in_tensor, clamp=True):
    out_tensor = in_tensor
    # multiple std
    out_tensor[0] *= std[0]
    out_tensor[1] *= std[1]
    out_tensor[2] *= std[2]
    # plus mean
    out_tensor[0] += mean[0]
    out_tensor[1] += mean[1]
    out_tensor[2] += mean[2]
    if clamp:
        out_tensor = torch.clamp(out_tensor, min=0.0, max=1.0)
    return out_tensor


def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def PIL_to_CV(PIL_img):
    if isinstance(PIL_img, str):
        data = Image.open(PIL_img)
    else:
        data = PIL_img
    return cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)


def CV_to_PIL(CV_img):
    if isinstance(CV_img, str):
        data = cv2.imread(CV_img)
    else:
        data = CV_img
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return Image.fromarray(data)


def get_img_shape(in_shape):
    if isinstance(in_shape, int):
        return (in_shape, in_shape)
    else:
        assert isinstance(in_shape, (list, tuple)) and len(in_shape) >= 2
        return tuple(in_shape[:2])


def load_video(file, im_size=None):
    imgs = []
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print("Error opening file", file)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if im_size is not None:
                frame = CV_to_PIL(cv2.resize(frame, (im_size[1], im_size[0])))
            imgs.append(frame)
        else:
            break
    cap.release()
    return imgs


def load_imgs_in_directory(path, ext, im_size=None):
    files = sorted(glob.glob(path + "/*." + ext))
    if im_size is not None:
        imgs = [CV_to_PIL(cv2.resize(cv2.imread(file), (im_size[1], im_size[0]))) for file in files]
    else:
        imgs = [CV_to_PIL(cv2.imread(file)) for file in files]
    return imgs


# plot ROC curve
def plot_ROC(true_vals, pred_vals, pos_label=1):
    fpr, tpr, thresholds = roc_curve(true_vals, pred_vals, pos_label=pos_label)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(true_vals, pred_vals))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# image gradients along x and y axes
# input is a batch of size (b, c, h, w)
def image_gradient(tensor, out_abs=False):
    assert len(tensor.shape) == 4
    h, w = tensor.shape[-2:]
    left, top = tensor, tensor
    right = F.pad(tensor, [0, 1, 0, 0])[:, :, :, 1:]
    bottom = F.pad(tensor, [0, 0, 0, 1])[:, :, 1:, :]
    # dx
    dx = right - left
    dx[:, :, :, -1] = 0
    # dy
    dy = bottom - top
    dy[:, :, -1, :] = 0
    #
    if out_abs:
        return torch.abs(dx), torch.abs(dy)
    return dx, dy


# torchsummary for automatically estimating feature shape
# code adapted from https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
def summary(model, input_size, batch_size=-1, device="cuda", print_details=False):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if print_details:
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0

    sequence_layer_shapes = []

    for layer in summary:
        sequence_layer_shapes.append(summary[layer]["output_shape"])

        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]

        # input_shape, output_shape, trainable, nb_params
        if print_details:
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    if print_details:
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
    # return summary
    return total_params, trainable_params, sequence_layer_shapes


# Class forming dataset helper
# name: name of dataset for further storing files
# im_size: expected image size
# in_path: path for loading data
# out_file: path for saving processed data (if in_path contains raw data)
# ext:
#       "/" for directories containing images
#       ".pt" for preprocessed data files
#       ".xxx" for reading videos in the path
# batch_size: number of frames feeding into network for each propagation
class DataHelper(torch.utils.data.Dataset):
    def __init__(self, name, im_size, in_path, out_file=None, default_ext=".avi", batch_size=24):
        super(DataHelper, self).__init__()
        # set name and data path
        self.name = name
        self.in_path = in_path
        self.out_file = out_file

        # load data in the provided path
        print("Processing data path: '%s'..." % in_path)
        if in_path[-3:] == ".pt" or (out_file is not None and out_file[:-3] == ".pt"):  # data processed and saved in single file
            self.data = torch.load(in_path)
        else:                   # load ensemble of single files
            extensions = [".pt", "/", ".*"]
            if default_ext not in extensions:
                extensions.insert(0, default_ext)
            ext = None
            for tmp_ext in extensions:
                template = in_path + "/*" + tmp_ext
                data_paths = sorted(glob.glob(template))
                if len(data_paths) > 0:
                    ext = tmp_ext
                    break
            if ext is None:
                print("ERROR: data files not found!")
                return
            if ext == ".pt":    # data from .pt files
                self.data = [torch.load(data_path) for data_path in data_paths]
            elif ext == "/":    # load images in multiple directories
                self.data = [torch.stack([tensor_normalize(datum)
                             for datum in load_imgs_in_directory(data_path, "*", im_size)]) for data_path in data_paths]
            else:               # load all videos in the specified path
                self.data = [torch.stack([tensor_normalize(datum) for datum in load_video(data_path, im_size)]) for data_path in data_paths]

            # set directory for storing preprocessed data files
            if out_file is not None:
                out_path, _ = os.path.split(out_file)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                # storing data file
                torch.save(self.data, out_file)

        # set default batch_size
        self.set_batch_size(batch_size)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        if batch_size > 0:
            self.batch_nums = [int(np.ceil(len(self.data[i])/self.batch_size)) for i in range(len(self.data))]
            self.batch_indices = [[(i*self.batch_size, min((i+1)*self.batch_size, len(self.data[j]))) for i in range(self.batch_nums[j])]
                                  for j in range(len(self.data))]
        else:
            self.batch_nums = len(self.data)
            self.batch_indices = list(range(self.batch_nums))

    def get_data_shape(self):
        return [datum.shape for datum in self.data]

    def get_num_of_batchs(self):
        return np.sum(self.batch_nums) if self.batch_size > 0 else self.batch_nums

    def __getitem__(self, index):
        assert index in range(len(self.data))
        # return whole video data and list of clip indices
        if self.batch_size > 0:
            return index, self.batch_indices[index]
        else:
            return index, self.data[index]

    def __len__(self):
        return len(self.data)


# Class defining specific dataset for training and evaluation
class DatasetDefiner():
    def __init__(self, name, im_size, batch_size):
        assert name in ("UCSDped1", "UCSDped2", "Avenue", "Entrance", "Exit", "Shanghai", "Crime", "Belleview", "Train", "just4test")
        # set basic attributes
        self._name = name
        self._batch_size = batch_size
        self._im_size = im_size
        # other attributes for further uses
        self._training_data = None
        self._evaluation_data = None
        # set dataset attributes
        self._set_dataset_attributes()

    # get private attributes by name
    def get_attribute(self, attribute):
        assert isinstance(attribute, str)
        if attribute == "name":
            return self._name
        if attribute == "seq_len":
            return self._seq_len
        if attribute == "stride":
            return self._stride
        if attribute == "im_size":
            return self._im_size
        if attribute == "training_data":
            return self._training_data
        if attribute == "evaluation_data":
            return self._evaluation_data
        raise ValueError("Unknown attribute %s" % attribute)

    # create data helper for training set
    def load_training_data(self, out_file=None):
        if self._training_data is not None:
            print("# ERROR: training data have already loaded!")
            return
        print("%s -> Loading training data..." % self._name)
        self._training_data = DataHelper(self._name, self._im_size, self._training_path,
                                         out_file=out_file, default_ext=".pt", batch_size=self._batch_size)
        print("%s -> Training data loaded!" % self._name)
        print("Data shape:", self._training_data.get_data_shape())

    # create data helper for evaluation set
    def load_evaluation_data(self, out_file=None):
        if self._evaluation_data is not None:
            print("# ERROR: evaluation data have already loaded!")
            return
        print("%s -> Loading evaluation data..." % self._name)
        self._evaluation_data = DataHelper(self._name, self._im_size, self._evaluation_path,
                                           out_file=out_file, default_ext=".pt", batch_size=self._batch_size)
        print("%s -> Evaluation data loaded!" % self._name)
        print("Data shape:", self._evaluation_data.get_data_shape())

    # clip_results: sequence of anomaly scores (clips) for the whole test set
    # clip_results must be an array of arrays (either numpy or torch tensor)
    def evaluate(self, clip_results):
        assert len(clip_results) == self._n_clip_test
        groundtruths = [np.zeros_like(clip_result) for clip_result in clip_results]
        # set frame-level groundtruth scores
        for clip_idx in range(len(self._eval_groundtruth_clips)):
            anomaly_intervals = self._eval_groundtruth_frames[clip_idx]
            for i in range(len(anomaly_intervals)//2):
                # -1 because _eval_groundtruth_frames given in 1-based index
                start = anomaly_intervals[2*i] - 1
                end = anomaly_intervals[2*i+1] - 1
                groundtruths[clip_idx][start:end] = 1
        # flatten groundtruth and predicted scores for evaluation
        true_results = np.concatenate(groundtruths, axis=0)
        pred_results = np.concatenate(clip_results, axis=0)

        # plot roc ROC curve
        # plot_ROC(true_results, pred_results)
        return roc_auc_score(true_results, pred_results)

    # set attributes related to each dataset
    def _set_dataset_attributes(self):
        if self._name in data_info:
            info = data_info[self._name]
            self._n_clip_train = info["n_clip_train"]
            self._n_clip_test = info["n_clip_test"]
            self._training_path = info["training_path"]
            self._evaluation_path = info["evaluation_path"]
            self._eval_groundtruth_frames = info["eval_groundtruth_frames"]
            self._eval_groundtruth_clips = info["eval_groundtruth_clips"]
        else:
            raise ValueError("Unknown dataset")
        assert len(self._eval_groundtruth_clips) == len(self._eval_groundtruth_frames)


# Modified from https://stackoverflow.com/questions/3160699/python-progress-bar
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=80, fmt=DEFAULT, symbol='#', output=sys.stderr):
        assert len(symbol) == 1
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)
        self.current = 0

    def __call__(self, msg=''):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '|' + self.symbol * size + ' ' * (self.width - size) + '|'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args + msg, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
