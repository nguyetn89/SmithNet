import cv2
import glob
import torch
import os
import sys
import time
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy.io import loadmat
from collections import OrderedDict

from flowlib import flow_to_image
from CONFIG import data_info


def extend_flow_channel_in_batch(in_flows, channel_first=True):
    assert len(in_flows.shape) == 4
    if in_flows.shape[1] == 3:
        return in_flows
    elif in_flows.shape[1] == 2:
        # magnitudes = torch.stack([torch.sum(flow**2, dim=0)**0.5 for flow in in_flows])
        if isinstance(in_flows, torch.Tensor):
            magnitudes = torch.sum(in_flows**2, dim=1, keepdim=True)**0.5
            return torch.cat([in_flows, magnitudes], dim=1)
        elif isinstance(in_flows, np.ndarray):
            magnitudes = np.sum(in_flows**2, axis=1, keepdims=True)**0.5
            return np.concatenate([in_flows, magnitudes], axis=1)
        else:
            print("Unsupported data type:", type(in_flows))
    else:
        print("Unknown batch of optical flows shape:", in_flows.shape)
        return in_flows


def image_from_flow(in_flow, channel_first):
    assert len(in_flow.shape) == 3
    assert isinstance(in_flow, np.ndarray)
    if channel_first:
        assert in_flow.shape[0] in (2, 3)
        out_image = flow_to_image(np.transpose(in_flow, (1, 2, 0))) / 255.0
        out_image = np.transpose(out_image, (2, 0, 1))
    else:
        out_image = flow_to_image(in_flow) / 255.0
    return out_image.astype(np.float32)


def visualize_error_map(error_map, channel_first=True):
    if isinstance(error_map, torch.Tensor):
        min_func, max_func = torch.min, torch.max
    else:   # expect numpy array
        min_func, max_func = np.min, np.max
    if channel_first:
        for i in range(error_map.shape[0]):
            error_map[i] = (error_map[i] - min_func(error_map[i])) / (max_func(error_map[i]) - min_func(error_map[i]))
    else:
        for i in range(error_map.shape[-1]):
            error_map[..., i] = (error_map[..., i] - min_func(error_map[..., i])) / (max_func(error_map[..., i]) - min_func(error_map[..., i]))
    return error_map


# this function focuses on two types of input:
#   + image: input range [-1, 1]
#   + flow: input range [any, any]
def images_restore(in_data, clamp=True, convert_unit8=False, is_optical_flow=False):
    if isinstance(in_data, (list, tuple)):
        if type(in_data[0]) == np.ndarray:
            if is_optical_flow:
                out_data = np.array([image_from_flow(flow, channel_first=True)
                                     for flow in in_data]).astype(np.float32)
            else:
                out_data = np.array(in_data).astype(np.float32)
        elif type(in_data[0]) == torch.Tensor:
            if is_optical_flow:
                out_data = torch.stack([torch.tensor(image_from_flow(flow.cpu().numpy()), channel_first=True)
                                        for flow in in_data], dim=0).type(torch.float32)
            else:
                out_data = torch.stack(in_data, dim=0).type(torch.float32)
        else:
            print("Unsupported data type:", type(in_data[0]))
            return in_data
    else:
        assert isinstance(in_data, (np.ndarray, torch.Tensor))
        out_data = in_data.astype(np.float32) if type(in_data) == np.ndarray else in_data.type(torch.float32)
        if is_optical_flow:
            out_data = image_from_flow(out_data, channel_first=True) if type(in_data) == np.ndarray else \
                       torch.tensor(image_from_flow(out_data.cpu().numpy(), channel_first=True)).type(torch.float32)

    if not is_optical_flow:
        out_data = (out_data + 1.) * 0.5

    if clamp or convert_unit8:
        if type(out_data) == np.ndarray:
            out_data = np.clip(out_data, 0., 1.)
        elif type(out_data) == torch.Tensor:
            out_data = torch.clamp(out_data, min=0., max=1.)

    if convert_unit8:
        if type(out_data) == np.ndarray:
            out_data = (out_data * 255.).astype(np.uint8)
        elif type(out_data) == torch.Tensor:
            out_data = out_data.type(torch.uint8)

    return out_data


def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False


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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            if im_size is not None:
                frame = cv2.resize(frame, (im_size[1], im_size[0]))
            imgs.append(frame)
        else:
            break
    cap.release()
    return imgs


def load_imgs_in_directory(path, ext, im_size=None):
    files = sorted(glob.glob(path + "/*." + ext))
    if im_size is not None:
        imgs = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), (im_size[1], im_size[0])) for file in files]
    else:
        imgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
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
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    assert len(tensor.shape) == 4
    h, w = tensor.shape[-2:]
    left, top = tensor, tensor
    right = F.pad(tensor, [0, 1, 0, 0])[:, :, :, 1:]
    bottom = F.pad(tensor, [0, 0, 0, 1])[:, :, 1:, :]
    # dx
    dx = right - left
    # dx[:, :, :, -1] = 0
    # dy
    dy = bottom - top
    # dy[:, :, -1, :] = 0
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
    x = [torch.ones(2, *in_size).type(dtype) for in_size in input_size]
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
# out_path: path for saving processed data (if in_path contains raw data)
# ext:
#       "/" for directories containing images
#       ".xxx" for reading videos in the path
class DataHelper(torch.utils.data.Dataset):
    def __init__(self, name, im_size, in_path, expected_n_clip,
                 out_path, extension=".npy", force_calc=False):
        super(DataHelper, self).__init__()
        # set name and data path
        self._name = name
        self._im_size = im_size
        self._in_path = in_path
        self._out_path = out_path
        if not os.path.exists(out_path):
            os.mkdirs(out_path)
        self._data_to_access = None

        # list of desired data files
        self._out_data_files = [os.path.join(out_path, str(i+1).zfill(len(str(expected_n_clip))) + ".npy")
                                for i in range(expected_n_clip)]
        file_out_existing = [os.path.exists(file) for file in self._out_data_files]

        # skip processing data if all files are existed (i.e. processed)
        if all(file_out_existing):
            return

        # otherwise, check each clip and load/convert (if not existed)
        self._in_data_paths = sorted(glob.glob(in_path + "/*" + extension))
        assert len(self._in_data_paths) == expected_n_clip
        if extension == ".npy" and not force_calc:  # data already prepared
            return

        # check each clip
        for clip_idx in range(expected_n_clip):
            if file_out_existing[clip_idx] and not force_calc:
                continue
            self.load_clip(clip_idx, get_output=False, force_calc=True)

    # process data of 1 clip and save to .npy file
    # in_path: input data path (either a directory of images or a video file)
    # out_file: path to .npy file storing preprocessed data from in_path
    # im_size: desired frame resolution
    def load_clip(self, clip_idx, get_output=False, force_calc=False):
        out_file = self._out_data_files[clip_idx]

        # clip was already processed -> just load it
        if os.path.exists(out_file) and not force_calc:
            if get_output:
                return np.load(out_file)
            else:
                return

        # do not process .npy file (original data should be images or video)
        in_path = self._in_data_paths[clip_idx]
        if in_path[-4:] == ".npy":
            # print("Input file %s is already a .npy file -> check again" % in_path)
            # return None
            return np.load(in_path)

        # process data from directory / file
        im_size = self._im_size
        if os.path.isdir(in_path):
            data = load_imgs_in_directory(in_path, "*", im_size)
        elif os.path.isfile(in_path):
            data = load_video(in_path, im_size)
        np.save(out_file, data)

        # only return if necessary
        if get_output:
            return data

    def set_clip_idx(self, clip_idx):
        assert clip_idx in range(len(self._out_data_files))
        self._data_to_access = np.transpose(self.load_clip(clip_idx, get_output=True, force_calc=False), (0, 3, 1, 2))

    def __getitem__(self, index):
        assert self._data_to_access is not None
        return self._data_to_access[index]

    def __len__(self):
        return len(self._data_to_access)


# Class defining specific dataset for training and evaluation
# data_path: path that processed data files (.npy) are stored
# mode: this controller is for training or evaluation (for simplifying operations)
class DatasetDefiner():
    def __init__(self, name, im_size, data_path, mode):
        assert mode in ("train", "test")
        self._mode = mode

        assert name in ("UCSDped1", "UCSDped2",
                        "Avenue", "Entrance", "Exit",
                        "ShanghaiTech", "UCF_Crime",
                        "Belleview", "Train",
                        "just4test")
        # set basic attributes
        self._name = name
        self._data_path = data_path
        self._im_size = im_size
        # data controllers
        self.data = {"train": None, "test": None}
        # set dataset attributes
        self._set_dataset_attributes()

    def load_data(self, clip_idx):
        part = self._mode
        assert clip_idx in range(self._n_clip[part])
        if self.data[part] is None:
            self.data[part] = DataHelper(self._name, self._im_size, self._path[part],
                                         self._n_clip[part], out_path=self._data_path,
                                         extension=self._extension)
        self.data[part].set_clip_idx(clip_idx)
        return self.data[part]._in_data_paths[clip_idx]

    # clip_results: sequence of anomaly scores (clips) for the whole test set
    # clip_results must be an array of arrays (either numpy or torch tensor)
    def evaluate(self, clip_results_raw, normalize_each_clip):
        assert self._name == "UCF_Crime" or len(clip_results_raw) == self._n_clip["test"]

        # manual evaluation for Subway datasets
        if self._name in ("Entrance", "Exit"):
            print("Manual evaluation is required due to the missing of frame-level groundtruth!")
            return None, None

        clip_results = copy.deepcopy(clip_results_raw)

        # set frame groundtruths
        if self._name in ("ShanghaiTech", "Belleview", "Train"):
            groundtruths = self._eval_groundtruth_frames
            assert len(clip_results) == len(groundtruths)
            assert all([len(clip_result) == len(groundtruth) for (clip_result, groundtruth) in zip(clip_results, groundtruths)])
        elif self._name in ("UCSDped2", "Avenue"):
            groundtruths = [np.zeros_like(clip_result) for clip_result in clip_results]
            # set frame-level groundtruth scores
            for idx, clip_idx in enumerate(self._eval_groundtruth_clips):
                anomaly_intervals = self._eval_groundtruth_frames[idx]
                for i in range(len(anomaly_intervals)//2):
                    # -1 because _eval_groundtruth_frames given in 1-based index
                    start = anomaly_intervals[2*i] - 1
                    end = anomaly_intervals[2*i+1] - 1
                    groundtruths[clip_idx][start:end] = 1
        elif self._name == "UCF_Crime":
            n_clip = self.get_n_clip("test")
            allowed_videos = sorted(list(self._eval_groundtruth_frames.keys()))
            video_lengths = {}
            count = 0
            for clip_idx in range(n_clip):
                file_path = self.load_data(clip_idx)
                _, file_name = os.path.split(file_path)
                if not any (x in file_name for x in allowed_videos):
                    continue
                key = [x for x in allowed_videos if x in file_name][0]
                video_lengths[key] = video_lengths.get(key, 0) + len(clip_results[count])
                count += 1
            #
            groundtruths = [np.zeros(video_lengths[key], dtype=int) for key in allowed_videos]
            # set frame-level groundtruth scores
            for idx, key in enumerate(allowed_videos):
                anomaly_intervals = self._eval_groundtruth_frames[key]
                for i in range(len(anomaly_intervals)//2):
                    # -1 because _eval_groundtruth_frames given in 1-based index
                    start = anomaly_intervals[2*i] - 1
                    end = anomaly_intervals[2*i+1] - 1
                    groundtruths[idx][start:end] = 1
        else:
            raise NotImplementedError("Unimplemented dataset (%s)" % self._name)

        if normalize_each_clip:
            for clip_idx in range(len(clip_results)):
                clip_results[clip_idx] /= max(clip_results[clip_idx])

        # flatten groundtruth and predicted scores for evaluation
        true_results = np.concatenate(groundtruths, axis=0)
        pred_results = np.concatenate(clip_results, axis=0)
        assert len(true_results) == len(pred_results)
        print("Ground truth: %d abnormal frames, %d normal frames" %
              (int(np.sum(true_results)), len(true_results) - int(np.sum(true_results))))

        # plot roc ROC curve
        # plot_ROC(true_results, pred_results)
        return roc_auc_score(true_results, pred_results), average_precision_score(true_results, pred_results)

    # set attributes related to each dataset
    def _set_dataset_attributes(self):
        if self._name in data_info:
            info = data_info[self._name]
            self._n_clip = {"train": info["n_clip_train"], "test": info["n_clip_test"]}
            self._extension = info["extension"]
            self._path = {"train": info["training_path"], "test": info["evaluation_path"]}
            self._eval_groundtruth_frames = info["eval_groundtruth_frames"]
            self._eval_groundtruth_clips = info["eval_groundtruth_clips"]
            #
            # print("Dataset info:")
            # print(self._n_clip)
            # print(self._extension)
            # print(self._path)
        else:
            raise ValueError("Unknown dataset")

        # specify frame-level groundtruth for some datasets
        if self._name == "Avenue":
            self._eval_groundtruth_frames = \
                load_groundtruth_Avenue(self._path["test"] + "/../testing_gt", self._n_clip["test"])
        elif self._name == "ShanghaiTech":
            self._eval_groundtruth_frames = \
                load_groundtruth_ShanghaiTech(self._path["test"] + "/../test_frame_mask", self._n_clip["test"])
        elif self._name in ("Belleview", "Train"):
            self._eval_groundtruth_frames = \
                load_groundtruth_Traffic_Train_and_Belleview(self._name, self._path["test"] + "/../GT")
        elif self._name == "UCF_Crime":
            groundtruth_file = info["groundtruth_path"] + "/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
            self._eval_groundtruth_frames = load_groundtruth_UCF_Crime(groundtruth_file)

        # done
        if self._name not in ("Entrance", "Exit", "UCF_Crime"):
            assert len(self._eval_groundtruth_clips) == len(self._eval_groundtruth_frames)

    def get_n_clip(self, part):
        assert part in ("train", "test")
        return self._n_clip[part]


# Modified from https://stackoverflow.com/questions/3160699/python-progress-bar
class ProgressBar(object):
    def __init__(self, total, width=80, use_ETA=True, symbol='#', output=sys.stderr):
        assert len(symbol) == 1
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        #
        self.use_ETA = use_ETA
        fmt = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go"
        if use_ETA:
            fmt += " -- time: %(elapsed)s/%(eta)s"
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)
        self.current = 0
        self.start_time = time.time()

    def _format_seconds(self, seconds):
        rounded = round(seconds)
        sec = str(rounded % 60).zfill(2)
        min = rounded // 60
        hour = str(min//60).zfill(2)
        min = str(min % 60).zfill(2)
        return "%s:%s:%s" % (hour, min, sec)

    def __call__(self, msg=''):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '|' + self.symbol * size + '.' * (self.width - size) + '|'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        # ETA
        if self.use_ETA:
            elapsed = time.time() - self.start_time
            per_item = elapsed / max(self.current, 1e-6)
            eta = remaining * per_item
            elapsed_str = self._format_seconds(elapsed)
            eta_str = self._format_seconds(eta)
            args['elapsed'] = elapsed_str
            args['eta'] = eta_str

        print('\r' + self.fmt % args + msg, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


# groundtruth-related functions

def load_groundtruth_Avenue(path, n_clip):

    # get two ends of anomalous events
    def find_ends(seq):
        tmp = np.insert(seq, 0, -10)
        diff = tmp[1:] - tmp[:-1]
        peaks = np.where(diff != 1)[0]
        #
        ret = np.empty((len(peaks), 2), dtype=int)
        for i in range(len(ret)):
            ret[i] = [peaks[i], (peaks[i+1]-1) if i < len(ret)-1 else (len(seq)-1)]
        return ret

    # get indices of two ends of anomalous events
    def get_segments(seq):
        ends = find_ends(seq)
        return np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(-1) + 1  # +1 for 1-based index (same as UCSD data)

    groundtruth = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (path, i + 1)
        # print(filename)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])
        abnormal_frames = np.where(n_bin > 0)[0]
        groundtruth.append(get_segments(abnormal_frames))
    return groundtruth


def load_groundtruth_ShanghaiTech(path, n_clip):
    groundtruth = []
    files = sorted(glob.glob("%s/*.npy" % path))
    assert len(files) == n_clip
    for i in range(n_clip):
        frame_labels = np.load(files[i])
        groundtruth.append(frame_labels)
    return groundtruth


def load_groundtruth_UCF_Crime(txt_file_path):
    with open(txt_file_path, 'r') as reader:
        lines = reader.read().split("\n")
    groundtruth = {}
    for line in lines:
        values = line.split()
        if len(values) < 4:
            continue
        key = values[0].replace("_x264.mp4", "")
        groundtruth[key] = [int(values[2]), int(values[3])]
        if int(values[4]) != -1:
            groundtruth[key] += [int(values[4]), int(values[5])]
    return groundtruth


def load_groundtruth_Traffic_Train_and_Belleview(dataset_name, path):
    assert dataset_name in ("Belleview", "Train")
    frames_range = (300, 2918) if dataset_name == "Belleview" else (13840, 18000)
    files = ["%s/gt-%s.png" % (path, str(i).zfill(5)) for i in range(frames_range[0], frames_range[1])]
    groundtruth = np.array([np.sum(cv2.imread(file)/255.) for file in files])
    groundtruth[groundtruth > 0.] = 1.
    assert len(np.unique(groundtruth)) == 2
    return [np.array(groundtruth)]
