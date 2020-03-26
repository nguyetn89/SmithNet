import os
import argparse
import torch
import numpy as np

# from WGAN import WGAN_GP
# from NoGan import NoGAN
from DCGAN import DCGAN
from CONFIG import data_info

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set init seed values
torch.manual_seed(2020)
np.random.seed(2020)


def print_params(params):
    print("dataset: %s | task: %s | epoch: %s" % (params["dataset"], params["task"], params["epoch"]))
    print("method: %s | workspace: %s" % (params["method"], params["workspace"]))
    print("height: %d | width: %d | batch: %d" % (params["height"], params["width"], params["batch"]))
    print("optical: %d | unet: %s | cross: %d" % (params["optical"], params["unet"], params["cross_pred"]))
    if params["task"] == "eval":
        print("power: %d | patch: %d | stride: %d" % (params["power"], params["patch"], params["stride"]))


def get_epoch_info(epoch_str):
    if '-' in epoch_str:
        vals = epoch_str.split('-')
        assert len(vals) == 2
        return int(vals[0]), int(vals[1])
    return 0, int(epoch_str)


# main execution
def run(params):
    # parse parameters
    dataset = params["dataset"]
    task = params["task"]
    method = params["method"]
    im_size = (params["height"], params["width"])
    store_path = params["workspace"]
    use_optical_flow = params["optical"] != 0
    use_UNET = params["unet"]
    use_cross_pred = params["cross_pred"] != 0
    use_progress_bar = params["progressbar"] != 0
    # init model controller
    if method == "WGAN":
        # controller = WGAN_GP(dataset, im_size, store_path)
        print("To be implemented...")
        return
    elif method == "NoGAN":
        # controller = NoGAN(dataset, im_size, store_path)
        print("To be implemented...")
        return
    else:
        controller = DCGAN(dataset, im_size, store_path,
                           use_optical_flow,
                           use_UNET,
                           use_cross_pred,
                           use_progress_bar=use_progress_bar)
    #
    if task == "train":
        print("========== Mode: Training ==========")
        epoch_start, epoch_end = get_epoch_info(params["epoch"])
        print("Epoch %d to %d" % (epoch_start, epoch_end))
        batch_size = params["batch"]
        n_epoch_to_save = params["every"]
        controller.train(epoch_start, epoch_end, batch_size, n_epoch_to_save)
    elif task == "infer":
        print("========== Mode: Inference ==========")
        _, epoch_eval = get_epoch_info(params["epoch"])
        print("Epoch %d" % epoch_eval)
        batch_size = params["batch"]
        controller.infer(epoch_eval, batch_size, data_set=params["subset"])
    elif task == "eval":
        print("========== Mode: Evaluation ==========")
        _, epoch_eval = get_epoch_info(params["epoch"])
        power = params["power"]
        patch_size = params["patch"]
        stride = params["stride"]
        print("Epoch %d" % epoch_eval)
        AUCs = controller.evaluate(epoch_eval, power, patch_size, stride)
        print("Epoch", epoch_eval)
        print("AUCs:", AUCs)
    else:
        print("Unknown task", task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--subset", type=str, default="test_set")
    parser.add_argument("--epoch", type=str, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--optical", type=int, default=1)
    parser.add_argument("--unet", type=str, default="none")
    parser.add_argument("--cross_pred", type=int, default=1)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--progressbar", type=int, default=0)
    parser.add_argument("--method", type=str, default="DCGAN")
    parser.add_argument("--workspace", type=str, default="./workspace_flow")
    parser.add_argument("--power", type=int, default=1)
    parser.add_argument("--patch", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--print", type=int, default=1)
    args = vars(parser.parse_args())
    # validate arguments
    assert args["dataset"] in data_info
    assert args["task"] in ["train", "infer", "eval"]
    assert args["subset"] in ["training_set", "test_set"]
    assert args["batch"] > 1
    assert args["method"] in ["DCGAN", "WGAN", "NoGAN"]
    #
    if args["print"] != 0:
        print_params(args)
    run(args)
