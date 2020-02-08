# import os
# import sys
import torch

from WGAN import WGAN_GP
from NoGan import NoGAN
from DCGAN import DCGAN


# set seed values
torch.manual_seed(2020)

# global variables
IM_SIZE = (128, 192)
IM_SIZE = (192, 288)


# method (string): WGAN or NoGAN
# dataset (string): name of dataset
# im_size (tuple): resolution of input frames
# mode (tuple): train, infer, and eval
def run(store_path, params, mode=("train", "infer", "eval")):
    assert isinstance(params, dict)
    assert isinstance(mode, (tuple, list))
    # parse parameters
    method = params["method"]
    assert method in ("WGAN", "NoGAN", "DCGAN")
    dataset = params["dataset"]
    im_size = params["im_size"]
    # init model
    if method == "WGAN":
        controller = WGAN_GP(dataset, im_size, store_path)
    elif method == "NoGAN":
        controller = NoGAN(dataset, im_size, store_path)
    else:
        controller = DCGAN(dataset, im_size, store_path)
    #
    if "train" in mode:
        epoch_start = params["epoch_start"] if "epoch_start" in params else 0            # default: start from 0
        epoch_end = params["epoch_end"] if "epoch_end" in params else (epoch_start + 2)  # default: train only 2 epochs
        batch_size = params["batch_size"] if "batch_size" in params else 16              # default: batch_size is 16
        n_epoch_to_save = params["every_epochs"] if "every_epochs" in params else 5      # default: save model after every 5 epochs
        controller.train(epoch_start, epoch_end, batch_size, n_epoch_to_save)
    #
    if "infer" in mode:
        if "epoch_eval" in params:
            epoch_eval = params["epoch_eval"]
            batch_size = params["batch_size"] if "batch_size" in params else 16              # default: batch_size is 16
            controller.infer(epoch_eval, batch_size, data_set="test_set")
        else:
            print("ERROR: epoch for inference not found => skip inference stage!")
    #
    if "eval" in mode:
        if "epoch_eval" in params:
            epoch_eval = params["epoch_eval"]
            AUCs = controller.evaluate(epoch_eval)
            print("Epoch", epoch_eval)
            print("AUCs:", AUCs)
        else:
            print("ERROR: epoch for evaluation not found => skip evaluation stage!")


def run_just4test(store_path):
    params = {
        "method": "WGAN",
        "dataset": "just4test",
        "im_size": (192, 288),
        "epoch_start": 0,
        "epoch_end": 1,
        "batch_size": 4,
        "every_epochs": 1,
        "epoch_eval": 1
    }
    mode = ("-train", "-infer", "eval")
    run(store_path, params, mode)


def run_UCSDped2(store_path):
    params = {
        "method": "DCGAN",
        "dataset": "UCSDped2",
        "im_size": (192, 288),
        "epoch_start": 0,
        "epoch_end": 80,
        "batch_size": 4,
        "every_epochs": 20,
        "epoch_eval": 80
    }
    mode = ("train", "infer", "eval")
    run(store_path, params, mode)


if __name__ == "__main__":
    run_UCSDped2("./workspace_DCGAN_context")
