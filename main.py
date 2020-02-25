import os
import argparse
# import sys
import torch
import numpy as np

# from WGAN import WGAN_GP
# from NoGan import NoGAN
from DCGAN import DCGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set init seed values
torch.manual_seed(2020)
np.random.seed(2020)
# torch.backends.cudnn.benchmark = False


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
        # controller = WGAN_GP(dataset, im_size, store_path)
        return
    elif method == "NoGAN":
        # controller = NoGAN(dataset, im_size, store_path)
        return
    else:
        controller = DCGAN(dataset, im_size, store_path, use_optical_flow=True)
    #
    if "train" in mode:
        print("========== Mode: Training ==========")
        epoch_start = params["epoch_start"] if "epoch_start" in params else 0            # default: start from 0
        epoch_end = params["epoch_end"] if "epoch_end" in params else (epoch_start + 2)  # default: train only 2 epochs
        batch_size = params["batch_size"] if "batch_size" in params else 16              # default: batch_size is 16
        n_epoch_to_save = params["every_epochs"] if "every_epochs" in params else 5      # default: save model after every 5 epochs
        controller.train(epoch_start, epoch_end, batch_size, n_epoch_to_save)
    #
    if "infer" in mode:
        print("========== Mode: Inference ==========")
        if "epoch_eval" in params:
            epoch_eval = params["epoch_eval"]
            batch_size = params["batch_size"] if "batch_size" in params else 16              # default: batch_size is 16
            controller.infer(epoch_eval, batch_size, data_set="test_set")
        else:
            print("ERROR: epoch for inference not found => skip inference stage!")
    #
    if "eval" in mode:
        print("========== Mode: Evaluation ==========")
        if "epoch_eval" in params:
            epoch_eval = params["epoch_eval"]
            AUCs = controller.evaluate(epoch_eval)
            print("Epoch", epoch_eval)
            print("AUCs:", AUCs)
        else:
            print("ERROR: epoch for evaluation not found => skip evaluation stage!")


def run_just4test(store_path, mode):
    params = {
        "method": "DCGAN",
        "dataset": "just4test",
        "im_size": (128, 192),
        "epoch_start": 0,
        "epoch_end": 2,
        "batch_size": 4,
        "every_epochs": 1,
        "epoch_eval": 2
    }
    # mode = ("-train", "infer", "eval")
    run(store_path, params, mode)


def run_UCSDped2(store_path, mode):
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
    # mode = ("train", "infer", "eval")
    run(store_path, params, mode)


def run_Avenue(store_path, mode):
    params = {
        "method": "DCGAN",
        "dataset": "Avenue",
        "im_size": (128, 192),
        "epoch_start": 0,
        "epoch_end": 40,
        "batch_size": 4,
        "every_epochs": 10,
        "epoch_eval": 40
    }
    # mode = ("train", "infer", "eval")
    run(store_path, params, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=int, default=0)
    parser.add_argument("--inference", type=int, default=0)
    parser.add_argument("--evaluation", type=int, default=0)
    args = parser.parse_args()
    mode = ("-train" if args.training == 0 else "train",
            "-infer" if args.inference == 0 else "infer",
            "-eval" if args.evaluation == 0 else "eval")
    # run_UCSDped2("./workspace_DCGAN_context")
    # run_Avenue("./workspace_DCGAN_context_fixedNet")
    run_just4test("./workspace_flow", mode)
