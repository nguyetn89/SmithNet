import numpy as np

loss_weights = {"g_loss": 0.25, "frame": 1., "flow": 2.}

common_path = "./datasets"

UCSDped2 = {
    "n_clip_train": 16,
    "n_clip_test": 12,
    "extension": ".npy",
    "training_path": "%s/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train" % common_path,
    "evaluation_path": "%s/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test" % common_path,
    "eval_groundtruth_frames": [(61, 180), (95, 180), (1, 146), (31, 180), (1, 129),
                                (1, 159), (46, 180), (1, 180), (1, 120), (1, 150), (1, 180), (88, 180)],
    "eval_groundtruth_clips": np.arange(12)
}

just4test = {
    "n_clip_train": 2,
    "n_clip_test": 2,
    "extension": ".npy",
    "training_path": "%s/just4test/train" % common_path,
    "evaluation_path": "%s/just4test/test" % common_path,
    "eval_groundtruth_frames": [(61, 180), (95, 180)],
    "eval_groundtruth_clips": np.arange(2)
}

UCSDped1 = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": "%s/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train" % common_path,
    "evaluation_path": "%s/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test" % common_path,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

Avenue = {
    "n_clip_train": 16,
    "n_clip_test": 21,
    "extension": ".npy",
    "training_path": "%s/Avenue_Dataset/training_videos" % common_path,
    "evaluation_path": "%s/Avenue_Dataset/testing_videos" % common_path,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": np.arange(21)
}

Entrance = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": None,
    "evaluation_path": None,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

Exit = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": None,
    "evaluation_path": None,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

ShanghaiTech = {
    "n_clip_train": 330,
    "n_clip_test": 107,
    "extension": ".npy",
    "training_path": "%s/shanghaitech/training/videos" % common_path,
    "evaluation_path": "%s/shanghaitech/testing/frames" % common_path,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": np.arange(107)
}

Crime = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": None,
    "evaluation_path": None,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

Belleview = {
    "n_clip_train": 1,
    "n_clip_test": 1,
    "extension": ".npy",
    "training_path": "%s/Traffic-Belleview/train" % common_path,
    "evaluation_path": "%s/Traffic-Belleview/test" % common_path,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": [0]
}

Train = {
    "n_clip_train": 1,
    "n_clip_test": 1,
    "extension": ".npy",
    "training_path": "%s/Traffic-Train/train" % common_path,
    "evaluation_path": "%s/Traffic-Train/test" % common_path,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": [0]
}

data_info = {
    "UCSDped2": UCSDped2,
    "just4test": just4test,
    "UCSDped1": UCSDped1,
    "Avenue": Avenue,
    "Entrance": Entrance,
    "Exit": Exit,
    "ShanghaiTech": ShanghaiTech,
    "Crime": Crime,
    "Belleview": Belleview,
    "Train": Train
}
