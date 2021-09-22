import numpy as np

loss_weights = {"g_loss": 0.25, "frame": 1., "flow": 2.}

common_path = "/project/def-jeandiro/nguyetn/datasets"

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
    "n_clip_train": 34,
    "n_clip_test": 36,
    "extension": ".npy",
    "training_path": "%s/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train" % common_path,
    "evaluation_path": "%s/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test" % common_path,
    "eval_groundtruth_frames": [(60, 152), (50, 175), (91, 200), (31, 168), (5, 90, 140, 200), (1, 100, 110, 200), (1, 175),
                                (1, 94), (1, 48), (1, 140), (70, 165), (130, 200), (1, 156), (1, 200), (138, 200), (123, 200),
                                (1, 47), (54, 120), (64, 138), (45, 175), (31, 200), (16, 107), (8, 165), (50, 171), (40, 135),
                                (77, 144), (10, 122), (105, 200), (1, 15, 45, 113), (175, 200), (1, 180), (1, 52, 65, 115), (5, 165),
                                (1, 121), (86, 200), (15, 108)],
    "eval_groundtruth_clips": np.arange(36)
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
    "n_clip_train": 12,
    "n_clip_test": 76,
    "extension": ".npy",
    "training_path": "%s/Entrance/train" % common_path,
    "evaluation_path": "%s/Entrance/test" % common_path,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

Exit = {
    "n_clip_train": 1,
    "n_clip_test": 30,
    "extension": ".npy",
    "training_path": "%s/Exit/train" % common_path,
    "evaluation_path": "%s/Exit/test" % common_path,
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

UCF_Crime = {
    "n_clip_train": 1806,   #1806 Beluga, 1806 Cedar
    "n_clip_test": 1328,
    "extension": ".npy",
    "training_path": "/scratch/nguyetn/UCF_Crime/tmp/anomaly_detection/clips/Train",
    "evaluation_path": "/scratch/nguyetn/UCF_Crime/tmp/anomaly_detection/clips/Test",
    "groundtruth_path": "/scratch/nguyetn/UCF_Crime/tmp/anomaly_detection/annotation",
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": np.arange(1328)
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
    "UCF_Crime": UCF_Crime,
    "Belleview": Belleview,
    "Train": Train
}
