import numpy as np

loss_weights = {"g_loss": 0.25, "context": 0.5, "reconst": 1, "instant": 1, "longterm": 1}

UCSDped2 = {
    "n_clip_train": 16,
    "n_clip_test": 12,
    "extension": ".npy",
    "training_path": "./datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train",
    "evaluation_path": "./datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test",
    "eval_groundtruth_frames": [(61, 180), (95, 180), (1, 146), (31, 180), (1, 129),
                                (1, 159), (46, 180), (1, 180), (1, 120), (1, 150), (1, 180), (88, 180)],
    "eval_groundtruth_clips": np.arange(12)
}

just4test = {
    "n_clip_train": 2,
    "n_clip_test": 2,
    "extension": ".npy",
    "training_path": "./datasets/just4test/train",
    "evaluation_path": "./datasets/just4test/test",
    "eval_groundtruth_frames": [(61, 180), (95, 180)],
    "eval_groundtruth_clips": np.arange(2)
}

UCSDped1 = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": "./datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train",
    "evaluation_path": "./datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test",
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

Avenue = {
    "n_clip_train": 16,
    "n_clip_test": 21,
    "extension": ".npy",
    "training_path": "./datasets/Avenue_Dataset/training_videos",
    "evaluation_path": "./datasets/Avenue_Dataset/testing_videos",
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

Shanghai = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": None,
    "evaluation_path": None,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
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
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": None,
    "evaluation_path": None,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

Train = {
    "n_clip_train": None,
    "n_clip_test": None,
    "extension": ".npy",
    "training_path": None,
    "evaluation_path": None,
    "eval_groundtruth_frames": None,
    "eval_groundtruth_clips": None
}

data_info = {
    "UCSDped2": UCSDped2,
    "just4test": just4test,
    "UCSDped1": UCSDped1,
    "Avenue": Avenue,
    "Entrance": Entrance,
    "Exit": Exit,
    "Shanghai": Shanghai,
    "Crime": Crime,
    "Belleview": Belleview,
    "Train": Train
}
