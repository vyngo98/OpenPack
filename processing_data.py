from pathlib import Path

import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import openpack_toolkit as optk
from omegaconf import DictConfig, OmegaConf


class LoadData:
    def __init__(self, user_id, session_id, device_id, e4_device_id, openpack_version, dataset_rootdir):
        self.user_id = user_id
        self.session_id = session_id
        self.device_id = device_id  # atr device [atr01, atr02, atr03, atr04]
        self.e4_device_id = e4_device_id  # e4 device [e401, e402]
        self.openpack_version = openpack_version
        self.dataset_rootdir = dataset_rootdir

    @staticmethod
    def plot_openpack_operations(df: pd.DataFrame, xlim=None, figsize=(30, 7),
                                 OPENPACK_OPERATIONS=optk.OPENPACK_OPERATIONS):
        seq_len = len(df)

        df["cls_idx"] = optk.OPENPACK_OPERATIONS.convert_id_to_index(df["operation"])

        df_head = df.drop_duplicates(["user", "session", "box"], keep="first")
        df_tail = df.drop_duplicates(["user", "session", "box"], keep="last")
        df_box = pd.DataFrame({
            "box": df_head["box"],
            "start": df_head.index,
            "end": df_tail.index,
        }).reset_index(drop=True)

        # == Plot ==
        fig, ax0 = plt.subplots(1, 1, figsize=figsize)
        xloc = np.arange(seq_len)

        ax0.plot(xloc, df["cls_idx"], lw=3)
        for index, row in df_box.iterrows():
            ax0.fill_between([row.start, row.end], 0, 11, color=f"C{row.box % 10}", alpha=0.2)
            ax0.text(
                row.start, 11, f"Box{row.box:0=2}",
                fontweight="bold", color="black",
            )

        xticks = np.arange(0, seq_len, 60 * 2)
        xticks_minor = np.arange(0, seq_len, 30)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticks // 60)
        ax0.set_xticks(xticks_minor, minor=True)
        ax0.set_xlabel("Time [min]", fontweight="bold")
        if xlim is None:
            ax0.set_xlim([0, seq_len])
        else:
            ax0.set_xlim(xlim)

        yticklabels = [k for k in OPENPACK_OPERATIONS.get_ids()]
        ax0.set_yticks(np.arange(len(OPENPACK_OPERATIONS)))
        ax0.set_yticklabels(yticklabels)
        ax0.set_ylabel("Class ID")

        ax0.grid(True, which="minor", linestyle=":")

        ax0.set_title(f"OPENPACK OPERATIONS", fontsize="x-large", fontweight="bold")

        fig.tight_layout()
        fig.show()

    def get_annotation_data(self, cfg, visualize=False):
        # Set parameters to the config object.
        # NOTE: user.name is already defined above. See [2]
        cfg.dataset.annotation = optk.configs.datasets.annotations.ACTIVITY_1S_ANNOTATION
        cfg.session = self.session_id

        path = Path(
            cfg.dataset.annotation.path.dir,
            cfg.dataset.annotation.path.fname,
        )
        print(path)

        # Load CSV file
        df = pd.read_csv(path)

        if visualize:
            self.plot_openpack_operations(df, xlim=None, figsize=(30, 7), OPENPACK_OPERATIONS=optk.OPENPACK_OPERATIONS)
        return df

    @staticmethod
    def plot_atr_qags(df: pd.DataFrame, cfg: DictConfig):
        seq_len = len(df)

        fig = plt.figure(figsize=(30, 2.5 * 4))
        gs_master = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
        gs_line = GridSpecFromSubplotSpec(
            nrows=3, ncols=1, subplot_spec=gs_master[0], hspace=0.05)
        gs_hist = GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[1])

        data = [
            {
                "label": "Acc [G]",
                "cols": ["acc_x", "acc_y", "acc_z"],
                "lim": [-4.0, 4.0],
            },
            {
                "label": "Gyro [dps]",
                "cols": ["gyro_x", "gyro_y", "gyro_z"],
                "lim": [-500.0, 500.0],
            },
            {
                "label": "Quaternion",
                "cols": ["quat_w", "quat_x", "quat_y", "quat_z"],
                "lim": [-1.5, 1.5],
            },
        ]
        xloc = df.index.values
        for i, d in enumerate(data):
            cols = d["cols"]
            ylabel = d["label"]
            lim = d["lim"]

            X = df[cols].values.T

            # -- Sequence (Acc / Gyro / Quat) --
            ax0 = fig.add_subplot(gs_line[i])
            for ch, col_name in enumerate(cols):
                ax0.plot(xloc, X[ch], label=col_name, color=f"C{ch}", alpha=0.75)

            xticks = np.arange(0, seq_len + 1, 30 * 60 * 2)
            xticks_minor = np.arange(0, seq_len + 1, 30 * 30)
            ax0.set_xticks(xticks)
            ax0.set_xticklabels(xticks // (30 * 60))
            ax0.set_xticks(xticks_minor, minor=True)
            ax0.set_xlim([0, seq_len])

            ax0.set_ylabel(ylabel, fontweight="bold")

            ax0.grid(True, which="minor", linestyle=":")
            ax0.legend(loc="upper right")

            if i == 2:
                ax0.set_xlabel("Time [min]", fontweight="bold")
            else:
                ax0.tick_params(
                    labelbottom=False
                )

            # -- Histogram --
            ax1 = fig.add_subplot(gs_hist[i])
            for ch, col_name in enumerate(cols):
                ax1.hist(
                    X[ch],
                    range=lim,
                    bins=50,
                    label=col_name,
                    color=f"C{ch}",
                    alpha=0.50)

            ax1.set_xlabel(ylabel, fontweight="bold")
            ax1.set_ylabel("Freq", fontweight="bold")
            ax1.legend(loc="upper right")

        fig.suptitle(
            f"IMU - {cfg.device} | {cfg.user.name}-{cfg.session}",
            fontsize="x-large",
            fontweight="black")
        fig.tight_layout()
        fig.show()

    def get_imu_data(self, cfg, visualize=False):
        # Set parameters to the config object.
        cfg.dataset.stream = optk.configs.datasets.streams.ATR_QAGS_STREAM
        cfg.session = self.session_id
        cfg.device = self.device_id

        path = Path(
            cfg.dataset.stream.path.dir,
            cfg.dataset.stream.path.fname,
        )
        print(path)

        df = pd.read_csv(path)

        if visualize:
            self.plot_atr_qags(df, cfg)
        return df

    @staticmethod
    def plot_kinect_2d_kpt(annots: dict, cfg: DictConfig):
        FS = 15
        seq_len = len(annots)

        # == Plot ==
        sns.set("notebook", "whitegrid")
        fig = plt.figure(figsize=(30, 2.5 * 3))
        gs_master = GridSpec(nrows=3, ncols=1)

        # -- Keypoints Location --
        data = [
            {
                "node": "nose",
                "idx": 0,
            },
            {
                "node": "Left Shoulder",
                "idx": 5,
            },
        ]
        xloc = np.arange(len(annots))
        for i, d in enumerate(data):
            title = d["node"]
            kpt_idx = d["idx"]

            X = np.array([annots[i]["keypoints"] for i in range(len(annots))])

            # -- Sequence (Acc / Gyro / Quat) --
            ax0 = fig.add_subplot(gs_master[i])
            ax1 = ax0.twinx()

            # prediction score
            ax1.fill_between(xloc, X[:, kpt_idx, 2], label="Score", color="C0", alpha=0.1)
            # Plot position
            for ch in range(2):
                ax0.plot(
                    xloc,
                    X[:, kpt_idx, ch],
                    label="X-axis" if ch == 0 else "Y-axis",
                    color=f"C{ch}",
                    alpha=0.75,
                )

            xticks = np.arange(0, seq_len + 1, FS * 60 * 2)
            xticks_minor = np.arange(0, seq_len + 1, FS * 30)
            ax0.set_xticks(xticks)
            ax0.set_xticklabels(xticks // (FS * 60))
            ax0.set_xticks(xticks_minor, minor=True)
            ax0.set_xlim([0, seq_len])
            ax0.set_ylabel("Position [px]", fontweight="bold")

            ax1.set_yticks(np.arange(0, 1.1, 0.2))
            ax1.set_ylabel("Score", fontweight="bold")

            ax0.grid(True, which="minor", linestyle=":")
            ax0.legend(loc="upper left")
            ax1.legend(loc="upper right")
            ax0.set_title(f"{title} [IDX={kpt_idx}]", fontweight="bold", fontsize="x-large")

            if i == 1:
                ax0.set_xlabel("Time [min]", fontweight="bold")
            else:
                ax0.tick_params(
                    labelbottom=False
                )

        # -- [2] Tracking Id --
        ax0 = fig.add_subplot(gs_master[2])
        X = np.array([annots[i]["track_id"] for i in range(len(annots))])
        ax0.scatter(xloc, X, color="C0", alpha=0.75, s=5)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticks // (FS * 60))
        ax0.set_xticks(xticks_minor, minor=True)
        ax0.set_xlim([0, seq_len])
        ax0.set_ylabel("Track ID", fontweight="bold")
        ax0.grid(True, which="minor", linestyle=":")

        fig.suptitle(
            f"Kinect 2d-kpt | {cfg.user.name}-{cfg.session}",
            fontsize="x-large",
            fontweight="black")
        fig.tight_layout()
        fig.show()

    def get_kinect_data(self, cfg, visualize=False):
        # Set parameters to the config object.
        cfg.dataset.stream = optk.configs.datasets.streams.KINECT_2D_KPT_STREAM
        cfg.session = self.session_id

        path = Path(
            cfg.dataset.stream.path.dir,
            cfg.dataset.stream.path.fname,
        )
        print(path)

        # Load JSON file
        with open(path, "r") as f:
            data = json.load(f)

        if visualize:
            self.plot_kinect_2d_kpt(data["annotations"], cfg)
        return data

    @staticmethod
    def plot_e4_all(
            data: dict,
            user: str,
            session: str,
            device: str,
            version=None,
            xlim=None,
            figsize=(30, 7)):

        E4_SENSORS = ["acc", "bvp", "eda", "temp"]

        # == Plot ==
        fig = plt.figure(figsize=(30, 3 * 5))
        gs_master = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])
        gs_line = GridSpecFromSubplotSpec(
            nrows=4, ncols=1, subplot_spec=gs_master[0], hspace=0.05)
        gs_hist = GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=gs_master[1])

        metadata = {
            "acc": {
                "label": "Acc [G]",
                "cols": ["acc_x", "acc_y", "acc_z"],
                "lim": [-2.0, 2.0],
                "fs": 32,
            },
            "bvp": {
                "label": "BVP [mmHg]",
                "cols": ["bvp"],
                "lim": [-2000.0, 2000.0],
                "fs": 64,
            },
            "eda": {
                "label": "EDA\n[microsiemens]",
                "cols": ["eda"],
                "lim": [0.0, 25.0],
                "fs": 4,
            },
            "temp": {
                "label": "Temp [°C]",
                "cols": ["temp"],
                "lim": [30.0, 40.0],
                "fs": 4,
            },
        }
        # xloc = df.index.values
        for i, sensor in enumerate(E4_SENSORS):
            d = metadata[sensor]

            df = data[sensor]
            cols = d["cols"]
            ylabel = d["label"]
            lim = d["lim"]
            fs = d["fs"]

            X = df[cols].values.T
            xloc = df.index.values
            seq_len = len(xloc)

            # -- Sequence (Acc / Gyro / Quat) --
            ax0 = fig.add_subplot(gs_line[i])
            for ch, col_name in enumerate(cols):
                ax0.plot(xloc, X[ch], label=col_name, color=f"C{ch}", alpha=0.75)

            xticks = np.arange(0, seq_len + 1, fs * 60)
            xticks_minor = np.arange(0, seq_len + 1, fs * 30)
            ax0.set_xticks(xticks)
            ax0.set_xticklabels(xticks // (fs * 60))
            ax0.set_xticks(xticks_minor, minor=True)
            ax0.set_xlim([0, seq_len])

            ax0.set_ylabel(ylabel, fontweight="bold")
            ax0.grid(True, which="minor", linestyle=":")
            ax0.legend(loc="upper right")

            if i == 3:
                ax0.set_xlabel("Time [min]", fontweight="bold")
            else:
                ax0.tick_params(
                    labelbottom=False
                )

            # -- Histgram --
            ax1 = fig.add_subplot(gs_hist[i])
            for ch, col_name in enumerate(cols):
                ax1.hist(
                    X[ch],
                    range=lim,
                    bins=50,
                    label=col_name,
                    color=f"C{ch}",
                    alpha=0.50)

            ax1.set_xlabel(ylabel, fontweight="bold")
            ax1.set_ylabel("Freq", fontweight="bold")
            ax1.legend(loc="upper right")

        fig.suptitle(
            f"E4 - {device} | {user}-{session}",
            fontsize="x-large",
            fontweight="black")
        fig.tight_layout()
        fig.show()

    def get_e4_data(self, cfg, visualize=False):
        # region E4 ACC
        # Set parameters to the config object.
        cfg.dataset.stream = optk.configs.datasets.streams.E4_ACC_STREAM
        # cfg.user = optk.configs.users.U0101
        cfg.session = self.session_id
        cfg.device = self.e4_device_id

        path = Path(
            cfg.dataset.stream.path.dir,
            cfg.dataset.stream.path.fname,
        )
        print(path)

        # Load CSV file
        df_e4_acc = pd.read_csv(path)
        # endregion

        # region E4 BVP
        # Set parameters to the config object.
        cfg.dataset.stream = optk.configs.datasets.streams.E4_BVP_STREAM
        # cfg.user = optk.configs.users.U0101
        # cfg.session = "S0100"
        # cfg.device = "e401"

        path = Path(
            cfg.dataset.stream.path.dir,
            cfg.dataset.stream.path.fname,
        )
        print(path)
        # Load CSV file
        df_e4_bvp = pd.read_csv(path)
        # endregion

        # region E4 EDA
        cfg.dataset.stream = optk.configs.datasets.streams.E4_EDA_STREAM
        path = Path(
            cfg.dataset.stream.path.dir,
            cfg.dataset.stream.path.fname,
        )
        print(path)
        # Load CSV file
        df_e4_eda = pd.read_csv(path)
        # endregion

        # region E4 TEMP
        # Set parameters to the config object.
        cfg.dataset.stream = optk.configs.datasets.streams.E4_TEMP_STREAM
        path = Path(
            cfg.dataset.stream.path.dir,
            cfg.dataset.stream.path.fname,
        )
        print(path)
        # Load CSV file
        df_e4_temp = pd.read_csv(path)
        # endregion

        data = {
            "acc": df_e4_acc,
            "bvp": df_e4_bvp,
            "eda": df_e4_eda,
            "temp": df_e4_temp,
        }

        if visualize:
            self.plot_e4_all(data, cfg.user.name, cfg.session, cfg.device)
        return data

    def process(self):
        # define root config subject
        cfg = OmegaConf.create({
            "user": getattr(optk.configs.users, self.user_id),
            "session": None,
            "path": {
                "openpack": {
                    "version": self.openpack_version,
                    "rootdir": self.dataset_rootdir + "/openpack/${.version}",
                },
            },
            "dataset": {
                "annotation": None,
                "stream": None,
            }
        })

        all_data = []

        # read annotation data
        # df_ann_data = self.get_annotation_data(cfg, visualize=True)
        df_ann_data = self.get_annotation_data(cfg)
        all_data.append(df_ann_data)

        # read acc, gyro, quaternion from IMU
        df_imu_data = self.get_imu_data(cfg)
        all_data.append(df_imu_data)

        # read kinect 2d keypoint data
        # js_kinect_data = self.get_kinect_data(cfg)
        # all_data.append(js_kinect_data)

        # read data from E4 (Acceratation, BVP, EDA, and Temperature; e4-XXX)
        # e4_data = self.get_e4_data(cfg)
        # all_data.append(e4_data)

        return all_data





