from pathlib import Path

import numpy as np
import pandas as pd
import json
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import openpack_toolkit as optk
from omegaconf import DictConfig, OmegaConf
from wfdb.processing import resample_sig

# import tfrecord
import define


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
        key_points = np.array([data['annotations'][i]["keypoints"] for i in range(len(data['annotations']))])

        if visualize:
            self.plot_kinect_2d_kpt(data["annotations"], cfg)
        return key_points

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
        js_kinect_data = self.get_kinect_data(cfg)
        all_data.append(js_kinect_data)

        # read data from E4 (Acceratation, BVP, EDA, and Temperature; e4-XXX)
        e4_data = self.get_e4_data(cfg)
        all_data.append(e4_data)

        return all_data


class LoadDataMultiDevice:
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

    @staticmethod
    def calc_fusion_val(df):
        ava = np.abs(df['acc_x']*np.sin(df['gyro_z']) + df['acc_y'] + np.sin(df['gyro_y']) - df['acc_z']*np.cos(df['gyro_y'])*np.cos(df['gyro_z']))
        df['ava'] = ava
        mag_acc = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['mag_acc'] = mag_acc
        mag_gyro = np.sqrt(df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2)
        df['mag_gyro'] = mag_gyro
        # region complementary filter
        theta = np.zeros(len(df['acc_x']))
        for i in range(len(theta)):
            if i == 0:
                theta_new = 0.98*1/define.FS_TARGET + 0.02*np.arctan2(df['acc_x'][i], df['acc_y'][i])
            else:
                theta_new = 0.98*(theta[i-1] + 1/define.FS_TARGET) + 0.02*np.arctan2(df['acc_x'][i], df['acc_y'][i])
            theta[i] = theta_new
        df['theta'] = theta
        # endregion
        return df

    def get_imu_data(self, cfg, visualize=False):
        # Set parameters to the config object.
        cfg.dataset.stream = optk.configs.datasets.streams.ATR_QAGS_STREAM
        cfg.session = self.session_id
        concat_df = []
        for e, each_atr_device in enumerate(self.device_id):
            cfg.device = each_atr_device

            path = Path(
                cfg.dataset.stream.path.dir,
                cfg.dataset.stream.path.fname,
            )
            print(path)

            df = pd.read_csv(path)
            df = self.calc_fusion_val(df)
            df.columns = [str(col) + '_{}'.format(each_atr_device) if col != 'unixtime' else str(col) for col in
                          df.columns]
            if e == 0:
                concat_df = df
            else:
                df = df.drop(columns=['unixtime'])
                concat_df = pd.concat([concat_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

            if visualize:
                self.plot_atr_qags(df, cfg)
        return concat_df

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
        key_points = np.array([data['annotations'][i]["keypoints"] for i in range(len(data['annotations']))])

        if visualize:
            self.plot_kinect_2d_kpt(data["annotations"], cfg)
        return key_points

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
        cfg.session = self.session_id
        concat_e4_acc, concat_e4_bvp, concat_e4_eda, concat_e4_temp = [], [], [], []
        for e, each_e4_device in enumerate(self.e4_device_id):
            cfg.device = each_e4_device

            # region E4 ACC
            # Set parameters to the config object.
            cfg.dataset.stream = optk.configs.datasets.streams.E4_ACC_STREAM
            path = Path(
                cfg.dataset.stream.path.dir,
                cfg.dataset.stream.path.fname,
            )
            print(path)

            # Load CSV file
            df_e4_acc = pd.read_csv(path)
            df_e4_acc.columns = [str(col) + '_{}'.format(each_e4_device) if col != 'time' else str(col)
                                 for col in df_e4_acc.columns]
            # endregion

            # region E4 BVP
            # Set parameters to the config object.
            cfg.dataset.stream = optk.configs.datasets.streams.E4_BVP_STREAM

            path = Path(
                cfg.dataset.stream.path.dir,
                cfg.dataset.stream.path.fname,
            )
            print(path)
            # Load CSV file
            df_e4_bvp = pd.read_csv(path)
            df_e4_bvp.columns = [str(col) + '_{}'.format(each_e4_device) if col != 'time' else str(col)
                                 for col in df_e4_bvp.columns]
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
            df_e4_eda.columns = [str(col) + '_{}'.format(each_e4_device) if col != 'time' else str(col)
                                 for col in df_e4_eda.columns]
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
            df_e4_temp.columns = [str(col) + '_{}'.format(each_e4_device) if col != 'time' else str(col)
                                  for col in df_e4_temp.columns]
            # endregion

            if e == 0:
                concat_e4_acc = df_e4_acc
                concat_e4_bvp = df_e4_bvp
                concat_e4_eda = df_e4_eda
                concat_e4_temp = df_e4_temp
            else:
                df_e4_acc = df_e4_acc.drop(columns=['time'])
                df_e4_bvp = df_e4_bvp.drop(columns=['time'])
                df_e4_eda = df_e4_eda.drop(columns=['time'])
                df_e4_temp = df_e4_temp.drop(columns=['time'])
                concat_e4_acc = pd.concat([concat_e4_acc.reset_index(drop=True), df_e4_acc.reset_index(drop=True)],
                                          axis=1)
                concat_e4_bvp = pd.concat([concat_e4_bvp.reset_index(drop=True), df_e4_bvp.reset_index(drop=True)],
                                          axis=1)
                concat_e4_eda = pd.concat([concat_e4_eda.reset_index(drop=True), df_e4_eda.reset_index(drop=True)],
                                          axis=1)
                concat_e4_temp = pd.concat([concat_e4_temp.reset_index(drop=True), df_e4_temp.reset_index(drop=True)],
                                           axis=1)

            data = {
                "acc": df_e4_acc,
                "bvp": df_e4_bvp,
                "eda": df_e4_eda,
                "temp": df_e4_temp,
            }

            if visualize:
                self.plot_e4_all(data, cfg.user.name, cfg.session, cfg.device)

        concat_data = {
            "acc": concat_e4_acc,
            "bvp": concat_e4_bvp,
            "eda": concat_e4_eda,
            "temp": concat_e4_temp,
        }
        return concat_data

    def process_multi_device(self):
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
        js_kinect_data = self.get_kinect_data(cfg)
        all_data.append(js_kinect_data)

        # read data from E4 (Acceratation, BVP, EDA, and Temperature; e4-XXX)
        e4_data = self.get_e4_data(cfg)
        all_data.append(e4_data)

        return all_data

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
        js_kinect_data = self.get_kinect_data(cfg)
        all_data.append(js_kinect_data)

        # read data from E4 (Acceratation, BVP, EDA, and Temperature; e4-XXX)
        e4_data = self.get_e4_data(cfg)
        all_data.append(e4_data)

        return all_data


class ProcessData:
    def __init__(self, user_id, session_id, device_id, e4_device_id, all_data, tfrecord_path):
        self.user_id = user_id
        self.session_id = session_id
        self.device_id = device_id
        self.e4_device_id = e4_device_id
        self.all_data = all_data
        self.tfrecord_path = tfrecord_path

    # def extract_feature(data):
    #     mean_ft = np.array(data.mean())[1:]
    #     std_ft = np.array(data.std())[1:]
    #     max_ft = np.array(data.max())[1:]
    #     min_ft = np.array(data.min())[1:]
    #     var_ft = np.array(data.var())[1:]
    #     features = np.array([mean_ft, std_ft, max_ft, min_ft, var_ft]).T.flatten()
    #     return features

    @staticmethod
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:][:10]

    @staticmethod
    def shannon_entropy(x):
        pd_series = pd.Series(x)
        counts = pd_series.value_counts()
        entropy = scipy.stats.entropy(counts)

        return entropy

    @staticmethod
    def cal_dominant_freq_ratio(data, fs):
        fourier = np.fft.fft(data)
        frequencies = np.fft.fftfreq(np.shape(data)[0], d=1 / fs)
        # positive_frequencies = frequencies[np.where(frequencies >= 0)]
        magnitudes = abs(fourier[np.where(frequencies >= 0)])

        peak_magnitude = np.max(magnitudes, axis=0)
        dominant_freq_ratio = peak_magnitude / sum(magnitudes)

        energy = sum(magnitudes ** 2)
        return dominant_freq_ratio, energy

    def extract_feature(self, data, fs):
        mean_ft = np.mean(data, axis=0)
        std_ft = np.std(data, axis=0)
        max_ft = np.max(data, axis=0)
        min_ft = np.min(data, axis=0)
        var_ft = np.var(data, axis=0)
        med_ft = np.median(data, axis=0)
        sum_ft = np.sum(data, axis=0)
        import scipy
        skew = scipy.stats.skew(data, axis=0)
        kurtosis = scipy.stats.kurtosis(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        iqr = q75 - q25
        rms = np.sqrt(np.mean(data ** 2, axis=0))
        dominant_freq_ratio, energy = self.cal_dominant_freq_ratio(data, fs)
        # autocorrelation = np.array([self.autocorr(data[:, x]) for x in range(np.shape(data)[1])]).T
        shannon_entropy = np.array([self.shannon_entropy(data[:, x]) for x in range(np.shape(data)[1])])
        features = np.array(
            [mean_ft, std_ft, max_ft, min_ft, var_ft, med_ft, sum_ft, skew, kurtosis, q25, q75, iqr, shannon_entropy,
             rms, dominant_freq_ratio, energy]).T.flatten()
        # features = np.concatenate([features, autocorrelation], axis=0).T.flatten()
        features = np.nan_to_num(features)

        # freqs = np.fft.fftfreq(data.size, time_step)
        # import pywt
        # pywt.wavedec(data[:, 0], 'db1', level=2)

        return features

    @staticmethod
    def segment(data, max_time, sub_window_size, stride_size):
        sub_windows = np.arange(sub_window_size)[None, :] + np.arange(0, max_time, stride_size)[:, None]

        row, col = np.where(sub_windows >= max_time)
        uniq_row = len(np.unique(row))

        if uniq_row > 0 and row[0] > 0:
            sub_windows = sub_windows[:-uniq_row, :]

        return data[sub_windows]

    @staticmethod
    def cal_angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        # ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        # return ang + 360 if ang < 0 else ang
        return angle

    @staticmethod
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def itpl_nan(self, y):
        nans, x = self.nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    def extract_joint_angles(self, kp_data):
        left_hip_knee = np.asarray([self.cal_angle(kp_data[i, 13, :], kp_data[i, 11, :], kp_data[i, 12, :])
                                    for i in range(len(kp_data))])
        left_hip_knee = self.itpl_nan(left_hip_knee)
        right_hip_knee = np.asarray([self.cal_angle(kp_data[i, 14, :], kp_data[i, 12, :], kp_data[i, 11, :])
                                     for i in range(len(kp_data))])
        right_hip_knee = self.itpl_nan(right_hip_knee)
        left_knee_ankle = np.asarray([self.cal_angle(kp_data[i, 11, :], kp_data[i, 13, :], kp_data[i, 15, :])
                                      for i in range(len(kp_data))])
        left_knee_ankle = self.itpl_nan(left_knee_ankle)
        right_knee_ankle = np.asarray([self.cal_angle(kp_data[i, 12, :], kp_data[i, 14, :], kp_data[i, 16, :])
                                       for i in range(len(kp_data))])
        right_knee_ankle = self.itpl_nan(right_knee_ankle)
        left_elbow_shoulder_hip = np.asarray([self.cal_angle(kp_data[i, 7, :], kp_data[i, 5, :], kp_data[i, 11, :])
                                              for i in range(len(kp_data))])
        left_elbow_shoulder_hip = self.itpl_nan(left_elbow_shoulder_hip)
        right_elbow_shoulder_hip = np.asarray([self.cal_angle(kp_data[i, 8, :], kp_data[i, 6, :], kp_data[i, 12, :])
                                               for i in range(len(kp_data))])
        right_elbow_shoulder_hip = self.itpl_nan(right_elbow_shoulder_hip)
        left_wrist_elbow_shoulder = np.asarray([self.cal_angle(kp_data[i, 9, :], kp_data[i, 7, :], kp_data[i, 5, :])
                                                for i in range(len(kp_data))])
        left_wrist_elbow_shoulder = self.itpl_nan(left_wrist_elbow_shoulder)
        right_wrist_elbow_shoulder = np.asarray(
            [self.cal_angle(kp_data[i, 10, :], kp_data[i, 8, :], kp_data[i, 6, :])
             for i in range(len(kp_data))])
        right_wrist_elbow_shoulder = self.itpl_nan(right_wrist_elbow_shoulder)
        joint_angles = np.array(
            [left_hip_knee, right_hip_knee, left_knee_ankle, right_knee_ankle, left_elbow_shoulder_hip,
             right_elbow_shoulder_hip, left_wrist_elbow_shoulder, right_wrist_elbow_shoulder]).T
        return joint_angles

    def process(self, write_tfrecord=True):
        annotation = self.all_data[0]
        data = self.all_data[1]
        e4_data = self.all_data[-1]
        kp_data = self.all_data[-2]

        # sort data based on timestamp
        annotation.sort_values('unixtime')
        data.sort_values('unixtime')

        # drop duplicate
        annotation = annotation.drop_duplicates()
        data = data.drop_duplicates()

        # interpolate missing values
        data_itpl = data.interpolate()
        data_itpl = data_itpl.fillna(0)
        data_itpl = data_itpl.sort_values(by=['unixtime'])

        annotation_itpl = annotation.interpolate()
        annotation_itpl = annotation_itpl.fillna(0)
        annotation_itpl = annotation_itpl.sort_values(by=['unixtime'])

        annotation_itpl["cls_idx"] = optk.OPENPACK_OPERATIONS.convert_id_to_index(annotation_itpl["operation"])
        annotation_itpl = annotation_itpl.sort_values(by=['unixtime'])

        filt_data_itpl = data_itpl[(data_itpl['unixtime'] >= annotation_itpl['unixtime'].iloc[0]) &
                                   (data_itpl['unixtime'] < annotation_itpl['unixtime'].iloc[
                                       -1] + define.ONE_SECOND_IN_MILISECOND)]

        filt_data_itpl = filt_data_itpl.sort_values(by=['unixtime'])
        filt_data_itpl = np.array(filt_data_itpl)[:, 1:]

        # region Segmentation
        resamp_data = np.asarray([])
        for ch in range(np.shape(filt_data_itpl)[1]):
            resamp_sig, _ = resample_sig(filt_data_itpl[:, ch], define.FS_ORG, define.FS_TARGET)
            if len(resamp_data) == 0:
                resamp_data = resamp_sig.reshape(-1, 1)
            else:
                resamp_data = np.concatenate((resamp_data, resamp_sig.reshape(-1, 1)), axis=1)

        data_seg = self.segment(resamp_data, max_time=len(resamp_data), sub_window_size=define.WINDOW_SIZE * define.FS_TARGET,
                                stride_size=(define.WINDOW_SIZE - define.OVERLAP) * define.FS_TARGET)

        label_seg = self.segment(np.array(annotation_itpl["cls_idx"]),
                                 max_time=len(np.array(annotation_itpl["cls_idx"])),
                                 sub_window_size=define.WINDOW_SIZE, stride_size=(define.WINDOW_SIZE - define.OVERLAP))

        # label_time_seg = self.segment(np.array(annotation_itpl["unixtime"]),
        #                          max_time=len(np.array(annotation_itpl["cls_idx"])),
        #                          sub_window_size=df.WINDOW_SIZE, stride_size=(df.WINDOW_SIZE - df.OVERLAP))

        feature_seg = [self.extract_feature(data_seg[i], define.FS_TARGET) for i in range(len(data_seg))]

        # region kinect point data
        kp_data = kp_data[:, :, :2]
        filt_kp_data = kp_data[: (len(annotation) * define.FS_KEYPOINT)]

        joint_angles = self.extract_joint_angles(filt_kp_data)
        joint_angles_seg = self.segment(joint_angles, max_time=len(filt_kp_data),
                                        sub_window_size=define.WINDOW_SIZE * define.FS_KEYPOINT,
                                        stride_size=(define.WINDOW_SIZE - define.OVERLAP) * define.FS_KEYPOINT)
        feature_joint_angles_seg = [self.extract_feature(joint_angles_seg[i], define.FS_KEYPOINT) for i in
                                    range(len(joint_angles_seg))]

        data_kp_seg = self.segment(filt_kp_data.reshape(filt_kp_data.shape[0], -1), max_time=len(filt_kp_data),
                                   sub_window_size=define.WINDOW_SIZE * define.FS_KEYPOINT,
                                   stride_size=(define.WINDOW_SIZE - define.OVERLAP) * define.FS_KEYPOINT)
        feature_kp_seg = [self.extract_feature(data_kp_seg[i], define.FS_KEYPOINT) for i in range(len(data_kp_seg))]

        # combine eda and temp into a dataframe
        eda = e4_data['eda']
        eda = pd.concat([eda.reset_index(drop=True), e4_data['temp'].drop(columns=['time']).reset_index(drop=True)],
                        axis=1)
        filt_eda = eda[(eda['time'] >= annotation_itpl['unixtime'].iloc[0]) &
                       (eda['time'] < annotation_itpl['unixtime'].iloc[
                           -1] + define.ONE_SECOND_IN_MILISECOND)]

        filt_eda = filt_eda.sort_values(by=['time'])
        filt_eda = np.array(filt_eda)[:, 1:]

        # resamp_eda = np.asarray([])
        # for ch in range(np.shape(filt_eda)[1]):
        #     resamp_sig, _ = resample_sig(filt_eda[:, ch], 3.9, df.FS_E4)
        #     if len(resamp_eda) == 0:
        #         resamp_eda = resamp_sig.reshape(-1, 1)
        #     else:
        #         resamp_eda = np.concatenate((resamp_eda, resamp_sig.reshape(-1, 1)), axis=1)

        data_eda_seg = self.segment(filt_eda, max_time=len(filt_eda), sub_window_size=define.WINDOW_SIZE * define.FS_E4,
                                    stride_size=(define.WINDOW_SIZE - define.OVERLAP) * define.FS_E4)

        feature_eda_seg = [self.extract_feature(data_eda_seg[i], define.FS_E4) for i in range(len(data_eda_seg))]

        # region bvp data
        bvp = e4_data['bvp']
        filt_bvp = bvp[(bvp['time'] >= annotation_itpl['unixtime'].iloc[0]) &
                       (bvp['time'] < annotation_itpl['unixtime'].iloc[
                           -1] + define.ONE_SECOND_IN_MILISECOND)]

        filt_bvp = filt_bvp.sort_values(by=['time'])
        filt_bvp = np.array(filt_bvp)[:, 1:]
        data_bvp_seg = self.segment(filt_bvp, max_time=len(filt_bvp), sub_window_size=define.WINDOW_SIZE * define.FS_BVP,
                                    stride_size=(define.WINDOW_SIZE - define.OVERLAP) * define.FS_BVP)

        feature_bvp_seg = [self.extract_feature(data_bvp_seg[i], define.FS_BVP) for i in range(len(data_bvp_seg))]
        # endregion

        list_len_data = [np.shape(data_seg)[0], np.shape(data_kp_seg)[0], np.shape(data_eda_seg)[0],
                         np.shape(data_bvp_seg)[0], np.shape(joint_angles_seg)[0]]

        if any(x != np.shape(label_seg)[0] for x in list_len_data):
            min_value = np.min(
                [np.shape(data_seg)[0], np.shape(label_seg)[0], np.shape(data_kp_seg)[0], np.shape(data_eda_seg)[0],
                 np.shape(data_bvp_seg)[0], np.shape(joint_angles_seg)[0]])
            data_seg = data_seg[:min_value]
            feature_seg = np.array(feature_seg[:min_value])
            label_seg = label_seg[:min_value]
            data_kp_seg = data_kp_seg[:min_value]
            data_eda_seg = data_eda_seg[:min_value]
            data_bvp_seg = data_bvp_seg[:min_value]
            feature_eda_seg = np.array(feature_eda_seg[:min_value])
            feature_kp_seg = np.array(feature_kp_seg[:min_value])
            feature_bvp_seg = np.array(feature_bvp_seg[:min_value])
            feature_joint_angles_seg = np.array(feature_joint_angles_seg[:min_value])
        feature_seg = np.concatenate(
            [feature_seg, feature_eda_seg, feature_bvp_seg, feature_joint_angles_seg], axis=1)
        # feature_seg = np.concatenate(
        #     [feature_seg, feature_eda_seg, feature_kp_seg, feature_bvp_seg, feature_joint_angles_seg], axis=1)
        # feature_seg = np.concatenate([feature_seg, feature_eda_seg, feature_kp_seg, feature_bvp_seg,
        # joint_angles_seg.reshape(joint_angles_seg.shape[0], -1)], axis=1)
        # endregion

        delete_ind = []
        for i in range(len(label_seg)):
            if len(np.unique(label_seg[i])) > 1:
                delete_ind.append(i)
        data_seg = np.delete(data_seg, delete_ind, axis=0)
        label_seg = np.delete(label_seg, delete_ind, axis=0)[:, 0]
        feature_seg = np.delete(np.array(feature_seg), delete_ind, axis=0)
        data_kp_seg = np.delete(data_kp_seg, delete_ind, axis=0)
        data_eda_seg = np.delete(data_eda_seg, delete_ind, axis=0)
        data_bvp_seg = np.delete(data_bvp_seg, delete_ind, axis=0)
        a = 0
        # region segmentation old version
        # data_seg = []
        # label_seg = []
        # feature_seg = []
        # timestamp_ann = annotation_itpl['unixtime']
        # timestamp_data = data_itpl['unixtime']
        # for i in range(len(annotation_itpl)):
        #     seg = data_itpl[(data_itpl["unixtime"] >= annotation_itpl['unixtime'][i]) & (
        #                 data_itpl["unixtime"] <= annotation_itpl['unixtime'][i] + df.ONE_SECOND_IN_MILISECOND)]
        #     data_seg.append(np.asarray(seg[: df.DATA_LEN])[:, 1:])
        #     label_seg.append(annotation_itpl['cls_idx'][i])
        #     # extract features
        #     if feature:
        #         feature_seg.append(extract_feature(seg))
        # endregion

        # region Write tf_record file
        # if 0 < len(data_seg) == len(label_seg) > 0 and write_tfrecord:
        #     tfrecord.generate_data(
        #         path=self.tfrecord_path + f'/{self.user_id}-{self.session_id}_{self.device_id}_{self.e4_device_id}.tfrecord',
        #         dataset=(data_seg, label_seg, feature_seg)
        #     )
        #     print("Generate {}\n".format(
        #         self.tfrecord_path + f'/{self.user_id}-{self.session_id}_{self.device_id}_{self.e4_device_id}.tfrecord'))
        # endregion

        return data_seg, label_seg, feature_seg
