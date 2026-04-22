"""Data cleaning and generation helpers for MIP4Cluster experiments."""

from __future__ import annotations

import os
import random
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


class DataPreprocessing:
    """Load supported datasets and generate synthetic LDS data."""

    @staticmethod
    def _load_yaml(path):
        if yaml is not None:
            return yaml.safe_load(Path(path).read_text())

        data = {}
        current_key = None
        for raw_line in Path(path).read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- ") and current_key:
                data.setdefault(current_key, []).append(line[2:].strip())
                continue
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_key = key
            else:
                current_key = key
                if value.isdigit():
                    data[key] = int(value)
                else:
                    try:
                        data[key] = float(value)
                    except ValueError:
                        data[key] = value
        return data

    # ------------------------------------------------------------------
    # sytheticData
    # ------------------------------------------------------------------

    def default_lds_matrices(self):
        """Return the default LDS matrices used by the original tests."""
        g = np.array(
            [
                [
                    0.8 * np.matrix([[0.9, 0.2], [0.1, 0.1]]),
                    0.8 * np.matrix([[0.8, 0.2], [0.2, 0.1]]),
                ],
                [
                    0.6
                    * np.matrix(
                        [[1.0, 0.8, 0.8], [0.6, 0.1, 0.2], [0.3, 0.2, 0.2]]
                    ),
                    0.6
                    * np.matrix(
                        [[1.0, 1.0, 0.6], [0.7, 0.2, 0.2], [0.2, 0.1, 0.1]]
                    ),
                ],
                [
                    np.matrix(
                        [
                            [0.9, 0.8, 0.5, 0.2],
                            [0.9, 0.1, 0.3, 0.4],
                            [0.8, 0.2, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.7],
                        ]
                    )
                    * 0.4,
                    np.matrix(
                        [
                            [1.0, 0.8, 0.5, 0.3],
                            [0.6, 0.2, 0.3, 0.4],
                            [0.8, 0.2, 0.3, 0.1],
                            [0.2, 0.2, 0.3, 0.7],
                        ]
                    )
                    * 0.4,
                ],
            ],
            dtype=object,
        )

        f_dash = {
            0: [
                0.8 * np.array([[1.0, 1.0], [0.2, 0.2]]),
                0.8 * np.array([[0.8, 1.0], [0.1, 0.2]]),
            ],
            1: [
                0.6 * np.array([[0.7, 0.4, 0.3], [0.2, 0.6, 0.2]]),
                0.6 * np.array([[0.5, 0.4, 0.1], [0.2, 0.5, 0.1]]),
            ],
            2: [
                np.array([[0.2, 0.5, 0.1, 0.1], [0.8, 0.6, 0.1, 0.1]]) * 0.4,
                np.array([[0.2, 0.4, 0.1, 0.1], [0.6, 0.2, 0.2, 0.2]]) * 0.4,
            ],
        }
        return g, f_dash

    def dynamic_generate(self, g, f_dash, proc_noise_std, obs_noise_std, inputs, T):
        """Generate one LDS trajectory."""
        from inputlds import dynamical_system

        n = len(g)
        m = len(f_dash)
        if inputs == 0:
            inputs = np.zeros((m, T))
        dim = len(inputs)

        ds1 = dynamical_system(
            g,
            np.zeros((n, dim)),
            f_dash,
            np.zeros((m, dim)),
            process_noise="gaussian",
            observation_noise="gaussian",
            process_noise_std=proc_noise_std,
            observation_noise_std=obs_noise_std,
        )

        h0 = np.ones(ds1.d)
        ds1.solve(h0=h0, inputs=inputs, T=T)
        return np.asarray(ds1.outputs).reshape(T, m)

    def data_generation(self, g, f_dash, pro_rang, obs_rang, T, S, output_dir="./data"):
        """Generate two-cluster LDS synthetic datasets and save them to output_dir."""
        proL = len(pro_rang)
        obsL = len(obs_rang)
        file_name = []
        os.makedirs(output_dir, exist_ok=True)

        for gg in range(len(g)):
            n = gg + 2
            m = 2
            cluster_1 = []
            cluster_2 = []

            for i in range(proL):
                for j in range(obsL):
                    proc_noise_std = pro_rang[i]
                    obs_noise_std = obs_rang[j]
                    inputs = 0
                    for _ in range(S):
                        data_1 = self.dynamic_generate(
                            g[gg, 0],
                            f_dash[gg][0],
                            proc_noise_std,
                            obs_noise_std,
                            inputs,
                            T,
                        )
                        cluster_1.append(data_1)

                        data_2 = self.dynamic_generate(
                            g[gg, 1],
                            f_dash[gg][1],
                            proc_noise_std,
                            obs_noise_std,
                            inputs,
                            T,
                        )
                        cluster_2.append(data_2)

            Y = np.concatenate((np.array(cluster_1), np.array(cluster_2)), axis=0)
            Y_label = np.concatenate(
                (np.zeros(len(cluster_1)), np.ones(len(cluster_2))), axis=0
            )
            print(Y.shape, Y_label.shape)

            data = Y.reshape(320, -1)
            f_name = os.path.join(output_dir, f"{n}_{m}_test.npy")
            with open(f_name, "wb") as f:
                np.save(f, data)
            file_name.append(f_name)

        return file_name

    def generate_default_lds(self, output_dir: str | Path, T: int, S: int):
        """Generate default LDS datasets into output_dir."""
        g, f_dash = self.default_lds_matrices()
        pro_rang = np.arange(0.02, 0.1, 0.02)
        obs_rang = np.arange(0.02, 0.1, 0.02)
        return self.data_generation(
            g,
            f_dash,
            pro_rang,
            obs_rang,
            T,
            S,
            output_dir=str(output_dir),
        )

    def prepare_lds_from_config(self, project_dir: str | Path, result_dir: str | Path):
        """Generate LDS data from lds.yaml and export per-sample npy files."""
        project_dir = Path(project_dir)
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        conf = self._load_yaml(project_dir / "experiment_conf" / "problems" / "lds.yaml")

        S = int(conf["S_len"])
        I = int(conf["I_len"])
        T = int(conf["T_len"])
        M = int(conf["F_len"])
        data_root = project_dir / conf["data_root"]

        generated_files = self.generate_default_lds(data_root, T, S)
        print("generated lds files:", [str(path) for path in generated_files])

        label = np.concatenate((np.zeros(I), np.ones(I)), axis=0)
        label_path = result_dir / "lds_label.npy"
        np.save(label_path, label)

        exported = []
        for offset, data_file in enumerate(conf["data_files"]):
            data_path = data_root / data_file
            data_X = np.load(data_path)
            data_X = data_X.reshape(2 * S, I, T, M)

            X_s = np.zeros((S, 2 * I, T, M))
            for s in range(S):
                xx_1 = data_X[s]
                xx_2 = data_X[s + S]
                X_s[s] = np.concatenate((xx_1, xx_2), axis=0).reshape(
                    2 * I, T, M
                )

            n = 2 + offset
            for s in range(S):
                out = result_dir / f"lds{n}_sample{s}.npy"
                np.save(out, X_s[s])
                exported.append(out)

            print(data_file, "->", f"lds{n}", X_s.shape)

        print("label:", label_path)
        return {
            "generated_files": generated_files,
            "exported_files": exported,
            "label_file": label_path,
        }

    def load_lds_data(self, data_root, data_files, S, I, T, M):
        """Load generated LDS data files and reshape them for clustering tests."""
        X_list = []
        for data_file in data_files:
            path = Path(data_root) / data_file
            data_X = np.load(path)
            data_X = data_X.reshape(2 * S, I, T, M)

            X_s = np.zeros((S, 2 * I, T, M))
            for s in range(S):
                xx_1 = data_X[s]
                xx_2 = data_X[s + S]
                X_s[s] = np.concatenate((xx_1, xx_2), axis=0).reshape(
                    2 * I, T, M
                )
            X_list.append(X_s)

        X = np.array(X_list).reshape(len(data_files), S, 2 * I, T, M)
        label = np.concatenate((np.zeros(I), np.ones(I)), axis=0)
        return X, label

    # ------------------------------------------------------------------
    # ECG
    # ------------------------------------------------------------------

    def load_ecg_data(self, data_path, S, I, T, M, seed=None):
        """Load ECG5000 and return a balanced normal-vs-abnormal sample."""
        X_data = self._load_arff_as_dataframe(data_path)
        print(X_data.target.value_counts())

        X_1 = X_data[X_data.target == b"1"].iloc[:, :-1].values
        X_2 = X_data[X_data.target == b"2"].iloc[:, :-1].values
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        idx_1 = np.arange(len(X_1))
        idx_2 = np.arange(len(X_2))
        iidx_1 = random.sample(sorted(idx_1), I)
        iidx_2 = random.sample(sorted(idx_2), I)
        np.random.shuffle(iidx_1)
        np.random.shuffle(iidx_2)

        XX_1 = X_1[iidx_1]
        XX_2 = X_2[iidx_2]
        X = np.concatenate((XX_1, XX_2), axis=0).reshape(S, 2 * I, T, M)
        label = np.concatenate((np.zeros(I), np.ones(I)), axis=0)
        return X, label

    # ------------------------------------------------------------------
    # EEG
    # ------------------------------------------------------------------

    def load_eeg_data(
        self,
        data_path,
        label_path,
        n_normal=None,
        n_seizure=None,
        time_points=None,
        channels=None,
    ):
        """Load EEG seizure data and optionally return a balanced subset."""
        X = np.load(data_path, allow_pickle=True)
        label_path = Path(label_path)
        if label_path.suffix == ".npy":
            y = np.load(label_path, allow_pickle=True)
        else:
            y = pd.read_csv(label_path, header=None).iloc[:, 0].to_numpy()

        if n_normal is not None and n_seizure is not None:
            normal_idx = np.where(y == 0)[0][:n_normal]
            seizure_idx = np.where(y == 1)[0][:n_seizure]
            idx = np.r_[normal_idx, seizure_idx]
            X = X[idx]
            y = y[idx]

        if time_points is not None:
            X = X[:, :time_points, :]
        if channels is not None:
            X = X[:, :, :channels]

        return X, y

    def load_chbmit_seizure_prediction_data(
        self,
        raw_root: str | Path,
        subjects=(1, 3, 5),
        window_size: int = 50,
        sampling_rate: int = 256,
        selected_channels=("FP1-F3", "FP2-F4", "FP2-F8"),
        seizure_begin=None,
        seizure_files=None,
        normal_files=None,
        normalize: bool = True,
        return_dataframe: bool = False,
    ):
        """Load CHB-MIT seizure prediction data in the notebook format.

        The notebook keeps only subjects 1, 3, and 5, selects 14 recordings per
        subject (7 seizure and 7 normal), extracts a fixed-length window from
        each EDF file, and keeps three EEG channels.

        Returns
        -------
        X : np.ndarray
            Array with shape (len(subjects), 14, window_size * sampling_rate, 3)
        y : np.ndarray
            Array with shape (len(subjects), 14)
        """

        try:
            import mne
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "mne is required to read CHB-MIT EDF files."
            ) from exc

        raw_root = Path(raw_root)
        if seizure_begin is None:
            seizure_begin = {
                1: [2996, 1467, 1732, 1015, 1720, 327, 1862],
                3: [362, 731, 432, 2162, 1982, 2592, 1725],
                5: [417, 1086, 2317, 2451, 2348, 2348, 1086],
            }
        if seizure_files is None:
            seizure_files = {
                1: [3, 4, 15, 16, 18, 21, 26],
                3: [1, 2, 3, 4, 34, 35, 36],
                5: [6, 13, 16, 17, 22, 23, 24],
            }
        if normal_files is None:
            normal_files = {
                1: [6, 10, 22, 25, 30, 35, 40],
                3: [49, 51, 55, 58, 60, 66, 71],
                5: [81, 89, 104, 109, 113, 116, 98],
            }

        def _subject_folder(subject: int) -> Path:
            return raw_root / f"chb{subject:02d}"

        def _file_index(path: str | Path) -> int:
            name = Path(path).name
            stem = name.replace(".", "_")
            return int(stem.split("_")[-2].split("+")[0])

        def _load_edf_window(edf_path: Path, begin_sec: int):
            edf = mne.io.read_raw_edf(str(edf_path), verbose=False, preload=True)
            df_raw = pd.DataFrame(data=edf.get_data().T, columns=edf.ch_names)
            start = int(begin_sec * sampling_rate)
            end = int((begin_sec + window_size) * sampling_rate)
            window = df_raw.iloc[start:end].copy()
            if window.shape[0] < window_size * sampling_rate:
                pad_len = window_size * sampling_rate - window.shape[0]
                pad = pd.DataFrame(
                    np.nan,
                    index=np.arange(pad_len),
                    columns=window.columns,
                )
                window = pd.concat([window.reset_index(drop=True), pad], axis=0)
            return window

        subject_arrays = []
        subject_labels = []
        missing_files = []

        for subject in subjects:
            folder = _subject_folder(subject)
            edf_files = sorted(glob.glob(str(folder / "*.edf")))
            if not edf_files:
                missing_files.append(str(folder))
                continue

            file_map = {}
            for file_name in edf_files:
                try:
                    file_map[_file_index(file_name)] = Path(file_name)
                except Exception:
                    continue

            selected_indices = list(seizure_files.get(subject, [])) + list(
                normal_files.get(subject, [])
            )
            selected_subject_data = []
            selected_subject_labels = []

            for idx in selected_indices:
                edf_path = file_map.get(idx)
                if edf_path is None:
                    continue
                begin = 0
                if idx in seizure_files.get(subject, []):
                    seizure_pos = seizure_files[subject].index(idx)
                    begin = seizure_begin.get(subject, [0] * 7)[seizure_pos]
                    label = 1
                else:
                    label = 0

                try:
                    window = _load_edf_window(edf_path, begin)
                except Exception:
                    continue

                if return_dataframe:
                    selected_subject_data.append(window)
                    selected_subject_labels.append(label)
                    continue

                available_channels = list(window.columns)
                chosen = []
                for channel in selected_channels:
                    if channel in available_channels:
                        chosen.append(channel)
                if len(chosen) != len(selected_channels):
                    raise ValueError(
                        f"Missing channels in {edf_path}: "
                        f"expected {selected_channels}, found {available_channels[:10]}"
                    )

                sample = window.loc[:, chosen].to_numpy(dtype=float)
                selected_subject_data.append(sample)
                selected_subject_labels.append(label)

            if not selected_subject_data:
                continue

            if return_dataframe:
                subject_arrays.append(selected_subject_data)
                subject_labels.append(np.asarray(selected_subject_labels, dtype=int))
                continue

            subject_data = np.asarray(selected_subject_data, dtype=float)
            subject_label = np.asarray(selected_subject_labels, dtype=int)

            if normalize:
                mean = np.nanmean(subject_data, axis=(1, 2), keepdims=True)
                std = np.nanstd(subject_data, axis=(1, 2), keepdims=True)
                std[std == 0] = 1e-8
                subject_data = (subject_data - mean) / std

            subject_arrays.append(subject_data)
            subject_labels.append(subject_label)

        if not subject_arrays:
            raise FileNotFoundError(
                f"No CHB-MIT EDF files were loaded from {raw_root}. "
                f"Missing subject folders: {missing_files}"
            )

        if return_dataframe:
            return subject_arrays, subject_labels

        X = np.asarray(subject_arrays, dtype=float)
        y = np.asarray(subject_labels, dtype=int)
        return X, y

    def export_chbmit_seizure_prediction_data(
        self,
        raw_root: str | Path,
        output_dir: str | Path,
        subjects=(1, 3, 5),
        window_size: int = 50,
        sampling_rate: int = 256,
        selected_channels=("FP1-F3", "FP2-F4", "FP2-F8"),
    ):
        """Export notebook-style CHB-MIT seizure prediction arrays and labels."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X, y = self.load_chbmit_seizure_prediction_data(
            raw_root=raw_root,
            subjects=subjects,
            window_size=window_size,
            sampling_rate=sampling_rate,
            selected_channels=selected_channels,
            normalize=True,
        )

        if isinstance(X, np.ndarray) and X.ndim == 4:
            X = X.reshape(-1, X.shape[-2], X.shape[-1])
        if isinstance(y, np.ndarray) and y.ndim >= 2:
            y = y.reshape(-1)

        data_path = output_dir / "eeg_seizure_Train_normalized.npy"
        label_path = output_dir / "seizure_label.csv"
        np.save(data_path, X)
        np.savetxt(label_path, y.reshape(-1), delimiter=",", fmt="%d")
        return {
            "data_file": data_path,
            "label_file": label_path,
            "data_shape": X.shape,
            "label_shape": y.shape,
        }

    def export_default_chbmit_seizure_prediction_data(
        self,
        raw_root: str | Path,
        project_dir: str | Path,
        subjects=(1, 3, 5),
        window_size: int = 50,
        sampling_rate: int = 256,
        selected_channels=("FP1-F3", "FP2-F4", "FP2-F8"),
    ):
        """Export CHB-MIT seizure prediction files into project data/EEG."""

        project_dir = Path(project_dir)
        return self.export_chbmit_seizure_prediction_data(
            raw_root=raw_root,
            output_dir=project_dir / "data" / "EEG",
            subjects=subjects,
            window_size=window_size,
            sampling_rate=sampling_rate,
            selected_channels=selected_channels,
        )

    # ------------------------------------------------------------------
    # generic data cleaning
    # ------------------------------------------------------------------

    def datacleaning(self, data_dir, name, S, I, T, M, J):
        """Load and reshape supported datasets for MIP4Cluster tests."""
        if name == "lds":
            path_list = [
                "./data/raw/2_2_test.npy",
                "./data/raw/3_2_test.npy",
                "./data/raw/4_2_test.npy",
            ]
            return self.load_lds_data(Path("."), path_list, S, I, T, M)

        if name == "ecg":
            trainpath = "./data/raw/ECG5000_TRAIN.arff"
            return self.load_ecg_data(trainpath, S, I, T, M)

        if data_dir:
            return self._load_npy_directory(data_dir)

        print("Wrong data type!")
        return None

    @staticmethod
    def _load_arff_as_dataframe(path):
        data, _ = arff.loadarff(path)
        return pd.DataFrame(data)

    @staticmethod
    def _load_npy_directory(data_dir):
        """Load all .npy files from a directory into a name -> array mapping."""
        data_dict = {}
        for file_name in os.listdir(data_dir):
            if not file_name.endswith(".npy"):
                continue
            file_path = os.path.join(data_dir, file_name)
            key_name = file_name.replace(".npy", "")
            data_dict[key_name] = np.load(file_path, allow_pickle=True)

        if not data_dict:
            print("No .npy files found in the given data directory.")
            return None

        for key, value in data_dict.items():
            if value.ndim not in (3, 4):
                print(f"Warning: unsupported shape for {key}: {value.shape}")
        return data_dict
