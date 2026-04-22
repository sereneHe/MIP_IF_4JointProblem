"""Data cleaning and generation helpers for MIP4Cluster experiments."""

from __future__ import annotations

import os
import random
import glob
import gzip
import re
import tempfile
import warnings
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
        from mixture_lds.utils.inputlds import dynamical_system

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
        """Export CHB-MIT seizure prediction files into project data/processed/EEG."""

        project_dir = Path(project_dir)
        return self.export_chbmit_seizure_prediction_data(
            raw_root=raw_root,
            output_dir=project_dir / "data" / "processed" / "EEG",
            subjects=subjects,
            window_size=window_size,
            sampling_rate=sampling_rate,
            selected_channels=selected_channels,
        )

    def parse_chbmit_summary_events(self, raw_root: str | Path, subjects=(1, 3, 5)):
        """Parse CHB-MIT summary files and return seizure events."""

        raw_root = Path(raw_root)
        events = []
        for subject in subjects:
            subject_code = f"chb{subject:02d}"
            subject_dir = raw_root / subject_code
            candidates = sorted(subject_dir.glob(f"*{subject_code}-summary.txt"))
            if not candidates:
                candidates = sorted(subject_dir.glob("*summary.txt"))
            if not candidates:
                raise FileNotFoundError(f"No summary file found under {subject_dir}")

            text = candidates[0].read_text(errors="ignore")
            blocks = re.split(r"File Name:\s*", text)[1:]
            for block in blocks:
                file_name = block.split("\\", 1)[0].strip()
                starts = [
                    int(value)
                    for value in re.findall(r"Seizure Start Time:\s*(\d+) seconds", block)
                ]
                ends = [
                    int(value)
                    for value in re.findall(r"Seizure End Time:\s*(\d+) seconds", block)
                ]
                for start, end in zip(starts, ends):
                    events.append(
                        {
                            "subject": subject_code,
                            "file": file_name,
                            "start_sec": start,
                            "end_sec": end,
                            "duration_sec": end - start,
                        }
                    )
        return events

    @staticmethod
    def _resolve_edf_path(subject_dir: Path, file_name: str) -> Path:
        edf_path = subject_dir / file_name
        if edf_path.exists():
            return edf_path

        gz_path = subject_dir / f"{file_name}.gz"
        if gz_path.exists():
            return gz_path

        raise FileNotFoundError(f"Cannot find {edf_path} or {gz_path}")

    @staticmethod
    def _read_edf_window(edf_path: Path, start_sec: float, duration_sec: int, sampling_rate: int):
        try:
            import mne
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "mne is required to read CHB-MIT EDF files."
            ) from exc

        read_path = edf_path
        tmp_path = None
        if edf_path.suffix == ".gz":
            with gzip.open(edf_path, "rb") as source:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(source.read())
                    tmp_path = Path(tmp.name)
            read_path = tmp_path

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Channel names are not unique.*",
                    category=RuntimeWarning,
                )
                edf = mne.io.read_raw_edf(str(read_path), verbose=False, preload=True)
            channel_names = list(edf.ch_names)
            start = int(round(start_sec * sampling_rate))
            stop = start + int(duration_sec * sampling_rate)
            data = edf.get_data(start=start, stop=stop).T
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        target_len = int(duration_sec * sampling_rate)
        if data.shape[0] < target_len:
            pad = np.full((target_len - data.shape[0], data.shape[1]), np.nan)
            data = np.vstack([data, pad])
        elif data.shape[0] > target_len:
            data = data[:target_len]

        return data, channel_names

    @staticmethod
    def _edf_duration_sec(edf_path: Path) -> float:
        try:
            import mne
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "mne is required to inspect CHB-MIT EDF files."
            ) from exc

        read_path = edf_path
        tmp_path = None
        if edf_path.suffix == ".gz":
            with gzip.open(edf_path, "rb") as source:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(source.read())
                    tmp_path = Path(tmp.name)
            read_path = tmp_path

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Channel names are not unique.*",
                    category=RuntimeWarning,
                )
                edf = mne.io.read_raw_edf(str(read_path), verbose=False, preload=False)
            return edf.n_times / float(edf.info["sfreq"])
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _centered_window_start(start_sec: int, end_sec: int, duration_sec: int) -> float:
        center = (start_sec + end_sec) / 2.0
        return max(0.0, center - duration_sec / 2.0)

    @staticmethod
    def _same_file_far_normal_window_start(
        start_sec: int,
        end_sec: int,
        duration_sec: int,
        record_duration_sec: float,
        margin_sec: int = 300,
    ) -> tuple[float, str]:
        before_start = start_sec - margin_sec - duration_sec
        after_start = end_sec + margin_sec
        latest_start = max(0.0, record_duration_sec - duration_sec)

        candidates = []
        if before_start >= 0:
            candidates.append((float(before_start), "before"))
        if after_start + duration_sec <= record_duration_sec:
            candidates.append((float(after_start), "after"))

        if candidates:
            return max(candidates, key=lambda item: min(abs(start_sec - item[0]), abs(item[0] - end_sec)))

        fallback_before = max(0.0, start_sec - duration_sec)
        fallback_after = min(latest_start, end_sec)
        fallback_candidates = [
            (fallback_before, "fallback_before_no_margin"),
            (fallback_after, "fallback_after_no_margin"),
        ]
        return max(
            fallback_candidates,
            key=lambda item: min(abs(start_sec - item[0]), abs(item[0] - end_sec)),
        )

    @staticmethod
    def _segment_label(has_fragment_count: int) -> str:
        if has_fragment_count == 3:
            return "异常"
        if has_fragment_count in (1, 2):
            return "怀疑异常"
        return "正常"

    def export_chbmit_shortest_seizure_segments(
        self,
        raw_root: str | Path,
        output_dir: str | Path,
        subjects=(1, 3, 5),
        sampling_rate: int = 256,
    ):
        """Export the centered shortest seizure-window tensor.

        The exported array has shape ``(n_patients, duration * sampling_rate,
        n_channels)``. Each patient contributes a window centered inside that
        patient's own shortest seizure start/end interval. The shared duration
        is the global shortest seizure duration.
        """

        raw_root = Path(raw_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events = self.parse_chbmit_summary_events(raw_root, subjects=subjects)
        events_by_subject = {}
        for event in events:
            events_by_subject.setdefault(event["subject"], []).append(event)

        min_duration = min(event["duration_sec"] for event in events)
        specs = [
            ("shortest", min_duration, min),
        ]

        all_metadata = []
        exported = {}
        channel_names = None
        for kind, target_duration, selector in specs:
            patient_arrays = []
            has_fragment_count = 0
            for subject in [f"chb{subject:02d}" for subject in subjects]:
                subject_events = events_by_subject.get(subject, [])
                if not subject_events:
                    patient_arrays.append(np.full((target_duration * sampling_rate, 0), np.nan))
                    continue

                event = selector(subject_events, key=lambda row: row["duration_sec"])
                has_fragment_count += 1
                edf_path = self._resolve_edf_path(raw_root / subject, event["file"])
                window_start_sec = self._centered_window_start(
                    event["start_sec"],
                    event["end_sec"],
                    target_duration,
                )
                data, names = self._read_edf_window(
                    edf_path,
                    window_start_sec,
                    target_duration,
                    sampling_rate,
                )
                if channel_names is None:
                    channel_names = names
                patient_arrays.append(data)
                all_metadata.append(
                    {
                        "dataset": kind,
                        "subject": subject,
                        "file": event["file"],
                        "seizure_start_sec": event["start_sec"],
                        "seizure_end_sec": event["end_sec"],
                        "seizure_duration_sec": event["duration_sec"],
                        "window_start_sec": window_start_sec,
                        "window_end_sec": window_start_sec + target_duration,
                        "target_duration_sec": target_duration,
                        "has_seizure_fragment": True,
                    }
                )

            X = np.stack(patient_arrays, axis=0)
            label = self._segment_label(has_fragment_count)
            data_path = output_dir / f"eeg_chbmit_{kind}_{target_duration}s.npy"
            label_path = output_dir / f"eeg_chbmit_{kind}_{target_duration}s_label.txt"
            np.save(data_path, X)
            label_path.write_text(label + "\n")
            exported[kind] = {
                "data_file": data_path,
                "label_file": label_path,
                "shape": X.shape,
                "label": label,
            }

        pd.DataFrame(all_metadata).to_csv(
            output_dir / "eeg_chbmit_min_max_seizure_segments_metadata.csv",
            index=False,
        )
        if channel_names is not None:
            pd.DataFrame(
                {
                    "channel_index": np.arange(1, len(channel_names) + 1),
                    "channel_name": channel_names,
                }
            ).to_csv(output_dir / "eeg_chbmit_channel_names.csv", index=False)

        pd.DataFrame(events).to_csv(output_dir / "eeg_chbmit_summary_events.csv", index=False)
        return exported

    def export_chbmit_centered_seizure_event_tables(
        self,
        raw_root: str | Path,
        output_dir: str | Path,
        subjects=(1, 3, 5),
        window_duration_sec: int = 27,
        sampling_rate: int = 256,
        max_events_per_subject: int | None = None,
        include_normal: bool = False,
        normal_margin_sec: int = 300,
    ):
        """Export all centered seizure events as per-patient 3D arrays.

        Each subject gets one array with shape
        ``(sample_num, window_duration_sec * sampling_rate, n_channels)``.
        """

        raw_root = Path(raw_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events = self.parse_chbmit_summary_events(raw_root, subjects=subjects)
        events_by_subject = {}
        for event in events:
            events_by_subject.setdefault(event["subject"], []).append(event)

        exported = {}
        all_metadata = []
        channel_names = None
        for subject in [f"chb{subject:02d}" for subject in subjects]:
            seizure_arrays = []
            normal_arrays = []
            subject_metadata = []
            subject_events = events_by_subject.get(subject, [])
            if max_events_per_subject is not None:
                subject_events = subject_events[:max_events_per_subject]

            for event_id, event in enumerate(subject_events):
                window_start_sec = self._centered_window_start(
                    event["start_sec"],
                    event["end_sec"],
                    window_duration_sec,
                )
                edf_path = self._resolve_edf_path(raw_root / subject, event["file"])
                data, names = self._read_edf_window(
                    edf_path,
                    window_start_sec,
                    window_duration_sec,
                    sampling_rate,
                )
                if channel_names is None:
                    channel_names = names
                seizure_arrays.append(data)
                sample_id = event["file"].replace(".edf", "")
                subject_metadata.append(
                    {
                        "subject": subject,
                        "event_id": event_id,
                        "sample_id": sample_id,
                        "file": event["file"],
                        "segment": "abnormal",
                        "label": "异常",
                        "seizure_start_sec": event["start_sec"],
                        "seizure_end_sec": event["end_sec"],
                        "seizure_duration_sec": event["duration_sec"],
                        "window_start_sec": window_start_sec,
                        "window_end_sec": window_start_sec + window_duration_sec,
                        "window_duration_sec": window_duration_sec,
                    }
                )

                if include_normal:
                    record_duration_sec = self._edf_duration_sec(edf_path)
                    normal_start_sec, normal_strategy = self._same_file_far_normal_window_start(
                        event["start_sec"],
                        event["end_sec"],
                        window_duration_sec,
                        record_duration_sec,
                        margin_sec=normal_margin_sec,
                    )
                    normal_data, _ = self._read_edf_window(
                        edf_path,
                        normal_start_sec,
                        window_duration_sec,
                        sampling_rate,
                    )
                    normal_arrays.append(normal_data)
                    subject_metadata.append(
                        {
                            "subject": subject,
                            "event_id": event_id,
                            "sample_id": sample_id,
                            "file": event["file"],
                            "segment": "normal",
                            "label": "正常",
                            "seizure_start_sec": event["start_sec"],
                            "seizure_end_sec": event["end_sec"],
                            "seizure_duration_sec": event["duration_sec"],
                            "window_start_sec": normal_start_sec,
                            "window_end_sec": normal_start_sec + window_duration_sec,
                            "window_duration_sec": window_duration_sec,
                            "normal_strategy": normal_strategy,
                            "normal_margin_sec": normal_margin_sec,
                            "record_duration_sec": record_duration_sec,
                        }
                    )

            if seizure_arrays:
                X = np.stack(seizure_arrays, axis=0)
            else:
                n_channels = len(channel_names) if channel_names is not None else 0
                X = np.empty((0, window_duration_sec * sampling_rate, n_channels))

            data_path = output_dir / f"{subject}_centered_{window_duration_sec}s_seizures.npy"
            label_path = output_dir / f"{subject}_centered_{window_duration_sec}s_labels.csv"
            np.save(data_path, X)
            label_rows = [
                {"sample_id": row["sample_id"], "label": row["label"]}
                for row in subject_metadata
            ]
            pd.DataFrame(label_rows).to_csv(label_path, index=False)

            normal_path = ""
            if include_normal:
                normal_path = output_dir / f"{subject}_centered_{window_duration_sec}s_normal.npy"
                if normal_arrays:
                    np.save(normal_path, np.stack(normal_arrays, axis=0))
                else:
                    n_channels = len(channel_names) if channel_names is not None else 0
                    np.save(
                        normal_path,
                        np.empty((0, window_duration_sec * sampling_rate, n_channels)),
                    )

            all_metadata.extend(subject_metadata)
            exported[subject] = {
                "data_file": data_path,
                "label_file": label_path,
                "normal_file": normal_path,
                "shape": X.shape,
                "sample_num": X.shape[0],
            }

        pd.DataFrame(all_metadata).to_csv(
            output_dir / f"centered_{window_duration_sec}s_seizure_events_metadata.csv",
            index=False,
        )
        if channel_names is not None:
            pd.DataFrame(
                {
                    "channel_index": np.arange(1, len(channel_names) + 1),
                    "channel_name": channel_names,
                }
            ).to_csv(output_dir / "channel_names.csv", index=False)
        return exported

    def build_chbmit_eeg_dataset(
        self,
        raw_root: str | Path,
        output_dir: str | Path,
        subjects=(1, 3, 5),
        durations=None,
        sampling_rate: int = 256,
        max_events_per_subject: int | None = None,
        normal_margin_sec: int = 300,
    ):
        """Build CHB-MIT 27s/120s tensors through build_chbmit_eeg.py.

        This is a convenience wrapper so callers using ``DataPreprocessing`` can
        generate the final ``eeg-*.npy`` files without
        importing the build script directly. When ``max_events_per_subject`` is
        None, each patient keeps the smallest seizure count found across all
        selected patient summary files. When ``durations`` is None, the shortest
        and longest seizure durations are parsed from those summary files.
        """

        from mixture_lds.data.build_chbmit_eeg import build_chbmit_eeg

        return build_chbmit_eeg(
            raw_root=Path(raw_root),
            output_dir=Path(output_dir),
            subjects=subjects,
            durations=durations,
            sampling_rate=sampling_rate,
            max_events_per_subject=max_events_per_subject,
            normal_margin_sec=normal_margin_sec,
        )

    # ------------------------------------------------------------------
    # generic data cleaning
    # ------------------------------------------------------------------

    def datacleaning(self, data_dir, name, S, I, T, M, J):
        """Load and reshape supported datasets for MIP4Cluster tests."""
        if name == "lds":
            path_list = [
                "./data/synthetic/2_2_test.npy",
                "./data/synthetic/3_2_test.npy",
                "./data/synthetic/4_2_test.npy",
            ]
            return self.load_lds_data(Path("."), path_list, S, I, T, M)

        if name == "ecg":
            trainpath = "./data/raw/ECG/ECG5000_TRAIN.arff"
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
