# MIP4Cluster-LDS
Mixed-Integer Programming Method for Jointly Clustering (Multiple) Linear Dynamical Systems

# Project Description
This is the source code for the paper Joint Problems in Learning Multiple Dynamical Systems accepted by Allerton conference.

## Data
ECG data: The test on electrocardiogram (ECG) data gives an inspiring application on guiding cardiologist’s diagnosis and treatment. ECG5000 includes 500 sequences, where there are 292 normal samples and 208 samples of four types of heart failure. Each sequence contains a whole period of heartbeat with 140 time stamps.

### CHB-MIT EEG Data
The CHB-MIT dataset contains recordings from 24 individuals/patients. In this
project we use patients `chb01`, `chb03`, and `chb05`. These folders contain
42, 38, and 39 EDF recordings respectively, for a total of 119 EDF files. Each
recording has 23 EEG channels associated with different brain regions, sampled
at 256 Hz.

Each `chbnn-summary.txt` file records the montage used for the recordings and,
for EDF files containing seizures, the elapsed seizure start and end times in
seconds from the beginning of that EDF file. Across `chb01`, `chb03`, and
`chb05`, the parsed summaries contain 19 positive seizure samples and 100
negative samples.

The EDF-to-array preprocessing extracts the seizure event if a seizure is
present, or the first 50 seconds if no seizure is present. The processed EEG
data can be represented as a 4D NumPy array:

```text
(n_patients, n_samples, n_timesteps, n_channels)
= (3, 42 + 38 + 39, 12800, 23)
```

The selected per-subject seizure durations are:

| Subject | Seizures | Shortest seizure | Longest seizure |
|---|---:|---:|---:|
| `chb01` | 7 | 27 s (`chb01_04.edf`, 1467-1494) | 101 s (`chb01_26.edf`, 1862-1963) |
| `chb03` | 7 | 47 s (`chb03_34.edf`, 1982-2029) | 69 s (`chb03_03.edf`, 432-501) |
| `chb05` | 5 | 96 s (`chb05_16.edf`, 2317-2413) | 120 s (`chb05_17.edf`, 2451-2571) |

Overall, the shortest seizure is 27 seconds. The derived shortest seizure-window
tensor is saved under
`data/processed/EEG/seizure_segments/`:

```text
eeg_chbmit_shortest_27s.npy   # shape (3, 6912, 23), label: 异常
```

Here `6912 = 27 * 256`. The patient dimension keeps the subject codes `chb01`,
`chb03`, and `chb05`, and the channel metadata is saved in
`eeg_chbmit_channel_names.csv`. Each patient window is centered inside that
patient's own shortest-seizure start/end interval and uses the global shortest
seizure duration of 27 seconds. The exact EDF file and centered window
boundaries are saved in `eeg_chbmit_shortest_27s_metadata.csv`. If all three
patients contain a seizure fragment for the aligned brain-region/channel window,
the label is `异常`; if only one or two patients contain a seizure fragment, the
label is `怀疑异常`.

## Methods
This project includes IF-Gurobi, IF, EM, FFT and DTW 5 methods, for more information refer to paper
[Allerton paper](https://arxiv.org/pdf/2311.02181)

# Project structure
The repository is organized around a packaged Python implementation under
`src/mixture_lds`, with experiment configuration and shell entry points kept
separate.
```txt
├── data/                         # Local datasets and generated arrays
│   ├── raw/
│   │   ├── EEG/
│   │   └── ECG/
│   ├── processed/
│   │   └── EEG/
│   └── synthetic/
├── experiment_conf/              # YAML configs and legacy-compatible wrappers
│   ├── problems/
│   ├── solvers/
│   └── utils/
├── notebooks/                    # Exploratory and original notebook workflows
├── reports/                      # Figures and summary artifacts
├── Result_*/                     # Experiment outputs
├── src/mixture_lds/              # Packaged implementation
│   ├── data/preprocessing.py
│   ├── experiments.py
│   ├── models/mip_if_3dindexing.py
│   ├── solvers/solve_*.py
│   └── utils/
├── scripts/                      # Shell experiment entry points
├── pyproject.toml
└── uv.lock
```

## Easily to Get Started
If one would like to run experiments on local, try the notebooks on [Colab](https://colab.research.google.com/ ). If you use Mosek or Gurobi as a solver, a license is required. After applying for a license from Mosek and Gurobi, you can move "mosek.lic"  and "Gurobi.lic" under the setting path.

### Set license file path
```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

### install Bonmin
```bash
brew install bonmin
```

### Dependencies
1. Python>=3.9.7
2. Gurobi 12.0.1 https://www.gurobi.com/
3. Pyomo v6.6.2 https://www.pyomo.org/
4. Bonmin https://www.coin-or.org/Bonmin/
5. ncpol2sdpa 1.12.2 https://ncpol2sdpa.readthedocs.io/en/stable/index.html
6. Mosek 10.1 https://www.mosek.com/

### Notebooks
1. Dataset: "./notebooks/data_generation.ipynb" can be used to generate your own dataset. You can also utilize generated and real-world data under the folder "./data/raw". 

2. Regularizations: We were trying to find the best regularization parameter, i.e., with denser system matrices. Find more in jupyter notebooks with the name "./notebooks/Regularization.ipynb".

3. Experiments: We provide the experiment scripts of both proposed methods and baselines under "./notebooks/MIP_IF_4Cluster.ipynb". You can easily run them with your own preference.

4. Results: All results and figures mentioned in the paper are under the folder ./result. You can utilize "./notebooks/runtime_F1_plot.ipynb" to visualize the results.

## Want to develop
If one would like to develop the codes, pls fork to your git or follow the following steps to better manage in your code editer.

### Setup
1. Clone the repository:
    ```bash
   git clone https://github.com/sereneHe/MIP_IF_4JointProblem.git
   cd MIP_IF_4JointProblem/
   ```
2. Install uv:
   ```bash
   pip install uv
   ```
3. Install the required dependencies:
   ```bash
   uv sync
   ```
4. Run experiments through the packaged dispatcher:
   ```bash
   uv run python -m mixture_lds.experiments --multirun --config-name=config-cluster experiment='MIP4Cluster' problem="EEG" problem.eeg_mode="test" solver="fft" seed="30"
   ```
5. EEG experiments:
   ```bash
   sh scripts/run_eeg_test.sh   # 10 x 100 x 5 lightweight smoke test
   sh scripts/run_eeg.sh        # CHB-MIT EDF-based seizure prediction pipeline
   ```
   The CHB-MIT path can read raw EDF/EDF.GZ files and export packed
   `data/processed/EEG/eeg-27s.npy` and `data/processed/EEG/eeg-120s.npy`
   tensors when the raw data is present. If the raw data is not present,
   `scripts/run_eeg.sh` uses the existing packed `eeg-27s.npy` file.
6. Other experiment entry points:
   ```bash
   sh scripts/run_ecg.sh
   sh scripts/run_lds.sh
   ```


 
