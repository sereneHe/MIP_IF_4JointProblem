# MIP4Cluster-LDS
Mixed-Integer Programming Method for Jointly Clustering (Multiple) Linear Dynamical Systems

# Project Description
This is the source code for the paper Joint Problems in Learning Multiple Dynamical Systems accepted by Allerton conference.

## Data
ECG data: The test on electrocardiogram (ECG) data gives an inspiring application on guiding cardiologist’s diagnosis and treatment. ECG5000 includes 500 sequences, where there are 292 normal samples and 208 samples of four types of heart failure. Each sequence contains a whole period of heartbeat with 140 time stamps.

## Methods
This project includes IF-Gurobi, IF, EM, FFT and DTW 5 methods, for more information refer to paper
[Allerton paper](https://arxiv.org/pdf/2311.02181)

# Project structure
The project uses [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and is based on [Machine Learning Operations template](https://github.com/SkafteNicki/mlops_template).
```txt
├── .dvccontainer                     # DVC configuration files
├── .github/    
│   └── dependabot.yaml
│   └── workflows/
│       ├── linting.yaml
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
│       ├── 2_2_test.npy
│       └── 3_2_test.npy
│       ├── 4_2_test.npy
│       └── ECG5000_TRAIN.arff
├── dockerfiles/              # Dockerfiles
│   ├── evaluate.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/   
├── notebooks/                     # Data directory
│   ├── data_generation.ipynb
│   └── method_baseline.ipynb
│   ├── MIP_IF_4Cluster.ipynb
│   └── MIP4Cluster_py.ipynb
│   ├── Regularization.ipynb
│   └── runtime_F1_plot.ipynb
├── outputs/                   
├── reports/              
│   └── figures/
├── Result_ecg/                
│   └── ...
├── Result_ecg_cd/           
│   └── ...
├── Result_lds/               
│   └── ...
├── Result_lds2/                
│   └── ...
├── Result_lds3/                
│   └── ...
├── Result_lds4/                
│   └── ...
├── src/                      # Source code
│   ├── mixture_lds/
│   │   ├── __init__.py
│   │   ├── ClusterMultiLDS_Gurobi.py
│   │   ├── inputlds.py
│   │   ├── MIP_IF.py
│   │   ├── MIP4Cluster.py
│   │   ├── utils_MIP4Cluster.py
└── tests/                    # Tests
│   ├── Generate_data.py
│   ├── plot_test.py
│   ├── Test_ECG.py
│   └── Test_Synthetic_Data.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
└── tasks.py                  # Project tasks
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
2. Install uv (optional, for running scripts):
   ```bash
   pip install uv
   ```
3. Activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```bash
   uv sync
   pip install -r requirements.txt 
   uv pip install -r requirements.txt 
   ```
5. Data cleaning/generation
   ```bash
   uv run python tests/Generate_data.py
   ```
6. Tests
   ```bash
   uv run python tests/Test_ECG.py
   uv run python tests/Test_Synthetic_Data.py
   ```
6. Test MIP4Cluster method with more than 2 classes
   ```bash
   uv run python src/mixture_lds/ClusterMultiLDS_Gurobi.py
   ```


 
