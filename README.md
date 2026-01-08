# CS58010 Fall 2025 Group 6 - FedALA Integration for OpenFGL

## Results Spreadsheet
For refrence, some of the experimental results are recorded here: [Google Sheets - CS58010 Group 6 Results](https://docs.google.com/spreadsheets/d/1IW_IZTJXF-h-B8g9ly8zyBHfuTcnIyZpIQVH-xSyIZg/edit?usp=sharing)

---

## Installation & Environment Setup

**Important:** Before running any experiments, you must set up the conda environment.

1. **Read the installation instructions:**
```bash
   cat read_me_for_installing_the_env.txt
```

2. **Create the conda environment:**
```bash
   # Option 1: Using environment.yml
   conda env create -f docs/updated_environment.yml
   
   # Option 2: Using requirements.txt
   conda create -n openfgl python=3.10 -y
   conda activate openfgl
   pip install -r docs/new_requirements2.txt
```

3. **Activate the environment:**
```bash
   conda activate openfgl
```

---

## Instructions to Reproduce CS58010 Group 6's Results

### 1. Reproducing Results for Table 6 and Figure 2a of the Original OpenFGL Paper

**Results Tab:** `OpenFGL_reproduce` in the [spreadsheet](https://docs.google.com/spreadsheets/d/1IW_IZTJXF-h-B8g9ly8zyBHfuTcnIyZpIQVH-xSyIZg/edit?usp=sharing)

#### For FedALA and other algorithms that are already in the OpenFGL paper (replace "fedala" with the algorithm names):
```bash
# Small datasets (5 clients)
# python run_table6_graphfl.py --groups fedala --datasets MUTAG COX2 --num-clients 5 --seeds 42 123 456

# Standard datasets (10 clients)
python run_table6_graphfl.py --groups fedala --datasets BZR ENZYMES DD PROTEINS IMDB-BINARY IMDB-MULTI COLLAB --num-clients 10 --seeds 42 123 456
```

#### Alternatively:
Configuration files are inside `configs/graph_fl/`. Choose the specific case you want to run:
```bash
# Example: Run FedAvg on DD dataset
python -u run_experiments.py --config configs/graph_fl/dd/dd_fedavg.yaml

# Other examples:
python -u run_experiments.py --config configs/graph_fl/mutag/mutag_fedprox.yaml
python -u run_experiments.py --config configs/graph_fl/proteins/proteins_gcfl_plus.yaml
```

**Output Format:**
At the end of the output, results will appear in this order:
```
"Dataset", "Scenario", "Algorithm", "Accuracy Reported",
"Test Acc", "Std Dev", "Val Acc", "Best Round",
"Comm (MB)", "Time (s)", "Task", "Model",
"Simulation Mode", "Num Clients", "Dirichlet Alpha",
"Num Rounds", "Local Epochs", "Batch Size",
"LR", "Weight Decay", "Dropout", "Optimizer"
```

---

### 2. Running proposed approaches:

#### 2.1. FedALALayer0:
```bash
# Small datasets (5 clients)
python run_table6_graphfl.py --groups fedala --datasets MUTAG COX2 --num-clients 5 --layer_idx 0 --seeds 42 123 456 

# Standard datasets (10 clients)
python run_table6_graphfl.py --groups fedala --datasets BZR ENZYMES DD PROTEINS IMDB-BINARY IMDB-MULTI COLLAB --num-clients 10 --layer_idx 0 --seeds 42 123 456
```

#### 2.2. FedALAReg:
**Results Tab:** `regularization_experiments` in the [spreadsheet](https://docs.google.com/spreadsheets/d/1IW_IZTJXF-h-B8g9ly8zyBHfuTcnIyZpIQVH-xSyIZg/edit?usp=sharing)

#### FedALAReg Baseline (no regularization):
```bash
# Small datasets (5 clients)
python run_table6_graphfl.py --groups fedala_reg --datasets MUTAG COX2 --num-clients 5 --lambda-graph 0.0 --seeds 42 123 456

# Standard datasets (10 clients)
python run_table6_graphfl.py --groups fedala_reg --datasets BZR ENZYMES DD PROTEINS IMDB-BINARY IMDB-MULTI COLLAB --num-clients 10 --lambda-graph 0.0 --seeds 42 123 456
```

#### FedALARegL (Laplacian Regularization with λ=1000):
```bash
# for FedALARegL (Laplacian Regularization with λ=1000):

# Small datasets (5 clients)
python run_table6_graphfl.py --groups fedala_reg --datasets MUTAG COX2 --num-clients 5 --lambda-graph 1000 --graph-reg-type laplacian --seeds 42 123 456

# Standard datasets (10 clients)
python run_table6_graphfl.py --groups fedala_reg --datasets BZR ENZYMES DD PROTEINS IMDB-BINARY IMDB-MULTI COLLAB --num-clients 10 --lambda-graph 1000 --graph-reg-type laplacian --seeds 42 123 456
```
#### FedALARegD (Dirichlet Regularization with λ=1000):
```bash
# Small datasets (5 clients)
python run_table6_graphfl.py --groups fedala_reg --datasets MUTAG COX2 --num-clients 5 --lambda-graph 1000 --graph-reg-type dirichlet --seeds 42 123 456

# Standard datasets (10 clients)
python run_table6_graphfl.py --groups fedala_reg --datasets BZR ENZYMES DD PROTEINS IMDB-BINARY IMDB-MULTI COLLAB --num-clients 10 --lambda-graph 1000 --graph-reg-type dirichlet --seeds 42 123 456
```

#### Alternatively:
Configuration files are inside `configs/fedala_regularization_configs/`:
```bash
# Example: Run FedALA with Laplacian regularization on BZR
python -u run_experiments.py --config configs/fedala_regularization_configs/bzr_fedala_laplacian_reg_clean.yaml
```

We ran the experiments in truba in parallel, so instead of preparing differnt configuration files for every setup we just specified the specific parameter's value from the command line. So if wanted to reproduce all of our regularization results from the report, you could use the barbun_submit_all_fedala_experiments.sh file.  

**Output Format:** Same as above.

---


---

<br><br>

---

# Original OpenFGL Documentation

![1301717130101_ pic](https://github.com/zyl24/OpenFGL/assets/59046279/e21b410f-2b5d-4515-8ab5-a176f98805a7)


# Open Federated Graph Learning (OpenFGL)
OpenFGL is a comprehensive, user-friendly algorithm library, complemented by an integrated evaluation platform, designed specifically for researchers in the field of federated graph learning (FGL).

<p align="center">
  <a href="https://arxiv.org/abs/2408.16288">Paper</a> •
  <a href="#Library Highlights">Highlights</a> •
  <a href="https://pypi.org/project/openfgl-lib/">Installation</a> •
  <a href="https://openfgl.readthedocs.io/en/latest/">Docs</a> •
  <a href="#Citation">Citation</a> 
</p>



[![Stars](https://img.shields.io/github/stars/zyl24/OpenFGL.svg?color=orange)](https://github.com/zyl24/OpenFGL/stargazers) ![](https://img.shields.io/github/last-commit/zyl24/OpenFGL) 
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992) -->

 



## Highlights

- 2 FGL Scenarios: Graph-FL and Subgraph-FL
- 10+ FGL Algorithms
- 34 FGL Datasets
- 12 GNN Models
- 5 Downstream Tasks
- Comprehensive FGL Data Property Analysis

## Get Started

```python
import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "your_data_root"


args.dataset = ["Cora"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 10


if True:
    args.fl_algorithm = "fedavg"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedproto"
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]



trainer = FGLTrainer(args)

trainer.train()
```


## Citation
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
```
@misc{li2024openfglcomprehensivebenchmarksfederated,
      title={OpenFGL: A Comprehensive Benchmarks for Federated Graph Learning}, 
      author={Xunkai Li and Yinlin Zhu and Boyang Pang and Guochen Yan and Yeyu Yan and Zening Li and Zhengyu Wu and Wentao Zhang and Rong-Hua Li and Guoren Wang},
      year={2024},
      eprint={2408.16288},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.16288}, 
}
```
