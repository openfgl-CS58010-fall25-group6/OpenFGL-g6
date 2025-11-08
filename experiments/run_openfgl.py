import sys
from pathlib import Path

# --- FIX START ---
# Add the parent directory to the system path so Python can find 'openfgl'
# This assumes the 'openfgl' folder is one level up from the current script.
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- FIX END ---

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