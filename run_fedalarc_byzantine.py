"""
FedALARC Byzantine Robustness Experiments
Reproduces key results from ARC paper in OpenFGL setting
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from openfgl.config import args
from openfgl.utils.basic_utils import *
import yaml

def run_byzantine_experiment():
    """
    Experiment 1: Byzantine Workers (varying f)
    Tests robustness against sign-flip attacks
    """
    
    # Load base config
    with open('fedalarc.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Base settings
    args.scenario = "subgraph_fl"
    args.dataset = ["Cora"]
    args.task = "node_cls"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10
    args.num_rounds = 200
    args.dirichlet_alpha = 0.5  # Medium heterogeneity
    
    results = {}
    
    # Test with varying number of Byzantine workers
    for f in [0, 1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Testing with f={f} Byzantine workers")
        print(f"{'='*60}\n")
        
        # --- Run FedALA (baseline, no ARC) ---
        print(f"[1/2] Running FedALA (baseline)...")
        args.fl_algorithm = "fedala"
        args.use_arc = False
        args.byzantine_ids = list(range(f))  # First f clients are Byzantine
        args.attack_type = "sign_flip"
        
        acc_baseline = run_experiment(args)
        
        # --- Run FedALARC (with ARC) ---
        print(f"\n[2/2] Running FedALARC (with ARC)...")
        args.fl_algorithm = "fedalarc"
        args.use_arc = True
        args.max_byzantine = f
        args.byzantine_ids = list(range(f))
        args.attack_type = "sign_flip"
        
        acc_arc = run_experiment(args)
        
        # Store results
        results[f] = {
            'FedALA': acc_baseline,
            'FedALARC': acc_arc,
            'improvement': acc_arc - acc_baseline
        }
        
        print(f"\n--- Results for f={f} ---")
        print(f"FedALA:   {acc_baseline:.2f}%")
        print(f"FedALARC: {acc_arc:.2f}%")
        print(f"Gain:     +{acc_arc - acc_baseline:.2f}%")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Byzantine Robustness")
    print(f"{'='*60}")
    print(f"{'f':>5} | {'FedALA':>10} | {'FedALARC':>10} | {'Gain':>10}")
    print(f"{'-'*60}")
    for f, res in results.items():
        print(f"{f:>5} | {res['FedALA']:>9.2f}% | {res['FedALARC']:>9.2f}% | {res['improvement']:>9.2f}%")
    
    return results


def run_heterogeneity_experiment():
    """
    Experiment 2: Heterogeneity Sweep
    Tests how ARC performs under different heterogeneity levels
    """
    
    args.scenario = "subgraph_fl"
    args.dataset = ["Cora"]
    args.task = "node_cls"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10
    args.num_rounds = 200
    
    # Fixed: 1 Byzantine worker
    f = 1
    args.byzantine_ids = [0]
    args.attack_type = "sign_flip"
    
    results = {}
    alpha_values = {
        'Low (α=2)': 2.0,
        'Medium (α=0.5)': 0.5,
        'High (α=0.1)': 0.1
    }
    
    for name, alpha in alpha_values.items():
        print(f"\n{'='*60}")
        print(f"Testing {name} heterogeneity")
        print(f"{'='*60}\n")
        
        args.dirichlet_alpha = alpha
        
        # Run FedALA
        args.fl_algorithm = "fedala"
        args.use_arc = False
        acc_baseline = run_experiment(args)
        
        # Run FedALARC
        args.fl_algorithm = "fedalarc"
        args.use_arc = True
        args.max_byzantine = f
        acc_arc = run_experiment(args)
        
        results[name] = {
            'FedALA': acc_baseline,
            'FedALARC': acc_arc,
            'improvement': acc_arc - acc_baseline
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Heterogeneity Impact (f=1)")
    print(f"{'='*60}")
    print(f"{'Heterogeneity':>20} | {'FedALA':>10} | {'FedALARC':>10} | {'Gain':>10}")
    print(f"{'-'*60}")
    for name, res in results.items():
        print(f"{name:>20} | {res['FedALA']:>9.2f}% | {res['FedALARC']:>9.2f}% | {res['improvement']:>9.2f}%")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("FedALARC: Byzantine Robustness Evaluation")
    print("="*60)
    
    # Run experiments
    byzantine_results = run_byzantine_experiment()
    hetero_results = run_heterogeneity_experiment()
    
    print("\nAll experiments completed!")