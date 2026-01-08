"""
Configurable script to reproduce OpenFGL experiments
Reads configurations from YAML files
"""

import os
import random
import numpy as np
import torch
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import time
import copy
import yaml
import argparse
import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(exp_config, seed):
    """Run a single experiment with given config and seed"""
    set_seed(seed)

    # Fresh args
    args = copy.deepcopy(config.args)
    args.root = os.path.join(os.getcwd(), "data")

    # Apply experiment config
    args.dataset = exp_config['dataset']
    args.task = exp_config['task']
    args.scenario = exp_config['scenario']
    args.fl_algorithm = exp_config['algorithm']
    args.model = exp_config['model']

    # Simulation
    args.simulation_mode = exp_config['simulation_mode']
    args.num_clients = exp_config['num_clients']

    # IMPORTANT for label_skew sims
    args.dirichlet_alpha = exp_config.get("dirichlet_alpha", args.dirichlet_alpha)

    # Training
    args.num_rounds = exp_config['num_rounds']
    args.num_epochs = exp_config.get('local_epochs', exp_config.get('num_epochs', args.num_epochs))
    args.batch_size = exp_config['batch_size']
    args.lr = exp_config['lr']
    args.weight_decay = exp_config['weight_decay']
    args.dropout = exp_config['dropout']
    args.optim = exp_config.get('optim', 'adam')

    # # FedALA specific parameters
    args.eta = exp_config.get('eta', 1.0)
    args.threshold = exp_config.get('threshold', 0.1)
    args.num_pre_loss = exp_config.get('num_pre_loss', 10)
    args.rand_percent = exp_config.get('rand_percent', 80)
    args.layer_idx = exp_config.get('layer_idx', 1)
    args.lambda_graph = exp_config.get('lambda_graph', 0.0)
    args.graph_reg_type = exp_config.get('graph_reg_type', 'laplacian')

    # Eval/metrics
    args.metrics = exp_config.get('metrics', ['accuracy'])
    args.evaluation_mode = exp_config.get('evaluation_mode', args.evaluation_mode)

    # Logging
    args.comm_cost = exp_config['algorithm'] != 'isolate'
    args.debug = True
    args.log_root = "./logs_reproduce"
    args.log_name = f"{exp_config['name']}_seed{seed}"
    args.seed = seed

    if args.simulation_mode in ["graph_fl_label_skew", "subgraph_fl_label_skew"]:
        args.skew_alpha = exp_config.get("skew_alpha", 1.0)

    # Apply FedALA parameters if present in config #newaddition
    if 'eta' in exp_config:
        args.eta = exp_config['eta']
    if 'layer_idx' in exp_config:
        args.layer_idx = exp_config['layer_idx']
    if 'rand_percent' in exp_config:
        args.rand_percent = exp_config['rand_percent']
    if 'threshold' in exp_config:
        args.threshold = exp_config['threshold']
    if 'num_pre_loss' in exp_config:
        args.num_pre_loss = exp_config['num_pre_loss']
    
    # FedALA+ specific parameters
    if 'use_disagreement' in exp_config:
        args.use_disagreement = exp_config['use_disagreement']
    if 'selection_frequency' in exp_config:
        args.selection_frequency = exp_config['selection_frequency']
    if 'min_disagreement_samples' in exp_config:
        args.min_disagreement_samples = exp_config['min_disagreement_samples']    
    
    if 'layer_idx' in exp_config:
        args.layer_idx = exp_config['layer_idx']
        
    # Train
    start_time = time.time()
    trainer = FGLTrainer(args)

    # Patch torch.load for PyTorch 2.6+ issue
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load

    trainer.train()
    running_time = time.time() - start_time
    
    # Get results
    results = trainer.evaluation_result
    
    # Extract metrics based on task
    if exp_config['task'] in ['graph_cls', 'node_cls']:
        metric_name = 'accuracy'
    elif exp_config['task'] == 'graph_reg':
        metric_name = exp_config.get('metrics', ['mse'])[0]
    else:
        metric_name = 'accuracy'
    
    test_metric = results.get(f'best_test_{metric_name}', 0)
    val_metric = results.get(f'best_val_{metric_name}', 0)
    best_round = results.get('best_round', 0)
    
    # NEW: Try to get final round metrics (vs best) for convergence analysis
    final_test_metric = results.get(f'final_test_{metric_name}', test_metric)
    final_val_metric = results.get(f'final_val_{metric_name}', val_metric)

    # Get communication cost from logger pickle file
    log_path = os.path.join(args.log_root, f"{args.log_name}.pkl")
    total_comm_kb = 0
    per_round_metrics = []  # NEW: for convergence analysis
    
    if os.path.exists(log_path) and exp_config['algorithm'] != 'isolate':
        with open(log_path, 'rb') as f:
            log_data = pickle.load(f)
            avg_cost_per_round = log_data.get('avg_cost_per_round', 0)
            total_comm_kb = avg_cost_per_round
            # NEW: Try to extract per-round data if available
            per_round_metrics = log_data.get('per_round_metrics', [])
            # Debug: print available keys to discover more data
            # print(f"Log data keys: {log_data.keys()}")
    
    total_comm_mb = total_comm_kb / 1024  # Convert KB to MB
    
    # Convert to percentage if accuracy
    if metric_name == 'accuracy':
        if test_metric < 1:
            test_metric *= 100
        if val_metric < 1:
            val_metric *= 100
        if final_test_metric < 1:
            final_test_metric *= 100
        if final_val_metric < 1:
            final_val_metric *= 100
    
    print("-" * 50)
    print(f"curr_round: {exp_config['num_rounds']-1}  curr_val_{metric_name}: {final_val_metric:.4f}\tcurr_test_{metric_name}: {final_test_metric:.4f}")
    print(f"best_round: {best_round}  best_val_{metric_name}: {val_metric:.4f}\tbest_test_{metric_name}: {test_metric:.4f}")
    print("-" * 50)
    print(f"Best Round: {best_round}")
    print(f"Best Val {metric_name}: {val_metric:.2f}")
    print(f"Best Test {metric_name}: {test_metric:.2f}")
    print(f"Running Time: {running_time:.2f}s")
    if exp_config['algorithm'] != 'isolate':
        print(f"Communication Cost: {total_comm_mb:.2f} MB")
    else:
        print("Communication Cost: N/A (local training only)")
    
    return {
        'test_metric': test_metric,
        'val_metric': val_metric,  # NEW
        'best_round': best_round,  # NEW
        'final_test_metric': final_test_metric,  # NEW
        'final_val_metric': final_val_metric,  # NEW
        'running_time': running_time,
        'comm_cost_mb': total_comm_mb,
        'per_round_metrics': per_round_metrics,  # NEW (may be empty)
        'seed': seed,  # NEW: track which seed produced this
    }


def run_experiments(exp_config, seeds=[42, 123, 456]):
    """Run experiments with multiple seeds"""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_config['name']}")
    print("=" * 70)
    print(f"Dataset: {exp_config['dataset']}")
    print(f"Algorithm: {exp_config['algorithm']}")
    print(f"Scenario: {exp_config['scenario']}")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    
    all_results = []  # NEW: store full results per seed
    all_test_metrics = []
    all_val_metrics = []  # NEW
    all_best_rounds = []  # NEW
    all_times = []
    all_comm_costs = []
    
    for seed in seeds:
        result = run_experiment(exp_config, seed)
        all_results.append(result)  # NEW
        all_test_metrics.append(result['test_metric'])
        all_val_metrics.append(result['val_metric'])  # NEW
        all_best_rounds.append(result['best_round'])  # NEW
        all_times.append(result['running_time'])
        all_comm_costs.append(result['comm_cost_mb'])
    
    # Calculate statistics
    mean_metric = np.mean(all_test_metrics)
    std_metric = np.std(all_test_metrics)
    mean_val_metric = np.mean(all_val_metrics)  # NEW
    std_val_metric = np.std(all_val_metrics)  # NEW
    mean_best_round = np.mean(all_best_rounds)  # NEW
    std_best_round = np.std(all_best_rounds)  # NEW
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)
    mean_comm = np.mean(all_comm_costs)
    std_comm = np.std(all_comm_costs)
    
    metric_name = exp_config.get('metrics', ['accuracy'])[0]
    
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {exp_config['name']}")
    print("=" * 70)
    print(f"Test {metric_name.upper()}: {mean_metric:.2f} ± {std_metric:.2f}")
    print(f"Val {metric_name.upper()}: {mean_val_metric:.2f} ± {std_val_metric:.2f}")  # NEW
    print(f"Best Round: {mean_best_round:.1f} ± {std_best_round:.1f}")  # NEW
    print(f"Running Time (s): {mean_time:.2f} ± {std_time:.2f}")
    print(f"Communication Cost (MB): {mean_comm:.2f} ± {std_comm:.2f}")
    print(f"Individual Test Metrics: {[f'{m:.2f}' for m in all_test_metrics]}")
    print("=" * 70 + "\n")
    
    return {
        'name': exp_config['name'],
        # Test metrics
        'mean_metric': mean_metric,
        'std_metric': std_metric,
        'test_metrics': all_test_metrics,
        # NEW: Validation metrics
        'mean_val_metric': mean_val_metric,
        'std_val_metric': std_val_metric,
        'val_metrics': all_val_metrics,
        # NEW: Convergence info
        'mean_best_round': mean_best_round,
        'std_best_round': std_best_round,
        'best_rounds': all_best_rounds,
        # Time
        'mean_time': mean_time,
        'std_time': std_time,
        'times': all_times,
        # Communication
        'mean_comm': mean_comm,
        'std_comm': std_comm,
        'comm_costs': all_comm_costs,
        # NEW: Full per-seed results for detailed analysis
        'per_seed_results': all_results,
        # Config
        'config': exp_config,
    }


def print_google_sheets_format(result):
    """Print results in tab-separated format for easy copy-paste to Google Sheets"""
    cfg = result['config']
    
    dataset = cfg.get('dataset', ['N/A'])
    if isinstance(dataset, list):
        dataset = dataset[0]
    
    model = cfg.get('model', ['N/A'])
    if isinstance(model, list):
        model = model[0]
    
    # Build data row
    fields = [
        dataset,
        cfg.get('scenario', 'N/A'),
        cfg.get('algorithm', 'N/A'),
        "",  # Accuracy Reported (fill from paper)
        f"{result['mean_metric']:.2f}",
        f"{result['std_metric']:.2f}",
        f"{result['mean_val_metric']:.2f}",  # NEW
        f"{result['mean_best_round']:.1f}",  # NEW
        f"{result['mean_comm']:.2f}",
        f"{result['mean_time']:.2f}",
        cfg.get('task', 'N/A'),
        model,
        cfg.get('simulation_mode', 'N/A'),
        cfg.get('num_clients', 'N/A'),
        cfg.get('dirichlet_alpha', 'N/A'),
        cfg.get('num_rounds', 'N/A'),
        cfg.get('local_epochs', cfg.get('num_epochs', 'N/A')),
        cfg.get('batch_size', 'N/A'),
        cfg.get('lr', 'N/A'),
        cfg.get('weight_decay', 'N/A'),
        cfg.get('dropout', 'N/A'),
        cfg.get('optim', 'N/A'),
    ]
    
    headers = [
        "Dataset", "Scenario", "Algorithm", "Accuracy Reported",
        "Test Acc", "Std Dev", "Val Acc", "Best Round",
        "Comm (MB)", "Time (s)", "Task", "Model",
        "Simulation Mode", "Num Clients", "Dirichlet Alpha",
        "Num Rounds", "Local Epochs", "Batch Size",
        "LR", "Weight Decay", "Dropout", "Optimizer"
    ]
    
    print("\n" + "=" * 100)
    print("GOOGLE SHEETS FORMAT")
    print("=" * 100)
    print("\nHeaders:")
    print("\t".join(headers))
    print("\n>>> COPY THIS LINE <<<")
    print("\t".join(str(f) for f in fields))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run OpenFGL experiments from config files')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config file (YAML)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Random seeds to use (overrides config file)')
    
    args = parser.parse_args()
    
    # Load configuration
    exp_config = load_config(args.config)
    
    # Get seeds
    seeds = args.seeds if args.seeds is not None else exp_config.get('seeds', [42])
    
    # Run experiments
    result = run_experiments(exp_config, seeds=seeds)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Test Acc (%)':<15} {'Val Acc (%)':<15} {'Best Rnd':<10} {'Time (s)':<15} {'Comm (MB)':<15}")
    print("-" * 100)
    print(f"{result['name']:<30} "
          f"{result['mean_metric']:.2f}±{result['std_metric']:.2f}      "
          f"{result['mean_val_metric']:.2f}±{result['std_val_metric']:.2f}      "
          f"{result['mean_best_round']:.0f}±{result['std_best_round']:.0f}       "
          f"{result['mean_time']:.2f}±{result['std_time']:.2f}      "
          f"{result['mean_comm']:.2f}±{result['std_comm']:.2f}")
    print("=" * 70)
    
    print_google_sheets_format(result)
