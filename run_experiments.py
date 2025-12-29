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

    # Eval/metrics
    args.metrics = exp_config.get('metrics', ['accuracy'])
    args.evaluation_mode = exp_config.get('evaluation_mode', args.evaluation_mode)

    # Logging
    # args.comm_cost = True
    args.comm_cost = exp_config['algorithm'] != 'isolate'
    args.debug = True
    args.log_root = "./logs_reproduce"
    args.log_name = f"{exp_config['name']}_seed{seed}"
    args.seed = seed

    if args.simulation_mode in ["graph_fl_label_skew", "subgraph_fl_label_skew"]:
        args.skew_alpha = exp_config.get("skew_alpha", 1.0)
    
    # Train
    start_time = time.time()
    trainer = FGLTrainer(args)

    # Patch torch.load for PyTorch 2.6+ issue
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    trainer.train()
    running_time = time.time() - start_time
    
    # Get results
    results = trainer.evaluation_result
    
    # Extract metrics based on task
    if exp_config['task'] in ['graph_cls', 'node_cls']:
        metric_name = 'accuracy'
        test_metric = results.get(f'best_test_{metric_name}', 0)
        val_metric = results.get(f'best_val_{metric_name}', 0)
    elif exp_config['task'] == 'graph_reg':
        metric_name = exp_config.get('metrics', ['mse'])[0]
        test_metric = results.get(f'best_test_{metric_name}', 0)
        val_metric = results.get(f'best_val_{metric_name}', 0)
    else:
        test_metric = 0
        val_metric = 0
    
    best_round = results.get('best_round', 0)

    # Get communication cost from logger pickle file
    log_path = os.path.join(args.log_root, f"{args.log_name}.pkl")
    total_comm_kb = 0
    
    if os.path.exists(log_path) and exp_config['algorithm'] != 'isolate':
        with open(log_path, 'rb') as f:
            log_data = pickle.load(f)
            avg_cost_per_round = log_data.get('avg_cost_per_round', 0)
            total_comm_kb = avg_cost_per_round
    
    total_comm_mb = total_comm_kb / 1024  # Convert KB to MB
    
    # Convert to percentage if accuracy
    if metric_name == 'accuracy' and test_metric < 1:
        test_metric *= 100
        val_metric *= 100
    
    print(f"Best Round: {best_round}")
    print(f"Best Val {metric_name}: {val_metric:.2f}")
    print(f"Best Test {metric_name}: {test_metric:.2f}")
    print(f"Running Time: {running_time:.2f}s")
    if exp_config['algorithm'] != 'isolate':
        print(f"Communication Cost: {total_comm_mb:.2f} MB")
    else:
        print(f"Communication Cost: N/A (local training only)")
    
    return {
        'test_metric': test_metric,
        'running_time': running_time,
        'comm_cost_mb': total_comm_mb,
    }


def run_experiments(exp_config, seeds=[42, 123, 456]):
    """Run experiments with multiple seeds"""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_config['name']}")
    print("="*70)
    print(f"Dataset: {exp_config['dataset']}")
    print(f"Algorithm: {exp_config['algorithm']}")
    print(f"Scenario: {exp_config['scenario']}")
    print(f"Seeds: {seeds}")
    print("="*70)
    
    all_test_metrics = []
    all_times = []
    all_comm_costs = []
    
    for seed in seeds:
        result = run_experiment(exp_config, seed)
        all_test_metrics.append(result['test_metric'])
        all_times.append(result['running_time'])
        all_comm_costs.append(result['comm_cost_mb'])
    
    # Calculate statistics
    mean_metric = np.mean(all_test_metrics)
    std_metric = np.std(all_test_metrics)
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)
    mean_comm = np.mean(all_comm_costs)
    std_comm = np.std(all_comm_costs)
    
    metric_name = exp_config.get('metrics', ['accuracy'])[0]
    
    print("\n" + "="*70)
    print(f"FINAL RESULTS: {exp_config['name']}")
    print("="*70)
    print(f"Test {metric_name.upper()}: {mean_metric:.2f} ± {std_metric:.2f}")
    print(f"Running Time (s): {mean_time:.2f} ± {std_time:.2f}")
    print(f"Communication Cost (MB): {mean_comm:.2f} ± {std_comm:.2f}")
    print(f"Individual Test Metrics: {[f'{r:.2f}' for r in all_test_metrics]}")
    print("="*70 + "\n")
    
    return {
        'name': exp_config['name'],
        'mean_metric': mean_metric,
        'std_metric': std_metric,
        'mean_time': mean_time,
        'std_time': std_time,
        'mean_comm': mean_comm,
        'std_comm': std_comm,
        'test_metrics': all_test_metrics,
        'times': all_times,
        'comm_costs': all_comm_costs,
        'config': exp_config  # Store config for later use
    }


def print_google_sheets_format(result):
    """Print results in tab-separated format for easy copy-paste to Google Sheets"""
    config = result['config']
    
    # Extract values with proper defaults
    dataset = config.get('dataset', ['N/A'])[0] if isinstance(config.get('dataset'), list) else config.get('dataset', 'N/A')
    scenario = config.get('scenario', 'N/A')
    algorithm = config.get('algorithm', 'N/A')
    accuracy_reported = ""  # You'll fill this manually from the paper
    accuracy = f"{result['mean_metric']:.2f}"
    std_dev = f"{result['std_metric']:.2f}"
    comm_cost = f"{result['mean_comm']:.2f}"
    time_seconds = f"{result['mean_time']:.2f}"
    task = config.get('task', 'N/A')
    model = config.get('model', ['N/A'])[0] if isinstance(config.get('model'), list) else config.get('model', 'N/A')
    simulation_mode = config.get('simulation_mode', 'N/A')
    num_clients = config.get('num_clients', 'N/A')
    dirichlet_alpha = config.get('dirichlet_alpha', 'N/A')
    num_rounds = config.get('num_rounds', 'N/A')
    local_epochs = config.get('local_epochs', config.get('num_epochs', 'N/A'))
    batch_size = config.get('batch_size', 'N/A')
    lr = config.get('lr', 'N/A')
    weight_decay = config.get('weight_decay', 'N/A')
    dropout = config.get('dropout', 'N/A')
    optimizer = config.get('optim', 'N/A')
    
    # Print header
    print("\n" + "="*100)
    print("GOOGLE SHEETS FORMAT - COPY THE LINE BELOW (tab-separated)")
    print("="*100)
    
    # Print column names (for reference)
    header = "\t".join([
        "Dataset",
        "Scenario", 
        "Algorithm",
        "Accuracy Reported",
        "Accuracy",
        "Std Dev (acc)",
        "Communication Cost (MB)",
        "Time (seconds)",
        "Task",
        "Model",
        "Simulation Mode",
        "Num Clients",
        "Dirichlet Alpha",
        "Num Rounds",
        "Local Epochs",
        "Batch Size",
        "LR",
        "Weight Decay",
        "Dropout",
        "Optimizer"
    ])
    print("\nColumn Headers (for reference):")
    print(header)
    
    # Print the actual data row
    data_row = "\t".join([
        str(dataset),
        str(scenario),
        str(algorithm),
        str(accuracy_reported),  # Empty - fill from paper
        str(accuracy),
        str(std_dev),
        str(comm_cost),
        str(time_seconds),
        str(task),
        str(model),
        str(simulation_mode),
        str(num_clients),
        str(dirichlet_alpha),
        str(num_rounds),
        str(local_epochs),
        str(batch_size),
        str(lr),
        str(weight_decay),
        str(dropout),
        str(optimizer)
    ])
    
    print("\n>>> COPY THIS LINE TO GOOGLE SHEETS <<<")
    print(data_row)
    print("="*100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run OpenFGL experiments from config files')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to experiment config file (YAML)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Random seeds to use (overrides config file)')
    
    args = parser.parse_args()
    
    # Load configuration
    exp_config = load_config(args.config)
    
    # Get seeds from command line or config file
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = exp_config.get('seeds', [42])  # Default to [42] if not specified
    
    # Run experiments
    result = run_experiments(exp_config, seeds=seeds)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Experiment':<30} {'Accuracy (%)':<20} {'Time (s)':<20} {'Comm (MB)':<20}")
    print("-"*70)
    print(f"{result['name']:<30} "
          f"{result['mean_metric']:.2f}±{result['std_metric']:.2f}        "
          f"{result['mean_time']:.2f}±{result['std_time']:.2f}        "
          f"{result['mean_comm']:.2f}±{result['std_comm']:.2f}")
    print("="*70)
    
    # Print Google Sheets format
    print_google_sheets_format(result)
