"""
Experiment Runner for FedALA-ARC Evaluation

This script runs two types of experiments:
1. Table 8 replication (generalization across heterogeneity types)
2. Byzantine robustness experiments (the key FedALA-ARC contribution)

Usage:
    # Run Table 8 experiments
    python run_fedalarc_experiments.py --experiment table8 --scenario graph_fl
    
    # Run Byzantine robustness experiments
    python run_fedalarc_experiments.py --experiment byzantine --dataset PROTEINS
    
    # Run all experiments
    python run_fedalarc_experiments.py --experiment all
"""

import os
import sys
import copy
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random

def apply_universal_attack_patch(trainer, args, attack_type, byzantine_ids):
    """
    Injects attacks directly into the message pool before server execution.
    Ensures identical attack logic for ALL algorithms (FedAvg, FedALA, FedALARC).
    """
    if not byzantine_ids or attack_type == 'none':
        return

    # Capture the original execution method
    original_server_execute = trainer.server.execute

    def attacked_execute():
        # Access the message pool where clients deposited their weights
        pool = trainer.message_pool
        
        # Identify which active clients are Byzantine
        sampled_clients = pool.get("sampled_clients", [])
        active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]

        if active_attackers:
            # print(f"  [Attack Injector] Attacking {len(active_attackers)} clients with {attack_type}...")
            
            # --- DEFINE ATTACK MATH HERE ---
            for client_id in active_attackers:
                client_msg = pool[f"client_{client_id}"]
                weights = client_msg["weight"] # This is a list of Parameters/Tensors
                
                attacked_weights = []
                
                if attack_type == 'sign_flip':
                    # Global weights are needed to compute the update direction
                    # gradient = local - global
                    # flipped = global - gradient = global - (local - global) = 2*global - local
                    global_weights = list(trainer.server.task.model.parameters())
                    
                    for w_local, w_global in zip(weights, global_weights):
                        # Ensure we work with data tensors
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        
                        # Apply Sign Flip: w_new = 2 * w_global - w_local
                        w_attacked = 2 * data_global - data_local
                        
                        # Wrap back into Parameter
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'gaussian_noise':
                    # Add large Gaussian noise to weights
                    noise_scale = 10.0
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        noise = torch.randn_like(data_local) * noise_scale
                        w_attacked = data_local + noise
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'zero':
                    # Send zero weights
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        w_attacked = torch.zeros_like(data_local)
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'random':
                    # Send random weights
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        w_attacked = torch.randn_like(data_local)
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                else:
                    # Unknown attack, keep original
                    attacked_weights = weights

                # Update the pool with the malicious weights
                pool[f"client_{client_id}"]["weight"] = attacked_weights

        # Run the actual server aggregation
        original_server_execute()

    # Apply the hook
    trainer.server.execute = attacked_execute


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_experiment(config, seed):
    """
    Run a single experiment with given configuration and seed.
    
    Args:
        config: Dictionary with experiment configuration
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    # Create fresh args
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = seed
    
    # Apply configuration
    # Note: dataset and model must be lists (OpenFGL uses action='append')
    dataset = config['dataset']
    args.dataset = [dataset] if isinstance(dataset, str) else dataset
    args.task = config['task']
    args.scenario = config['scenario']
    args.fl_algorithm = config['algorithm']
    model = config['model']
    args.model = [model] if isinstance(model, str) else model
    args.simulation_mode = config['simulation_mode']
    args.num_clients = config['num_clients']
    args.num_rounds = config['rounds']
    args.num_epochs = config.get('local_steps', 1)
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    args.dropout = config['dropout']
    args.optim = config.get('optimizer', 'adam')
    metrics = config.get('metrics', ['accuracy'])
    args.metrics = [metrics] if isinstance(metrics, str) else metrics
    args.evaluation_mode = config.get('evaluation_mode', 'local_model_on_local_data')
    
    # Simulation-specific parameters
    args.dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
    # skew_alpha is used by the dataset loader for label_skew simulations
    args.skew_alpha = config.get('skew_alpha', config.get('dirichlet_alpha', 0.5))
    
    # Other simulation parameters that might be needed
    args.louvain_resolution = config.get('louvain_resolution', 1.0)
    args.louvain_delta = config.get('louvain_delta', 20)
    args.metis_num_coms = config.get('metis_num_coms', 100)
    
    # FedALA parameters
    args.eta = config.get('eta', 1.0)
    args.layer_idx = config.get('layer_idx', 1)
    args.rand_percent = config.get('rand_percent', 80)
    args.threshold = config.get('threshold', 0.1)
    args.num_pre_loss = config.get('num_pre_loss', 10)
    
    # FedALA-ARC / Byzantine parameters
    args.use_arc = config.get('use_arc', False)
    args.max_byzantine = config.get('max_byzantine', 0)
    args.byzantine_ids = config.get('byzantine_ids', [])
    args.attack_type = config.get('attack_type', 'none')  # Internal attack (disabled)
    args.attack_params = json.dumps(config.get('attack_params', {}))
    
    # Logging
    args.debug = True
    args.log_root = config.get('log_root', './logs_fedalarc')
    args.log_name = f"{config['name']}_seed{seed}"
    
    # Run training
    start_time = time.time()
    trainer = FGLTrainer(args)
    
    # --- external attack block before training starts ---
    # Retrieve the external attack configuration
    real_attack_type = config.get('external_attack_type', 'none')
    byz_ids = config.get('byzantine_ids', [])
    
    # Apply the patch to the trainer
    apply_universal_attack_patch(trainer, args, real_attack_type, byz_ids)
    # ----------------------

    # Patch torch.load for PyTorch 2.6+ compatibility
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    trainer.train()
    running_time = time.time() - start_time
    
    # Extract results
    results = trainer.evaluation_result
    metric_name = 'accuracy' if config['task'] in ['graph_cls', 'node_cls'] else 'mse'
    
    test_acc = results.get(f'best_test_{metric_name}', 0)
    val_acc = results.get(f'best_val_{metric_name}', 0)
    best_round = results.get('best_round', 0)
    
    # Convert to percentage if needed
    if metric_name == 'accuracy':
        if test_acc < 1:
            test_acc *= 100
        if val_acc < 1:
            val_acc *= 100
    
    return {
        'test_acc': test_acc,
        'val_acc': val_acc,
        'best_round': best_round,
        'time': running_time,
        'seed': seed
    }


def run_experiment_with_seeds(config, seeds):
    """Run experiment with multiple seeds and aggregate results"""
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']} | Seed: {seed}")
        print(f"{'='*60}")
        
        result = run_single_experiment(config, seed)
        all_results.append(result)
        
        print(f"Test Acc: {result['test_acc']:.2f}%")
        print(f"Time: {result['time']:.2f}s")
    
    # Aggregate
    test_accs = [r['test_acc'] for r in all_results]
    
    return {
        'name': config['name'],
        'mean_acc': np.mean(test_accs),
        'std_acc': np.std(test_accs),
        'all_accs': test_accs,
        'config': config
    }


# ============================================================
# Table 8 Experiments
# ============================================================

def get_table8_configs(scenario='graph_fl'):
    """Generate configs for Table 8 replication"""
    configs = []
    
    if scenario == 'graph_fl':
        datasets = ['PROTEINS', 'ENZYMES']
        simulations = [
            ('feature_skew', 'graph_fl_feature_skew', {'skew_alpha': 0.5}),
            ('label_skew', 'graph_fl_label_skew', {'dirichlet_alpha': 0.5, 'skew_alpha': 0.5}),
            ('topology_skew', 'graph_fl_topology_skew', {'skew_alpha': 0.5}),
        ]
        algorithms = [
            ('FedAvg', 'fedavg', {}),
            ('FedProx', 'fedprox', {'mu': 0.01}),
            ('FedALA', 'fedala', {'eta': 1.0, 'layer_idx': 1}),
            ('FedALARC', 'fedalarc', {'eta': 1.0, 'layer_idx': 1, 'use_arc': False}),
        ]
        task = 'graph_cls'
        model = 'gin'  # lowercase
        
    else:  # subgraph_fl
        datasets = ['Cora', 'CiteSeer']
        simulations = [
            ('louvain', 'subgraph_fl_louvain', {'skew_alpha': 0.5}),
            ('metis', 'subgraph_fl_metis', {'skew_alpha': 0.5}),
            ('louvain_plus', 'subgraph_fl_louvain_plus', {'skew_alpha': 0.5}),
        ]
        algorithms = [
            ('FedAvg', 'fedavg', {}),
            ('FedProx', 'fedprox', {'mu': 0.01}),
            ('FedALA', 'fedala', {'eta': 1.0, 'layer_idx': 1}),
            ('FedALARC', 'fedalarc', {'eta': 1.0, 'layer_idx': 1, 'use_arc': False}),
        ]
        task = 'node_cls'
        model = 'gcn'  # lowercase
    
    # Generate all combinations
    for dataset in datasets:
        for sim_name, sim_mode, sim_params in simulations:
            for alg_name, alg_id, alg_params in algorithms:
                config = {
                    'name': f"{dataset}_{sim_name}_{alg_name}",
                    'dataset': dataset,
                    'scenario': scenario,
                    'task': task,
                    'model': model,
                    'algorithm': alg_id,
                    'simulation_mode': sim_mode,
                    'num_clients': 5,
                    'rounds': 100,
                    'local_steps': 1,
                    'batch_size': 128,
                    'lr': 0.001,
                    'weight_decay': 0.0005,
                    'dropout': 0.5,
                    'optimizer': 'adam',
                    'metrics': ['accuracy'],
                    'skew_alpha': sim_params.get('skew_alpha', 0.5),
                    'dirichlet_alpha': sim_params.get('dirichlet_alpha', 0.5),
                    **alg_params,
                }
                # Add any additional sim_params that aren't skew_alpha or dirichlet_alpha
                for k, v in sim_params.items():
                    if k not in config:
                        config[k] = v
                configs.append(config)
    
    return configs


# ============================================================
# Byzantine Robustness Experiments
# ============================================================

def get_byzantine_configs(dataset='PROTEINS'):
    """Generate configs for Byzantine robustness experiments"""
    configs = []
    
    # Base config
    base = {
        'dataset': dataset,
        'scenario': 'graph_fl',
        'task': 'graph_cls',
        'model': 'gin',  # lowercase as expected by OpenFGL
        'simulation_mode': 'graph_fl_label_skew',
        'dirichlet_alpha': 0.5,
        'skew_alpha': 0.5,  # Add skew_alpha to match dirichlet_alpha
        'num_clients': 10,
        'rounds': 50,
        'local_steps': 1,
        'batch_size': 128,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'dropout': 0.5,
        'optimizer': 'adam',
        'metrics': ['accuracy'],
        # FedALA params
        'eta': 1.0,
        'layer_idx': 1,
        'rand_percent': 80,
        'threshold': 0.1,
        'num_pre_loss': 10,
    }
    
    # Attack scenarios
    attacks = [
        ('no_attack', 0, [], 'none'),
        ('f1_sign_flip', 1, [0], 'sign_flip'),
        ('f2_sign_flip', 2, [0, 1], 'sign_flip'),
        ('f3_sign_flip', 3, [0, 1, 2], 'sign_flip'),
    ]
    
    # Methods
    methods = [
        ('FedAvg', 'fedavg', False, 0),
        ('FedALA', 'fedala', False, 0),
        ('FedALARC-f1', 'fedalarc', True, 1),
        ('FedALARC-f2', 'fedalarc', True, 2),
        ('FedALARC-f3', 'fedalarc', True, 3),
    ]
    
    # Generate configs
    for attack_name, num_byz, byz_ids, attack_type in attacks:
        for method_name, alg, use_arc, max_byz in methods:
            config = {
                **base,
                'name': f"{dataset}_{attack_name}_{method_name}",
                'algorithm': alg,
                'use_arc': use_arc,
                'max_byzantine': max_byz,  # Keeping this because defense needs to know f
                'byzantine_ids': byz_ids,  # Keeping this because patch needs to know who to attack
                
                # this is the critical addition
                # We store the REAL attack type in a custom field for our patch
                'external_attack_type': attack_type, 
                
                # We set the internal attack type to 'none' so FedALARCClient 
                # doesn't apply the attack itself. We will apply the attack.
                'attack_type': 'none',     
            }
            configs.append(config)
            
    return configs

# ============================================================
# Results Formatting
# ============================================================

def format_table8_results(results, scenario):
    """Format results as Table 8"""
    # Group by dataset and simulation
    df_data = []
    
    for result in results:
        cfg = result['config']
        df_data.append({
            'Dataset': cfg['dataset'],
            'Simulation': cfg['simulation_mode'].split('_')[-1],
            'Algorithm': cfg['algorithm'],
            'Accuracy': f"{result['mean_acc']:.2f}±{result['std_acc']:.2f}",
            'Mean': result['mean_acc'],
            'Std': result['std_acc'],
        })
    
    df = pd.DataFrame(df_data)
    
    # Pivot to match Table 8 format
    pivot = df.pivot_table(
        index=['Dataset', 'Algorithm'],
        columns='Simulation',
        values='Accuracy',
        aggfunc='first'
    )
    
    return pivot


def format_byzantine_results(results):
    """Format results as Byzantine robustness table"""
    df_data = []
    
    for result in results:
        cfg = result['config']
        
        # FIX: Use 'external_attack_type' instead of 'attack_type'
        real_attack = cfg.get('external_attack_type', 'none')
        num_attackers = len(cfg.get('byzantine_ids', []))
        
        if real_attack == 'none' or num_attackers == 0:
            attack_label = 'No Attack'
        else:
            attack_label = f"f={num_attackers} {real_attack}"
        
        # Extract method name from the config name (last part after underscore)
        method_name = cfg['name'].split('_')[-1]
        
        df_data.append({
            'Dataset': cfg['dataset'],
            'Attack': attack_label,
            'Method': method_name,
            'Accuracy': f"{result['mean_acc']:.2f}±{result['std_acc']:.2f}",
            'Mean': result['mean_acc'],
        })
    
    df = pd.DataFrame(df_data)
    
    # Pivot to create the table with attacks as columns
    pivot = df.pivot_table(
        index=['Dataset', 'Method'],
        columns='Attack',
        values='Accuracy',
        aggfunc='first'
    )
    
    # Reorder columns to have 'No Attack' first, then sorted attack columns
    cols = list(pivot.columns)
    if 'No Attack' in cols:
        cols.remove('No Attack')
        cols = ['No Attack'] + sorted(cols)
        pivot = pivot[cols]
    
    return pivot


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FedALA-ARC Experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['table8', 'byzantine', 'all'],
                        help='Which experiment to run')
    parser.add_argument('--scenario', type=str, default='graph_fl',
                        choices=['graph_fl', 'subgraph_fl'],
                        help='Scenario for Table 8')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='Dataset for Byzantine experiments')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Random seeds')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configs without running')
    
    args = parser.parse_args()
    
    all_results = []
    
    # Table 8 experiments
    if args.experiment in ['table8', 'all']:
        print("\n" + "="*70)
        print("TABLE 8 EXPERIMENTS: Generalization Performance")
        print("="*70)
        
        configs = get_table8_configs(args.scenario)
        
        if args.dry_run:
            print(f"Would run {len(configs)} experiments:")
            for cfg in configs:
                print(f"  - {cfg['name']}")
        else:
            for config in configs:
                result = run_experiment_with_seeds(config, args.seeds)
                all_results.append(result)
            
            # Print formatted table
            table = format_table8_results(all_results, args.scenario)
            print("\n" + "="*70)
            print("TABLE 8 RESULTS")
            print("="*70)
            print(table.to_string())
    
    # Byzantine experiments
    if args.experiment in ['byzantine', 'all']:
        print("\n" + "="*70)
        print("BYZANTINE ROBUSTNESS EXPERIMENTS")
        print("="*70)
        
        configs = get_byzantine_configs(args.dataset)
        
        if args.dry_run:
            print(f"Would run {len(configs)} experiments:")
            for cfg in configs:
                ext_attack = cfg.get('external_attack_type', 'none')
                byz_ids = cfg.get('byzantine_ids', [])
                print(f"  - {cfg['name']} | attack={ext_attack} | byzantine_ids={byz_ids}")
        else:
            byzantine_results = []
            for config in configs:
                result = run_experiment_with_seeds(config, args.seeds)
                byzantine_results.append(result)
            
            # Print formatted table
            table = format_byzantine_results(byzantine_results)
            print("\n" + "="*70)
            print("BYZANTINE ROBUSTNESS RESULTS")
            print("="*70)
            print(table.to_string())
    
    # Save results
    if not args.dry_run and all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/fedalarc_results_{timestamp}.json"
        os.makedirs('results', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump([{
                'name': r['name'],
                'mean_acc': r['mean_acc'],
                'std_acc': r['std_acc'],
                'all_accs': r['all_accs'],
            } for r in all_results], f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()