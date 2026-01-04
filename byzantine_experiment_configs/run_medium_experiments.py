"""
LIGHTWEIGHT EXPERIMENT RUNNER - Medium 20-round version

Balanced version for validation:
- 20 rounds (vs 30 in full, 10 in quick)
- Default 1 seed (use --seeds 0 1 2 for confidence intervals)
- 5 methods: FedAvg, FedProx, FedALA, FedALARC-f2, FedALARC-f3
- 3 attack scenarios: no attack, f=2, f=3
- Total: 15 experiments
- ~15-20 min with 1 seed
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random

_attack_call_count = 0
_attack_applied_count = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_attack_patch(trainer, attack_type, byzantine_ids, attack_strength=1.0):
    """Simple attack patch - modifies weights before server aggregation"""
    global _attack_call_count, _attack_applied_count
    _attack_call_count = 0
    _attack_applied_count = 0
    
    if not byzantine_ids or attack_type == 'none':
        return

    original_server_execute = trainer.server.execute

    def attacked_execute():
        global _attack_call_count, _attack_applied_count
        _attack_call_count += 1
        
        pool = trainer.message_pool
        sampled_clients = pool.get("sampled_clients", [])
        active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]

        if active_attackers:
            _attack_applied_count += 1
            
            for client_id in active_attackers:
                client_msg = pool[f"client_{client_id}"]
                weights = client_msg["weight"]
                attacked_weights = []
                
                if attack_type == 'scaled_sign_flip':
                    global_weights = list(trainer.server.task.model.parameters())
                    for w_local, w_global in zip(weights, global_weights):
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        gradient = data_local - data_global
                        w_attacked = data_global - attack_strength * gradient
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                else:
                    attacked_weights = weights

                pool[f"client_{client_id}"]["weight"] = attacked_weights

        original_server_execute()

    trainer.server.execute = attacked_execute


def run_single_experiment(config, seed):
    """Run single experiment"""
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = seed
    
    # Apply config - handle special cases
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
    args.num_epochs = config['local_epochs']
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    args.dropout = config['dropout']
    args.optim = config['optimizer']
    metrics = config['metrics']
    args.metrics = [metrics] if isinstance(metrics, str) else metrics
    args.evaluation_mode = config['evaluation_mode']
    
    # Simulation params
    args.dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
    args.skew_alpha = config.get('dirichlet_alpha', 0.5)  # ← FIX: Add this
    args.louvain_resolution = 1.0
    args.louvain_delta = 20
    args.metis_num_coms = 100
    
    # FedALA params
    args.eta = config.get('eta', 1.0)
    args.layer_idx = config.get('layer_idx', 1)
    args.rand_percent = config.get('rand_percent', 80)
    args.threshold = config.get('threshold', 0.1)
    args.num_pre_loss = config.get('num_pre_loss', 10)
    
    # FedALARC params
    args.use_arc = config.get('use_arc', False)
    args.max_byzantine = config.get('max_byzantine', 0)
    args.byzantine_ids = config.get('byzantine_ids', [])
    args.attack_type = 'none'
    args.attack_params = '{}'
    
    # Logging
    args.debug = False  # Less verbose for quick runs
    args.log_root = './logs_quick'
    args.log_name = f"{config['name']}_s{seed}"
    
    print(f"[{config['name']}] Seed {seed}: ", end='', flush=True)
    
    trainer = FGLTrainer(args)
    
    # Apply attack
    attack_type = config.get('external_attack_type', 'none')
    byz_ids = config.get('byzantine_ids', [])
    strength = config.get('attack_strength', 5.0)
    
    apply_attack_patch(trainer, attack_type, byz_ids, strength)
    
    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    
    # Get results
    results = trainer.evaluation_result
    metric = 'accuracy'
    
    best_acc = results.get(f'best_test_{metric}', 0) * 100
    
    # Get last round accuracy
    last_acc = 0
    tot = 0
    for cid in range(args.num_clients):
        res = trainer.clients[cid].task.evaluate()
        last_acc += res[f'{metric}_test'] * trainer.clients[cid].task.num_samples
        tot += trainer.clients[cid].task.num_samples
    last_acc = (last_acc / tot) * 100
    
    print(f"Last={last_acc:.1f}% Best={best_acc:.1f}% ({elapsed:.0f}s)")
    
    return {'last': last_acc, 'best': best_acc, 'time': elapsed, 'seed': seed}


def run_experiment_with_seeds(config, seeds):
    """Run with multiple seeds"""
    results = []
    for seed in seeds:
        try:
            results.append(run_single_experiment(config, seed))
        except Exception as e:
            print(f"  ERROR: {e}")
    
    if not results:
        return None
    
    lasts = [r['last'] for r in results]
    bests = [r['best'] for r in results]
    times = [r['time'] for r in results]  # Track times
    
    return {
        'name': config['name'],
        'config': config,
        'mean_last': np.mean(lasts),
        'std_last': np.std(lasts),
        'mean_best': np.mean(bests),
        'std_best': np.std(bests),
        'mean_time': np.mean(times),  # Add mean time
        'std_time': np.std(times),    # Add std time
    }


def get_configs(dataset):
    """Generate configs"""
    base = {
        'dataset': dataset,
        'scenario': 'graph_fl',
        'task': 'graph_cls',
        'model': 'gcn',
        'simulation_mode': 'graph_fl_label_skew',
        'dirichlet_alpha': 0.5,
        'num_clients': 10,
        'rounds': 20,  # ← MEDIUM VERSION: 20 rounds
        'local_epochs': 3,
        'batch_size': 16,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'optimizer': 'adam',
        'metrics': 'accuracy',
        'evaluation_mode': 'local_model_on_local_data',
    }
    
    attacks = [
        ('no', [], 'none', 0),
        ('f2', [0, 1], 'scaled_sign_flip', 5.0),
        ('f3', [0, 1, 2], 'scaled_sign_flip', 5.0),
    ]
    
    methods = [
        ('FedAvg', 'fedavg', False, 0),
        ('FedProx', 'fedprox', False, 0),
        ('FedALA', 'fedala', False, 0),
        ('ARC-f2', 'fedalarc', True, 2),
        ('ARC-f3', 'fedalarc', True, 3),
    ]
    
    configs = []
    for att_name, byz, att_type, strength in attacks:
        for meth_name, alg, arc, max_byz in methods:
            configs.append({
                **base,
                'name': f"{dataset}_{att_name}_{meth_name}",
                'algorithm': alg,
                'use_arc': arc,
                'max_byzantine': max_byz,
                'byzantine_ids': byz,
                'external_attack_type': att_type,
                'attack_strength': strength,
            })
    
    return configs


def format_results(results):
    """Format as table with timing information"""
    data = []
    for r in results:
        if not r:
            continue
        cfg = r['config']
        att = cfg.get('external_attack_type', 'none')
        byz = len(cfg.get('byzantine_ids', []))
        
        att_label = 'No Attack' if att == 'none' else f"f={byz}"
        meth = cfg['name'].split('_')[-1]
        
        data.append({
            'Attack': att_label,
            'Method': meth,
            'Last': f"{r['mean_last']:.1f}±{r['std_last']:.1f}",
            'Time': f"{r['mean_time']:.0f}s",
            'Mean': r['mean_last'],
        })
    
    if not data:
        print("No successful experiments!")
        return None, None
    
    df = pd.DataFrame(data)
    
    # Pivot for accuracy
    pivot_acc = df.pivot_table(
        index='Method',
        columns='Attack',
        values='Last',
        aggfunc='first'
    )
    
    # Pivot for time
    pivot_time = df.pivot_table(
        index='Method',
        columns='Attack',
        values='Time',
        aggfunc='first'
    )
    
    # Reorder columns
    for pivot in [pivot_acc, pivot_time]:
        cols = list(pivot.columns)
        if 'No Attack' in cols:
            cols.remove('No Attack')
            cols = ['No Attack'] + sorted(cols)
            # Use loc to reorder to avoid SettingWithCopyWarning
            pivot = pivot[cols]
    
    return pivot_acc, pivot_time


def main():
    parser = argparse.ArgumentParser(description='Quick 10-round experiments')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MEDIUM EXPERIMENTS (20 rounds)")
    print("="*70)
    print(f"Dataset: {args.dataset} | Seeds: {args.seeds}")
    print("="*70 + "\n")
    
    configs = get_configs(args.dataset)
    print(f"Running {len(configs)} experiments × {len(args.seeds)} seeds = {len(configs) * len(args.seeds)} total\n")
    
    results = []
    for cfg in configs:
        results.append(run_experiment_with_seeds(cfg, args.seeds))
    
    # Show table
    table_acc, table_time = format_results(results)
    if table_acc is not None:
        print("\n" + "="*70)
        print("RESULTS (Last Round Accuracy)")
        print("="*70)
        print(table_acc.to_string())
        print("="*70)
        
        print("\n" + "="*70)
        print("TIMING (seconds per experiment)")
        print("="*70)
        print(table_time.to_string())
        print("="*70)
    
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('results', exist_ok=True)
            
            # Save JSON
            json_file = f'results/medium_{args.dataset}_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump([{
                    'name': r['name'],
                    'mean_last': r['mean_last'],
                    'std_last': r['std_last'],
                    'mean_time': r['mean_time'],
                    'std_time': r['std_time'],
                } for r in results if r], f, indent=2)
            
            # Save CSV
            csv_file = f'results/medium_{args.dataset}_{timestamp}.csv'
            table_acc.to_csv(csv_file)
            
            csv_time_file = f'results/medium_time_{args.dataset}_{timestamp}.csv'
            table_time.to_csv(csv_time_file)
            
            print(f"\nSaved: {json_file}")
            print(f"Saved: {csv_file}")
            print(f"Saved: {csv_time_file}")


if __name__ == "__main__":
    main()
