"""
COMPLETE EXPERIMENT RUNNER - FedALA vs FedALARC Byzantine Robustness Evaluation

This version includes all fixes:
1. ✅ Correct attack timing (after client.send_message, before server.execute)
2. ✅ Reports LAST round accuracy (where attack effects show)
3. ✅ Properly saves and retrieves last round accuracy

Tests:
- FedAvg (baseline, no defense)
- FedALA (adaptive local aggregation)
- FedALARC (FedALA + Adaptive Robust Clipping)

Under different Byzantine attack scenarios:
- No attack (baseline)
- f=2 attackers (20% Byzantine)
- f=3 attackers (30% Byzantine)
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

# Global counter for debugging
_attack_call_count = 0
_attack_applied_count = 0


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_correct_attack_patch(trainer, args, attack_type, byzantine_ids, attack_strength=1.0):
    """
    Patches the TRAINING LOOP to inject attacks AFTER client.send_message().
    
    This is the correct timing: attack happens after clients upload their weights
    but before server aggregates them.
    
    Args:
        trainer: The FGLTrainer object
        args: Arguments
        attack_type: Type of attack ('scaled_sign_flip', 'gaussian_noise', etc.)
        byzantine_ids: List of Byzantine client IDs
        attack_strength: Multiplier for attack intensity
    """
    global _attack_call_count, _attack_applied_count
    _attack_call_count = 0
    _attack_applied_count = 0
    
    if not byzantine_ids or attack_type == 'none':
        return

    # Store original train method
    original_train = trainer.train

    def attacked_train():
        """Modified training loop that injects attacks at the RIGHT time"""
        for round_id in range(trainer.args.num_rounds):
            sampled_clients = sorted(random.sample(
                list(range(trainer.args.num_clients)), 
                int(trainer.args.num_clients * trainer.args.client_frac)
            ))
            print(f"round # {round_id}\t\tsampled_clients: {sampled_clients}")
            trainer.message_pool["round"] = round_id
            trainer.message_pool["sampled_clients"] = sampled_clients
            
            # Step 1: Server sends global model to clients
            trainer.server.send_message()
            
            # Step 2 & 3: Clients train and upload their weights
            for client_id in sampled_clients:
                trainer.clients[client_id].execute()
                trainer.clients[client_id].send_message()
            
            # ✅ Step 3.5: ATTACK! Modify uploaded weights in message_pool
            global _attack_call_count, _attack_applied_count
            _attack_call_count += 1
            
            active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]
            
            if active_attackers:
                _attack_applied_count += 1
                
                for client_id in active_attackers:
                    client_msg = trainer.message_pool[f"client_{client_id}"]
                    weights = client_msg["weight"]
                    attacked_weights = []
                    
                    # Apply attack based on type
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
                    
                    elif attack_type == 'gaussian_noise':
                        for w_local in weights:
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            noise_scale = attack_strength * data_local.std().item() * 10
                            noise = torch.randn_like(data_local) * max(noise_scale, 0.1)
                            w_attacked = data_local + noise
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    elif attack_type == 'sign_flip':
                        global_weights = list(trainer.server.task.model.parameters())
                        for w_local, w_global in zip(weights, global_weights):
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            data_global = w_global.data if hasattr(w_global, 'data') else w_global
                            w_attacked = 2 * data_global - data_local
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    else:
                        attacked_weights = weights
                    
                    # Actually replace the weights in the pool
                    trainer.message_pool[f"client_{client_id}"]["weight"] = attacked_weights
            
            # Step 4: Server aggregates (using attacked weights from pool)
            trainer.server.execute()
            
            # Step 5: Evaluate
            trainer.evaluate()
            print("-"*50)
        
        # ✅ SAVE LAST ROUND ACCURACY
        # Manually compute last round accuracy
        metric_name = 'accuracy' if trainer.args.task in ['graph_cls', 'node_cls'] else 'mse'
        
        last_test_acc = 0
        tot_samples = 0
        for client_id in range(trainer.args.num_clients):
            result = trainer.clients[client_id].task.evaluate()
            num_samples = trainer.clients[client_id].task.num_samples
            last_test_acc += result[f'{metric_name}_test'] * num_samples
            tot_samples += num_samples
        
        last_test_acc = last_test_acc / tot_samples
        
        # Store in evaluation_result for later access
        trainer.evaluation_result[f'last_test_{metric_name}'] = last_test_acc
        
        trainer.logger.save()

    # Replace the train method
    trainer.train = attacked_train


def print_attack_summary():
    """Print attack summary"""
    global _attack_call_count, _attack_applied_count
    if _attack_call_count > 0:
        print(f"\n[ATTACK SUMMARY] Rounds: {_attack_call_count}, Attacked: {_attack_applied_count}")


def run_single_experiment(config, seed):
    """Run a single experiment and return last round accuracy"""
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = seed
    
    # Apply configuration
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
    args.num_epochs = config.get('local_epochs', 3)
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    args.dropout = config['dropout']
    args.optim = config.get('optimizer', 'adam')
    metrics = config.get('metrics', ['accuracy'])
    args.metrics = [metrics] if isinstance(metrics, str) else metrics
    args.evaluation_mode = config.get('evaluation_mode', 'local_model_on_local_data')
    
    args.dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
    args.skew_alpha = config.get('skew_alpha', config.get('dirichlet_alpha', 0.5))
    args.louvain_resolution = config.get('louvain_resolution', 1.0)
    args.louvain_delta = config.get('louvain_delta', 20)
    args.metis_num_coms = config.get('metis_num_coms', 100)
    
    # FedALA parameters
    args.eta = config.get('eta', 1.0)
    args.layer_idx = config.get('layer_idx', 1)
    args.rand_percent = config.get('rand_percent', 80)
    args.threshold = config.get('threshold', 0.1)
    args.num_pre_loss = config.get('num_pre_loss', 10)
    
    # FedALARC / Byzantine parameters
    args.use_arc = config.get('use_arc', False)
    args.max_byzantine = config.get('max_byzantine', 0)
    args.byzantine_ids = config.get('byzantine_ids', [])
    args.attack_type = config.get('attack_type', 'none')
    args.attack_params = json.dumps(config.get('attack_params', {}))
    
    # Logging
    args.debug = True
    args.log_root = config.get('log_root', './logs_fedalarc')
    args.log_name = f"{config['name']}_seed{seed}"
    
    print(f"\n{'#'*70}")
    print(f"# {config['name']} | Seed: {seed}")
    print(f"# Algorithm: {args.fl_algorithm} | Attack: {config.get('external_attack_type', 'none')}")
    print(f"# ARC: {args.use_arc} | max_byzantine: {args.max_byzantine}")
    print(f"{'#'*70}")
    
    start_time = time.time()
    trainer = FGLTrainer(args)
    
    # Apply attack with CORRECT timing
    real_attack_type = config.get('external_attack_type', 'none')
    byz_ids = config.get('byzantine_ids', [])
    attack_strength = config.get('attack_strength', 5.0)
    
    if real_attack_type != 'none':
        print(f"[ATTACK] Installing {real_attack_type} on clients {byz_ids} (strength={attack_strength})")
    
    apply_correct_attack_patch(trainer, args, real_attack_type, byz_ids, attack_strength)

    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    trainer.train()
    running_time = time.time() - start_time
    
    print_attack_summary()
    
    # Extract results
    results = trainer.evaluation_result
    metric_name = 'accuracy' if config['task'] in ['graph_cls', 'node_cls'] else 'mse'
    
    best_test_acc = results.get(f'best_test_{metric_name}', 0)
    last_test_acc = results.get(f'last_test_{metric_name}', best_test_acc)  # Use saved last round accuracy
    
    if metric_name == 'accuracy':
        if best_test_acc < 1:
            best_test_acc *= 100
        if last_test_acc < 1:
            last_test_acc *= 100
    
    print(f"\n[RESULT] Last Round: {last_test_acc:.2f}% | Best Round: {best_test_acc:.2f}%")
    
    return {
        'last_acc': last_test_acc,
        'best_acc': best_test_acc,
        'time': running_time,
        'seed': seed
    }


def run_experiment_with_seeds(config, seeds):
    """Run experiment with multiple seeds and aggregate results"""
    all_results = []
    
    for seed in seeds:
        try:
            result = run_single_experiment(config, seed)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        return None
    
    # Aggregate results - focus on LAST round accuracy (where attack effects show)
    last_accs = [r['last_acc'] for r in all_results]
    best_accs = [r['best_acc'] for r in all_results]
    times = [r['time'] for r in all_results]
    
    return {
        'name': config['name'],
        'config': config,
        'mean_last_acc': np.mean(last_accs),
        'std_last_acc': np.std(last_accs),
        'mean_best_acc': np.mean(best_accs),
        'std_best_acc': np.std(best_accs),
        'time_mean': np.mean(times),
        'seeds': seeds,
        'all_last_accs': last_accs,
        'all_best_accs': best_accs
    }


def get_byzantine_configs(dataset):
    """Generate experiment configurations for Byzantine robustness evaluation"""
    configs = []
    
    base = {
        'dataset': dataset,
        'scenario': 'graph_fl',
        'task': 'graph_cls',
        'model': 'gcn',
        'simulation_mode': 'graph_fl_label_skew',
        'dirichlet_alpha': 0.5,
        'skew_alpha': 0.5,
        'num_clients': 10,
        'rounds': 30,
        'local_epochs': 3,
        'batch_size': 16,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'optimizer': 'adam',
        'metrics': ['accuracy'],
        'evaluation_mode': 'local_model_on_local_data',
        # FedALA parameters
        'eta': 1.0,
        'layer_idx': 1,
        'rand_percent': 80,
        'threshold': 0.1,
        'num_pre_loss': 10,
    }
    
    # Attack scenarios
    attacks = [
        ('no_attack', [], 'none', 0.0),
        ('f2_attack', [0, 1], 'scaled_sign_flip', 5.0),
        ('f3_attack', [0, 1, 2], 'scaled_sign_flip', 5.0),
    ]
    
    # Methods to compare
    methods = [
        ('FedAvg', 'fedavg', False, 0),
        ('FedALA', 'fedala', False, 0),
        ('FedALARC-f2', 'fedalarc', True, 2),
        ('FedALARC-f3', 'fedalarc', True, 3),
    ]
    
    for attack_name, byz_ids, attack_type, strength in attacks:
        for method_name, alg, use_arc, max_byz in methods:
            config = {
                **base,
                'name': f"{dataset}_{attack_name}_{method_name}",
                'algorithm': alg,
                'use_arc': use_arc,
                'max_byzantine': max_byz,
                'byzantine_ids': byz_ids,
                'external_attack_type': attack_type,
                'attack_strength': strength,
                'attack_type': 'none',  # Disable internal attack mechanism
            }
            configs.append(config)
            
    return configs


def format_byzantine_results(results):
    """Format results as a pivot table showing last round accuracy"""
    df_data = []
    
    for result in results:
        if result is None:
            continue
            
        cfg = result['config']
        
        # Determine attack label
        real_attack = cfg.get('external_attack_type', 'none')
        num_attackers = len(cfg.get('byzantine_ids', []))
        
        if real_attack == 'none' or num_attackers == 0:
            attack_label = 'No Attack'
        else:
            attack_label = f"f={num_attackers}"
        
        # Extract method name
        method_name = cfg['name'].split('_')[-1]
        
        df_data.append({
            'Dataset': cfg['dataset'],
            'Attack': attack_label,
            'Method': method_name,
            'LastAcc': f"{result['mean_last_acc']:.2f}±{result['std_last_acc']:.2f}",
            'BestAcc': f"{result['mean_best_acc']:.2f}±{result['std_best_acc']:.2f}",
            'Mean': result['mean_last_acc'],
        })
    
    df = pd.DataFrame(df_data)
    
    # Pivot using Last Round Accuracy
    pivot = df.pivot_table(
        index=['Dataset', 'Method'],
        columns='Attack',
        values='LastAcc',
        aggfunc='first'
    )
    
    # Reorder columns
    cols = list(pivot.columns)
    if 'No Attack' in cols:
        cols.remove('No Attack')
        cols = ['No Attack'] + sorted(cols)
        pivot = pivot[cols]
    
    return pivot


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='FedALA-ARC Byzantine Robustness Experiments')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                       help='Dataset to use (default: PROTEINS)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                       help='Random seeds to use (default: 0 1 2)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print experiment list without running')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FEDALARC BYZANTINE ROBUSTNESS EXPERIMENTS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Attack type: Scaled sign-flip (strength=5.0)")
    print(f"Evaluation metric: LAST ROUND accuracy")
    print("="*70)
    
    # Generate experiment configurations
    configs = get_byzantine_configs(args.dataset)
    
    if args.dry_run:
        print(f"\nWould run {len(configs)} experiments:")
        for cfg in configs:
            attack = cfg.get('external_attack_type', 'none')
            byz = cfg.get('byzantine_ids', [])
            print(f"  - {cfg['name']}: {cfg['algorithm']} | attack={attack} | byzantine={byz}")
        return
    
    # Run all experiments
    print(f"\nRunning {len(configs)} experiments with {len(args.seeds)} seeds each...")
    print(f"Total runs: {len(configs) * len(args.seeds)}")
    print()
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"Experiment {i}/{len(configs)}: {config['name']}")
        print(f"{'='*70}")
        
        result = run_experiment_with_seeds(config, args.seeds)
        if result:
            results.append(result)
    
    # Format and display results
    table = format_byzantine_results(results)
    print("\n" + "="*70)
    print("FINAL RESULTS (Last Round Accuracy)")
    print("="*70)
    print(table.to_string())
    print("="*70)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    output_file = f'results/fedalarc_byzantine_{args.dataset}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump([{
            'name': r['name'],
            'mean_last_acc': r['mean_last_acc'],
            'std_last_acc': r['std_last_acc'],
            'mean_best_acc': r['mean_best_acc'],
            'std_best_acc': r['std_best_acc'],
            'config': {k: v for k, v in r['config'].items() 
                      if k != 'byzantine_ids' or not isinstance(v, (list, dict))}
        } for r in results], f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Save table as CSV
    csv_file = f'results/fedalarc_byzantine_{args.dataset}_{timestamp}.csv'
    table.to_csv(csv_file)
    print(f"Table saved to: {csv_file}")


if __name__ == "__main__":
    main()
