"""
DEBUG VERSION v4 - Verify attack is actually working

This version adds extensive logging to verify:
1. Attack patch is being called
2. Weights are actually being modified
3. Modified weights are being used in aggregation
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


def apply_universal_attack_patch(trainer, args, attack_type, byzantine_ids, attack_strength=5.0):
    """Injects attacks with VERIFICATION."""
    
    if not byzantine_ids or attack_type == 'none':
        print(f"[ATTACK] No attack configured")
        return

    print(f"[ATTACK] Installing {attack_type} on clients {byzantine_ids} (strength={attack_strength})")
    
    original_server_execute = trainer.server.execute
    round_num = [0]

    def attacked_execute():
        round_num[0] += 1
        pool = trainer.message_pool
        sampled_clients = pool.get("sampled_clients", [])
        active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]
        
        # Debug: Print first few rounds
        if round_num[0] <= 3:
            print(f"\n[ATTACK DEBUG Round {round_num[0]}]")
            print(f"  sampled_clients: {sampled_clients}")
            print(f"  byzantine_ids: {byzantine_ids}")
            print(f"  active_attackers: {active_attackers}")

        if active_attackers and round_num[0] > 1:
            global_weights = list(trainer.server.task.model.parameters())
            
            for client_id in active_attackers:
                client_msg = pool[f"client_{client_id}"]
                weights = client_msg["weight"]
                
                # Debug: Check weights before attack
                if round_num[0] <= 3:
                    w0_before = weights[0].data if hasattr(weights[0], 'data') else weights[0]
                    print(f"  Client {client_id} weight[0] BEFORE: norm={w0_before.norm().item():.6f}, mean={w0_before.mean().item():.6f}")
                
                attacked_weights = []
                
                for w_local, w_global in zip(weights, global_weights):
                    data_local = w_local.data if hasattr(w_local, 'data') else w_local
                    data_global = w_global.data if hasattr(w_global, 'data') else w_global
                    
                    if attack_type in ['sign_flip', 'scaled_sign_flip']:
                        gradient = data_local - data_global
                        w_attacked = data_global - attack_strength * gradient
                    elif attack_type == 'gaussian':
                        w_attacked = data_local + torch.randn_like(data_local) * attack_strength
                    else:
                        w_attacked = data_local
                    
                    if isinstance(w_local, torch.nn.Parameter):
                        w_attacked = torch.nn.Parameter(w_attacked.clone())
                    attacked_weights.append(w_attacked)
                
                # Debug: Check weights after attack
                if round_num[0] <= 3:
                    w0_after = attacked_weights[0].data if hasattr(attacked_weights[0], 'data') else attacked_weights[0]
                    print(f"  Client {client_id} weight[0] AFTER:  norm={w0_after.norm().item():.6f}, mean={w0_after.mean().item():.6f}")
                
                # CRITICAL: Update the pool
                pool[f"client_{client_id}"]["weight"] = attacked_weights
                
                # VERIFY the update stuck
                if round_num[0] <= 3:
                    verify = pool[f"client_{client_id}"]["weight"][0]
                    v_data = verify.data if hasattr(verify, 'data') else verify
                    print(f"  Client {client_id} VERIFY in pool: norm={v_data.norm().item():.6f}")
        
        # Call original
        original_server_execute()
        
        # Debug: Check global model after aggregation
        if round_num[0] <= 3 and active_attackers:
            global_after = list(trainer.server.task.model.parameters())[0]
            print(f"  Global weight[0] AFTER aggregation: norm={global_after.data.norm().item():.6f}")

    trainer.server.execute = attacked_execute


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(config, seed):
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = seed
    
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
    args.metrics = ['accuracy']
    args.evaluation_mode = 'local_model_on_local_data'
    
    args.dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
    args.skew_alpha = config.get('skew_alpha', 0.5)
    args.louvain_resolution = 1.0
    args.louvain_delta = 20
    args.metis_num_coms = 100
    
    args.eta = config.get('eta', 1.0)
    args.layer_idx = config.get('layer_idx', 1)
    args.rand_percent = config.get('rand_percent', 80)
    args.threshold = config.get('threshold', 0.1)
    args.num_pre_loss = config.get('num_pre_loss', 10)
    
    args.use_arc = config.get('use_arc', False)
    args.max_byzantine = config.get('max_byzantine', 0)
    args.byzantine_ids = []
    args.attack_type = 'none'
    args.attack_params = '{}'
    
    args.debug = True
    args.log_root = './logs_debug'
    args.log_name = f"{config['name']}_seed{seed}"
    
    print(f"\n{'#'*70}")
    print(f"# {config['name']} | Seed: {seed}")
    print(f"# Algorithm: {args.fl_algorithm} | Attack: {config.get('external_attack_type', 'none')}")
    print(f"{'#'*70}")
    
    trainer = FGLTrainer(args)
    
    apply_universal_attack_patch(
        trainer, args,
        config.get('external_attack_type', 'none'),
        config.get('byzantine_ids', []),
        config.get('attack_strength', 5.0)
    )
    
    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    trainer.train()
    
    results = trainer.evaluation_result
    last_acc = results.get('curr_test_accuracy', results.get('best_test_accuracy', 0))
    best_acc = results.get('best_test_accuracy', 0)
    
    if last_acc < 1:
        last_acc *= 100
    if best_acc < 1:
        best_acc *= 100
    
    print(f"\n[RESULT] Last: {last_acc:.2f}% | Best: {best_acc:.2f}%")
    
    return {'last_acc': last_acc, 'best_acc': best_acc, 'seed': seed}


def run_experiment_with_seeds(config, seeds):
    results = [run_single_experiment(config, s) for s in seeds]
    last_accs = [r['last_acc'] for r in results]
    return {
        'name': config['name'],
        'mean': np.mean(last_accs),
        'std': np.std(last_accs),
        'all': last_accs,
        'config': config
    }


def get_configs(dataset='PROTEINS'):
    base = {
        'dataset': dataset,
        'scenario': 'graph_fl',
        'task': 'graph_cls',
        'model': 'gin',
        'simulation_mode': 'graph_fl_label_skew',
        'dirichlet_alpha': 0.5,
        'skew_alpha': 0.5,
        'num_clients': 10,
        'rounds': 10,  # Fewer rounds for debugging
        'local_epochs': 3,
        'batch_size': 128,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'dropout': 0.5,
        'optimizer': 'adam',
        'eta': 1.0,
        'layer_idx': 1,
        'rand_percent': 80,
        'threshold': 0.1,
        'num_pre_loss': 10,
    }
    
    # Just test FedAvg with and without attack for debugging
    return [
        {**base, 'name': f'{dataset}_no_attack_FedAvg', 'algorithm': 'fedavg',
         'use_arc': False, 'max_byzantine': 0, 'byzantine_ids': [],
         'external_attack_type': 'none', 'attack_strength': 0},
        
        {**base, 'name': f'{dataset}_f2_attack_FedAvg', 'algorithm': 'fedavg',
         'use_arc': False, 'max_byzantine': 0, 'byzantine_ids': [0, 1],
         'external_attack_type': 'scaled_sign_flip', 'attack_strength': 5.0},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PROTEINS')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    
    print("="*70)
    print("DEBUG: Verifying attack is working")
    print("="*70)
    
    configs = get_configs(args.dataset)
    results = []
    
    for cfg in configs:
        r = run_experiment_with_seeds(cfg, args.seeds)
        results.append(r)
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    for r in results:
        print(f"{r['name']}: {r['mean']:.2f}%")


if __name__ == "__main__":
    main()
