"""
Minimal test: Track global model weights to see if attack has ANY effect
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
import copy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment_and_track_weights(attack_type, byzantine_ids, attack_strength):
    """Run experiment and track global model weights"""
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(42)  # Different seed to see if that's the issue
    
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = 42
    
    args.dataset = ['PROTEINS']
    args.task = 'graph_cls'
    args.scenario = 'graph_fl'
    args.fl_algorithm = 'fedavg'
    args.model = ['gcn']
    args.simulation_mode = 'graph_fl_label_skew'
    args.num_clients = 10
    args.num_rounds = 5  # Just 5 rounds for quick test
    args.num_epochs = 3
    args.batch_size = 16
    args.lr = 0.01
    args.weight_decay = 5e-4
    args.dropout = 0.5
    args.metrics = ['accuracy']
    args.evaluation_mode = 'local_model_on_local_data'
    args.dirichlet_alpha = 0.5
    args.debug = True
    args.log_root = './logs_weight_track'
    args.log_name = f'track_{attack_type}'
    
    print(f"\n{'='*80}")
    print(f"Tracking weights: attack={attack_type}, byzantine={byzantine_ids}, strength={attack_strength}")
    print(f"{'='*80}\n")
    
    trainer = FGLTrainer(args)
    
    # Store original train
    original_train = trainer.train
    
    # Track global weights
    weight_history = []
    
    def tracked_train():
        """Training with weight tracking"""
        for round_id in range(trainer.args.num_rounds):
            sampled_clients = sorted(random.sample(
                list(range(trainer.args.num_clients)), 
                int(trainer.args.num_clients * trainer.args.client_frac)
            ))
            print(f"Round {round_id}")
            trainer.message_pool["round"] = round_id
            trainer.message_pool["sampled_clients"] = sampled_clients
            
            # Track BEFORE round
            w0_before = list(trainer.server.task.model.parameters())[0].data.clone()
            
            # Standard training
            trainer.server.send_message()
            
            for client_id in sampled_clients:
                trainer.clients[client_id].execute()
                trainer.clients[client_id].send_message()
            
            # ✅ APPLY ATTACK HERE
            active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]
            
            if active_attackers and attack_type != 'none':
                print(f"  Attacking clients: {active_attackers}")
                for client_id in active_attackers:
                    client_msg = trainer.message_pool[f"client_{client_id}"]
                    weights = client_msg["weight"]
                    attacked_weights = []
                    
                    global_weights = list(trainer.server.task.model.parameters())
                    for w_local, w_global in zip(weights, global_weights):
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        
                        if attack_type == 'scaled_sign_flip':
                            gradient = data_local - data_global
                            w_attacked = data_global - attack_strength * gradient
                        elif attack_type == 'random':
                            w_attacked = torch.randn_like(data_local) * attack_strength
                        else:
                            w_attacked = data_local
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                    
                    trainer.message_pool[f"client_{client_id}"]["weight"] = attacked_weights
            
            # Aggregate
            trainer.server.execute()
            
            # Track AFTER round
            w0_after = list(trainer.server.task.model.parameters())[0].data.clone()
            
            # Compute statistics
            norm_before = w0_before.norm().item()
            norm_after = w0_after.norm().item()
            change = (w0_after - w0_before).norm().item()
            mean_before = w0_before.mean().item()
            mean_after = w0_after.mean().item()
            
            print(f"  Before: norm={norm_before:.6f}, mean={mean_before:.6f}")
            print(f"  After:  norm={norm_after:.6f}, mean={mean_after:.6f}")
            print(f"  Change: {change:.6f}\n")
            
            weight_history.append({
                'round': round_id,
                'norm_before': norm_before,
                'norm_after': norm_after,
                'change': change,
                'mean_before': mean_before,
                'mean_after': mean_after,
            })
            
            # Evaluate
            trainer.evaluate()
            print("-"*50)
        
        trainer.logger.save()
    
    # Replace train
    trainer.train = tracked_train
    
    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    # Run
    trainer.train()
    
    # Get final accuracy
    results = trainer.evaluation_result
    best_acc = results.get('best_test_accuracy', 0) * 100
    
    return best_acc, weight_history


def main():
    """Compare weight evolution with and without attack"""
    
    print("="*80)
    print("WEIGHT EVOLUTION TRACKING TEST")
    print("="*80)
    
    # Run baseline
    print("\n[1/3] BASELINE (no attack)")
    acc_baseline, hist_baseline = run_experiment_and_track_weights('none', [], 0)
    
    # Run with moderate attack
    print("\n[2/3] MODERATE ATTACK (f=2, strength=5)")
    acc_attack1, hist_attack1 = run_experiment_and_track_weights('scaled_sign_flip', [0, 1], 5.0)
    
    # Run with strong attack
    print("\n[3/3] STRONG ATTACK (f=2, strength=50)")
    acc_attack2, hist_attack2 = run_experiment_and_track_weights('scaled_sign_flip', [0, 1], 50.0)
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Baseline accuracy:        {acc_baseline:.2f}%")
    print(f"Moderate attack accuracy: {acc_attack1:.2f}% (drop: {acc_baseline - acc_attack1:+.2f}%)")
    print(f"Strong attack accuracy:   {acc_attack2:.2f}% (drop: {acc_baseline - acc_attack2:+.2f}%)")
    
    print("\n" + "="*80)
    print("WEIGHT CHANGE PER ROUND")
    print("="*80)
    print(f"{'Round':<8} {'Baseline':<12} {'Moderate':<12} {'Strong':<12}")
    print("-"*80)
    for i in range(len(hist_baseline)):
        print(f"{i:<8} {hist_baseline[i]['change']:<12.6f} {hist_attack1[i]['change']:<12.6f} {hist_attack2[i]['change']:<12.6f}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    avg_change_baseline = np.mean([h['change'] for h in hist_baseline])
    avg_change_attack1 = np.mean([h['change'] for h in hist_attack1])
    avg_change_attack2 = np.mean([h['change'] for h in hist_attack2])
    
    print(f"Average weight change per round:")
    print(f"  Baseline:  {avg_change_baseline:.6f}")
    print(f"  Moderate:  {avg_change_attack1:.6f} ({avg_change_attack1/avg_change_baseline:.2f}x)")
    print(f"  Strong:    {avg_change_attack2:.6f} ({avg_change_attack2/avg_change_baseline:.2f}x)")
    
    if avg_change_attack1 > avg_change_baseline * 1.2:
        print("\n✅ Attacks ARE affecting weight evolution (>20% increase in change)")
    else:
        print("\n⚠️  Attacks show minimal effect on weight evolution (<20% change)")
    
    if abs(acc_baseline - acc_attack2) < 1.0:
        print("❌ But accuracy is NOT affected - something else is wrong!")
        print("\nPossible issues:")
        print("  1. Evaluation is cached or using wrong model")
        print("  2. Model updates aren't being applied to evaluation")  
        print("  3. Attack is being neutralized by some defense mechanism")
        print("  4. Dataset is too robust to weight perturbations")
    else:
        print("✅ Accuracy IS affected - attacks are working!")


if __name__ == "__main__":
    main()
