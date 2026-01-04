"""
Quick test: Use MUCH stronger attacks that should definitely cause damage
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_fedalarc_experiments_TRULY_FIXED import run_experiment_with_seeds
import numpy as np

def main():
    """Test with progressively stronger attacks"""
    
    base_config = {
        'dataset': 'PROTEINS',
        'task': 'graph_cls',
        'scenario': 'graph_fl',
        'model': 'gcn',
        'simulation_mode': 'graph_fl_label_skew',
        'num_clients': 10,
        'rounds': 20,
        'local_epochs': 3,
        'batch_size': 16,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'metrics': 'accuracy',
        'evaluation_mode': 'local_model_on_local_data',
        'dirichlet_alpha': 0.5,
    }
    
    experiments = [
        # Baseline
        {
            **base_config,
            'name': 'baseline',
            'algorithm': 'fedavg',
            'external_attack_type': 'none',
            'byzantine_ids': [],
        },
        
        # Moderate attack (f=2, strength=5)
        {
            **base_config,
            'name': 'f2_str5',
            'algorithm': 'fedavg',
            'external_attack_type': 'scaled_sign_flip',
            'byzantine_ids': [0, 1],
            'attack_strength': 5.0,
        },
        
        # Stronger attack (f=2, strength=50)
        {
            **base_config,
            'name': 'f2_str50',
            'algorithm': 'fedavg',
            'external_attack_type': 'scaled_sign_flip',
            'byzantine_ids': [0, 1],
            'attack_strength': 50.0,
        },
        
        # More attackers (f=4, strength=10)
        {
            **base_config,
            'name': 'f4_str10',
            'algorithm': 'fedavg',
            'external_attack_type': 'scaled_sign_flip',
            'byzantine_ids': [0, 1, 2, 3],
            'attack_strength': 10.0,
        },
        
        # Very strong + more attackers (f=4, strength=50)
        {
            **base_config,
            'name': 'f4_str50',
            'algorithm': 'fedavg',
            'external_attack_type': 'scaled_sign_flip',
            'byzantine_ids': [0, 1, 2, 3],
            'attack_strength': 50.0,
        },
    ]
    
    print("="*80)
    print("QUICK ATTACK STRENGTH TEST")
    print("="*80)
    
    seeds = [0]
    results = {}
    
    for config in experiments:
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"{'='*80}")
        result = run_experiment_with_seeds(config, seeds)
        if result:
            results[config['name']] = result['last_acc_mean']
            print(f"✓ {config['name']}: {result['last_acc_mean']:.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (LAST ROUND - where attack effects are visible)")
    print("="*80)
    baseline = results.get('baseline', 0)
    print(f"{'Config':<20} {'Last Round Acc':<15} {'Drop from baseline'}")
    print("-"*80)
    for name, acc in results.items():
        drop = baseline - acc
        symbol = "⚠️" if abs(drop) > 5 else "✓" if abs(drop) < 1 else " "
        print(f"{symbol} {name:<18} {acc:>6.2f}%         {drop:>+6.2f}%")
    
    # Analysis
    print("\n" + "="*80)
    max_drop = max([baseline - acc for name, acc in results.items() if name != 'baseline'], default=0)
    
    if max_drop > 5:
        print(f"✅ SUCCESS: Found effective attack (max drop: {max_drop:.2f}%)")
        print("   → Now we can test if FedALA/ARC can defend against it!")
    elif max_drop > 1:
        print(f"⚠️  WEAK EFFECT: Attacks only drop accuracy by {max_drop:.2f}%")
        print("   → May need even stronger attacks or different strategy")
    else:
        print(f"❌ NO EFFECT: Attacks have minimal impact ({max_drop:.2f}%)")
        print("   → Possible reasons:")
        print("      1. FedAvg aggregation is naturally robust")
        print("      2. PROTEINS dataset is too easy to damage")
        print("      3. Attack implementation may still have issues")


if __name__ == "__main__":
    main()
