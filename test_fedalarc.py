"""
Quick test to verify FedALARC implementation
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from run_experiments import run_experiments

# Create experiment config as a DICTIONARY (not Namespace)
test_config = {
    'name': 'FedALARC_QuickTest',
    'dataset': ['PROTEINS'],
    'task': 'graph_cls',
    'scenario': 'graph_fl',
    'algorithm': 'fedalarc',
    'model': ['GIN'],
    
    # Simulation
    'simulation_mode': 'graph_fl_label_skew',
    'num_clients': 5,
    'dirichlet_alpha': 0.5,
    
    # Training (reduced for quick test)
    'num_rounds': 20,
    'num_epochs': 1,
    'batch_size': 128,
    'lr': 0.01,
    'weight_decay': 0.0005,
    'dropout': 0.5,
    'optim': 'adam',
    
    # Metrics
    'metrics': ['accuracy'],
    'evaluation_mode': 'local_model_on_local_data',
    
    # ALA parameters
    'eta': 1.0,
    'layer_idx': 1,
    'rand_percent': 80,
    'threshold': 0.1,
    'num_pre_loss': 10,
    
    # FedALARC Byzantine parameters
    'use_arc': True,
    'max_byzantine': 1,
    'byzantine_ids': [0],
    'attack_type': 'sign_flip',
    'attack_params': '{}',
}

print("="*70)
print("Testing FedALARC with Byzantine client...")
print("="*70)
print(f"Settings:")
print(f"  - Algorithm: {test_config['algorithm']}")
print(f"  - ARC enabled: {test_config['use_arc']}")
print(f"  - Byzantine clients: {test_config['byzantine_ids']}")
print(f"  - Attack type: {test_config['attack_type']}")
print(f"  - Max Byzantine (f): {test_config['max_byzantine']}")
print(f"  - Num rounds: {test_config['num_rounds']} (quick test)")
print("="*70)

# Run with single seed for quick test
result = run_experiments(test_config, seeds=[42])

print("\n" + "="*70)
print("TEST RESULT")
print("="*70)
if result['mean_metric'] > 0:
    print("✅ FedALARC ran successfully!")
    print(f"   Test Accuracy: {result['mean_metric']:.2f}%")
    print(f"   Time: {result['mean_time']:.2f}s")
else:
    print("❌ Test failed - check errors above")
print("="*70)