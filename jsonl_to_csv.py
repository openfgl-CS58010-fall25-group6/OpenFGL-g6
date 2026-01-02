import json
import csv

def jsonl_to_csv_rows(jsonl_file, output_csv=None):
    """Convert JSONL results to CSV format matching your spreadsheet."""
    
    rows = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Format accuracy as "mean±std"
            accuracy_formatted = f"{data['mean_metric']:.2f}±{data['std_metric']:.2f}"
            
            row = {
                'Dataset': data['dataset'],
                'Scenario': 'Graph-FL',
                'Algorithm': f"GIN-FedALA (layer_idx={data.get('layer_idx', 'N/A')})",
                'Accuracy Reported in the paper': '',  # Fill manually
                'Accuracy': round(data['mean_metric'], 2),
                'std dev (acc)': round(data['std_metric'], 2),
                'Communication Cost (MB)': round(data['mean_comm'], 2),
                'Time (seconds)': round(data['mean_time'], 2),
                'task': 'graph_cls',
                'model': data['model'],
                'simulation_mode': data['simulation_mode'],
                'num_clients': data['num_clients'],
                'dirichlet_alpha': data['dirichlet_alpha'],
                'num_rounds': data['num_rounds'],
                'local_epochs': data['local_epochs'],
                'batch_size': data['batch_size'],
                'lr': data['lr'],
                'weight_decay': data['weight_decay'],
                'dropout': data['dropout'],
                'optimizer': 'adam',
                # Extra useful columns
                'layer_idx': data.get('layer_idx', 'N/A'),
                'val_accuracy': round(data['mean_val_metric'], 2),
                'best_round': round(data['mean_best_round'], 1),
            }
            rows.append(row)
    
    # Print as tab-separated for easy copy-paste
    print("\n" + "="*80)
    print("TAB-SEPARATED (copy-paste to Google Sheets):")
    print("="*80)
    
    # Header
    headers = list(rows[0].keys())
    print('\t'.join(headers))
    
    # Data rows
    for row in rows:
        print('\t'.join(str(row[h]) for h in headers))
    
    # Optionally save to CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {output_csv}")
    
    return rows


# Process both files
print("\n" + "#"*80)
print("LAYER_IDX = 1 RESULTS")
print("#"*80)
jsonl_to_csv_rows('results/graph_fl/table6_graphfl_20260102_020732.jsonl', 
                  'results/fedala_layer_idx_1.csv')

print("\n" + "#"*80)
print("LAYER_IDX = 0 RESULTS")
print("#"*80)
jsonl_to_csv_rows('results/graph_fl/table6_graphfl_20260102_020819.jsonl',
                  'results/fedala_layer_idx_0.csv')