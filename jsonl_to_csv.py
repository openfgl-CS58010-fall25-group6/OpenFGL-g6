import json
import csv
import argparse
import os


def jsonl_to_csv_rows(jsonl_file, output_csv=None):
    """Convert JSONL results to CSV format matching your spreadsheet."""
    
    rows = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Build algorithm name with relevant parameters
            algo_parts = [f"GIN-{data['algorithm'].upper()}"]
            if data.get('layer_idx') is not None:
                algo_parts.append(f"layer_idx={data['layer_idx']}")
            if data.get('lambda_graph') is not None and data.get('lambda_graph') > 0:
                algo_parts.append(f"λ={data['lambda_graph']}")
                algo_parts.append(f"{data.get('graph_reg_type', 'laplacian')}")
            
            algorithm_str = f"{algo_parts[0]} ({', '.join(algo_parts[1:])})" if len(algo_parts) > 1 else algo_parts[0]
            
            row = {
                'Dataset': data['dataset'],
                'Scenario': 'Graph-FL',
                'Algorithm': algorithm_str,
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
                # FedALA parameters
                'layer_idx': data.get('layer_idx', ''),
                # Extra useful columns
                'val_accuracy': round(data['mean_val_metric'], 2),
                'best_round': round(data['mean_best_round'], 1),
                # Regularization parameters
                'lambda_graph': data.get('lambda_graph', 0.0),
                'graph_reg_type': data.get('graph_reg_type', ''),
                # FedALA+ parameters
                'rand_percent': data.get('rand_percent', ''),
                'use_disagreement': data.get('use_disagreement', ''),
                'selection_frequency': data.get('selection_frequency', ''),
                'min_disagreement_samples': data.get('min_disagreement_samples', ''),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL experiment results to CSV")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of JSONL files to process"
    )
    
    args = parser.parse_args()
    
    for f in args.files:
        print(f"\nProcessing: {f}")
        base, ext = os.path.splitext(f)
        output_csv = base + ".csv" if ext == ".jsonl" else None
        jsonl_to_csv_rows(f, output_csv=output_csv)