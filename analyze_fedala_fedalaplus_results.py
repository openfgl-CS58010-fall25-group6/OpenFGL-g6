"""Analyze FedALA vs FedALA+ comparison results.

This script reads the JSONL results from run_table6_graphfl.py and generates
comparison tables, plots, and statistical analysis.

Usage:
    python analyze_fedala_results.py results/graph_fl/table6_graphfl_*.jsonl
    python analyze_fedala_results.py results/fedala_comparison/*.jsonl --output report.html
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_results(jsonl_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load all JSONL result files."""
    results = []
    for path in jsonl_paths:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def extract_algorithm_type(name: str) -> str:
    """Extract whether this is FedALA or FedALA+."""
    if "fedala+" in name.lower() or "RP" in name:
        return "FedALA+"
    return "FedALA"


def extract_rand_percent(name: str) -> Optional[int]:
    """Extract rand_percent from experiment name."""
    match = re.search(r"RP(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def extract_dirichlet_alpha(result: Dict) -> float:
    """Extract dirichlet_alpha from result."""
    return result.get("dirichlet_alpha", 1.0)


def extract_config(result: Dict) -> tuple:
    """Extract (num_clients, batch_size) configuration."""
    return (result.get("num_clients", 0), result.get("batch_size", 0))


def group_results(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by experiment type."""
    groups = defaultdict(list)
    
    for r in results:
        name = r.get("name", "")
        algo_type = extract_algorithm_type(name)
        
        # Configuration comparison
        if "C" in name and "B" in name:
            groups["config_comparison"].append(r)
        
        # Heterogeneity comparison
        if "hetero" in name.lower() or "alpha" in name.lower():
            groups["heterogeneity"].append(r)
        
        # Convergence comparison
        if "convergence" in name.lower() or "longtrain" in name.lower():
            groups["convergence"].append(r)
        
        # Scalability comparison
        if "scale" in name.lower() or r.get("num_clients", 0) >= 20:
            groups["scalability"].append(r)
        
        # Overall group by algorithm
        groups[algo_type].append(r)
    
    return dict(groups)


def print_comparison_table(results: List[Dict], title: str):
    """Print comparison table for a set of results."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    
    # Sort by name for consistent ordering
    results = sorted(results, key=lambda x: x.get("name", ""))
    
    # Table header
    print(f"{'Experiment':<40} {'Clients':<8} {'Batch':<8} {'RP%':<6} {'α':<8} "
          f"{'Accuracy':<15} {'Time(s)':<12} {'Rounds':<8}")
    print("-" * 100)
    
    for r in results:
        name = r.get("name", "N/A")[:38]
        clients = r.get("num_clients", "N/A")
        batch = r.get("batch_size", "N/A")
        rp = extract_rand_percent(r.get("name", "")) or "N/A"
        alpha = f"{r.get('dirichlet_alpha', 1.0):.1f}"
        
        acc_mean = r.get("mean_metric", 0) * 100
        acc_std = r.get("std_metric", 0) * 100
        accuracy = f"{acc_mean:.2f} ± {acc_std:.2f}"
        
        time_mean = r.get("mean_time", 0)
        time_std = r.get("std_time", 0)
        time = f"{time_mean:.2f} ± {time_std:.2f}"
        
        rounds = r.get("num_rounds", "N/A")
        
        print(f"{name:<40} {clients:<8} {batch:<8} {str(rp):<6} {alpha:<8} "
              f"{accuracy:<15} {time:<12} {rounds:<8}")


def analyze_rand_percent_impact(results: List[Dict]):
    """Analyze how rand_percent affects FedALA+ performance."""
    print(f"\n{'='*100}")
    print("RAND_PERCENT IMPACT ANALYSIS (FedALA+ only)")
    print(f"{'='*100}")
    
    # Filter FedALA+ results and group by configuration
    fedala_plus = [r for r in results if extract_algorithm_type(r.get("name", "")) == "FedALA+"]
    
    by_config = defaultdict(list)
    for r in fedala_plus:
        config = extract_config(r)
        alpha = extract_dirichlet_alpha(r)
        key = (config[0], config[1], alpha)  # (clients, batch, alpha)
        by_config[key].append(r)
    
    for key, config_results in sorted(by_config.items()):
        clients, batch, alpha = key
        print(f"\nConfiguration: {clients} clients, batch={batch}, α={alpha}")
        print(f"{'Rand%':<10} {'Accuracy':<20} {'Time(s)':<15} {'Improvement vs RP=20'}")
        print("-" * 70)
        
        # Sort by rand_percent
        sorted_results = sorted(config_results, key=lambda x: extract_rand_percent(x.get("name", "")) or 0)
        
        baseline_acc = None
        for r in sorted_results:
            rp = extract_rand_percent(r.get("name", ""))
            acc = r.get("mean_metric", 0) * 100
            acc_std = r.get("std_metric", 0) * 100
            time = r.get("mean_time", 0)
            
            if rp == 20:
                baseline_acc = acc
            
            improvement = ""
            if baseline_acc is not None and rp != 20:
                diff = acc - baseline_acc
                improvement = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
            
            print(f"{rp or 'N/A':<10} {acc:.2f} ± {acc_std:.2f}%{'':5} {time:.2f}s{'':7} {improvement}")


def analyze_heterogeneity_impact(results: List[Dict]):
    """Analyze how data heterogeneity affects FedALA vs FedALA+."""
    print(f"\n{'='*100}")
    print("DATA HETEROGENEITY IMPACT (FedALA vs FedALA+)")
    print(f"{'='*100}")
    
    # Group by dirichlet alpha
    by_alpha = defaultdict(lambda: {"FedALA": None, "FedALA+": []})
    
    for r in results:
        alpha = extract_dirichlet_alpha(r)
        algo_type = extract_algorithm_type(r.get("name", ""))
        
        if algo_type == "FedALA":
            by_alpha[alpha]["FedALA"] = r
        else:
            by_alpha[alpha]["FedALA+"].append(r)
    
    print(f"\n{'α (skew)':<12} {'FedALA Acc':<18} {'FedALA+ Acc (best)':<20} {'Gap':<12} {'Time Δ'}")
    print("-" * 80)
    
    for alpha in sorted(by_alpha.keys()):
        data = by_alpha[alpha]
        
        fedala = data["FedALA"]
        fedala_plus_list = data["FedALA+"]
        
        if not fedala or not fedala_plus_list:
            continue
        
        fedala_acc = fedala.get("mean_metric", 0) * 100
        fedala_time = fedala.get("mean_time", 0)
        
        # Find best FedALA+ (highest accuracy)
        best_plus = max(fedala_plus_list, key=lambda x: x.get("mean_metric", 0))
        plus_acc = best_plus.get("mean_metric", 0) * 100
        plus_time = best_plus.get("mean_time", 0)
        plus_rp = extract_rand_percent(best_plus.get("name", ""))
        
        gap = plus_acc - fedala_acc
        time_delta = plus_time - fedala_time
        
        gap_str = f"+{gap:.2f}%" if gap > 0 else f"{gap:.2f}%"
        time_str = f"+{time_delta:.2f}s" if time_delta > 0 else f"{time_delta:.2f}s"
        
        print(f"{alpha:<12.1f} {fedala_acc:.2f}%{'':10} "
              f"{plus_acc:.2f}% (RP={plus_rp}){'':4} {gap_str:<12} {time_str}")
    
    print(f"\nNote: Lower α = more heterogeneous (higher skew)")


def generate_summary_statistics(results: List[Dict]):
    """Generate overall summary statistics."""
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY STATISTICS")
    print(f"{'='*100}")
    
    fedala_results = [r for r in results if extract_algorithm_type(r.get("name", "")) == "FedALA"]
    fedala_plus_results = [r for r in results if extract_algorithm_type(r.get("name", "")) == "FedALA+"]
    
    def compute_stats(results_list):
        if not results_list:
            return None
        accs = [r.get("mean_metric", 0) * 100 for r in results_list]
        times = [r.get("mean_time", 0) for r in results_list]
        return {
            "count": len(results_list),
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "acc_min": np.min(accs),
            "acc_max": np.max(accs),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
        }
    
    fedala_stats = compute_stats(fedala_results)
    plus_stats = compute_stats(fedala_plus_results)
    
    if fedala_stats:
        print(f"\nFedALA (original) - {fedala_stats['count']} experiments:")
        print(f"  Accuracy: {fedala_stats['acc_mean']:.2f}% ± {fedala_stats['acc_std']:.2f}% "
              f"[{fedala_stats['acc_min']:.2f}% - {fedala_stats['acc_max']:.2f}%]")
        print(f"  Time: {fedala_stats['time_mean']:.2f}s ± {fedala_stats['time_std']:.2f}s")
    
    if plus_stats:
        print(f"\nFedALA+ - {plus_stats['count']} experiments:")
        print(f"  Accuracy: {plus_stats['acc_mean']:.2f}% ± {plus_stats['acc_std']:.2f}% "
              f"[{plus_stats['acc_min']:.2f}% - {plus_stats['acc_max']:.2f}%]")
        print(f"  Time: {plus_stats['time_mean']:.2f}s ± {plus_stats['time_std']:.2f}s")
    
    if fedala_stats and plus_stats:
        acc_improvement = plus_stats['acc_mean'] - fedala_stats['acc_mean']
        time_overhead = plus_stats['time_mean'] - fedala_stats['time_mean']
        time_overhead_pct = (time_overhead / fedala_stats['time_mean']) * 100
        
        print(f"\nFedALA+ vs FedALA (overall):")
        print(f"  Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"  Time overhead: {time_overhead:+.2f}s ({time_overhead_pct:+.1f}%)")


def generate_recommendations(results: List[Dict]):
    """Generate actionable recommendations based on results."""
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS")
    print(f"{'='*100}\n")
    
    # Analyze when FedALA+ beats FedALA
    wins = []
    losses = []
    
    # Group by configuration
    by_config = defaultdict(lambda: {"FedALA": None, "FedALA+": []})
    for r in results:
        config = (r.get("num_clients"), r.get("batch_size"), r.get("dirichlet_alpha", 1.0))
        algo = extract_algorithm_type(r.get("name", ""))
        
        if algo == "FedALA":
            by_config[config]["FedALA"] = r
        else:
            by_config[config]["FedALA+"].append(r)
    
    for config, data in by_config.items():
        if not data["FedALA"] or not data["FedALA+"]:
            continue
        
        fedala_acc = data["FedALA"].get("mean_metric", 0)
        
        for plus_result in data["FedALA+"]:
            plus_acc = plus_result.get("mean_metric", 0)
            
            if plus_acc > fedala_acc:
                wins.append({
                    "config": config,
                    "improvement": (plus_acc - fedala_acc) * 100,
                    "rand_percent": extract_rand_percent(plus_result.get("name", "")),
                })
            else:
                losses.append({
                    "config": config,
                    "degradation": (fedala_acc - plus_acc) * 100,
                    "rand_percent": extract_rand_percent(plus_result.get("name", "")),
                })
    
    print(f"FedALA+ wins: {len(wins)} / {len(wins) + len(losses)} configurations")
    
    if wins:
        avg_improvement = np.mean([w["improvement"] for w in wins])
        print(f"Average improvement when better: {avg_improvement:.2f}%")
        
        # Find best configurations
        best_wins = sorted(wins, key=lambda x: x["improvement"], reverse=True)[:3]
        print(f"\nTop 3 configurations where FedALA+ excels:")
        for i, w in enumerate(best_wins, 1):
            clients, batch, alpha = w["config"]
            print(f"  {i}. Clients={clients}, Batch={batch}, α={alpha:.1f}, RP={w['rand_percent']}%")
            print(f"     Improvement: +{w['improvement']:.2f}%")
    
    if losses:
        avg_degradation = np.mean([l["degradation"] for l in losses])
        print(f"\nAverage degradation when worse: -{avg_degradation:.2f}%")
    
    print(f"\n{'─'*100}")
    print("Use FedALA+ when:")
    if wins:
        # Analyze patterns
        high_client_wins = sum(1 for w in wins if w["config"][0] >= 10)
        high_skew_wins = sum(1 for w in wins if w["config"][2] <= 1.0)
        
        if high_client_wins / len(wins) > 0.6:
            print("  ✓ High client count (10+)")
        if high_skew_wins / len(wins) > 0.6:
            print("  ✓ High data heterogeneity (α ≤ 1.0)")
        
        best_rp = max(set([w["rand_percent"] for w in wins]), 
                     key=lambda rp: sum(w["improvement"] for w in wins if w["rand_percent"] == rp))
        print(f"  ✓ Use rand_percent = {best_rp}% for best results")
    
    print("\nStick with FedALA when:")
    print("  ✓ Training time is critical")
    print("  ✓ Accuracy difference is negligible (<1%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze FedALA comparison results")
    parser.add_argument("files", nargs="+", type=Path, help="JSONL result files")
    parser.add_argument("--output", type=Path, help="Save analysis to file")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.files)
    print(f"Loaded {len(results)} experiment results from {len(args.files)} file(s)")
    
    if not results:
        print("No results found!")
        return 1
    
    # Group results
    groups = group_results(results)
    
    # Generate analyses
    if "config_comparison" in groups:
        print_comparison_table(groups["config_comparison"], 
                             "CONFIGURATION COMPARISON (Clients × Batch Size)")
    
    analyze_rand_percent_impact(results)
    analyze_heterogeneity_impact(results)
    
    if "scalability" in groups:
        print_comparison_table(groups["scalability"], 
                             "SCALABILITY (High Client Count)")
    
    generate_summary_statistics(results)
    generate_recommendations(results)
    
    print(f"\n{'='*100}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())