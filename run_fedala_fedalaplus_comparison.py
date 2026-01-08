"""Comprehensive FedALA vs FedALA+ comparison experiments.

This script runs systematic experiments to compare original FedALA with FedALA+
across different configurations to identify when FedALA+ provides benefits.

Usage:
    # Run full suite (may take hours)
    python run_fedala_comparison.py
    
    # Run minimal test suite (faster)
    python run_fedala_comparison.py --mode minimal
    
    # Run only heterogeneity tests
    python run_fedala_comparison.py --mode heterogeneity
    
    # Dry run to see what will execute
    python run_fedala_comparison.py --dry-run
    
    # Custom seeds
    python run_fedala_comparison.py --seeds 0 1 2
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def run_experiment(config: Dict[str, Any], dry_run: bool = False, experiment_num: int = 0, total: int = 0) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    import time
    start_time = time.time()
    
    # Build command
    cmd = ["python", "run_table6_graphfl.py"]
    cmd.extend(["--groups", "fedala"])
    cmd.extend(["--datasets", "PROTEINS"])
    
    # Add all config parameters
    if "num_clients" in config:
        cmd.extend(["--num-clients", str(config["num_clients"])])
    if "batch_size" in config:
        cmd.extend(["--batch-size", str(config["batch_size"])])
    if "num_rounds" in config:
        cmd.extend(["--num-rounds", str(config["num_rounds"])])
    if "seeds" in config:
        cmd.extend(["--seeds"] + [str(s) for s in config["seeds"]])
    if "dirichlet_alpha" in config:
        cmd.extend(["--dirichlet-alpha", str(config["dirichlet_alpha"])])
    if "simulation_mode" in config:
        cmd.extend(["--simulation-mode", config["simulation_mode"]])
    
    # FedALA+ specific parameters
    if "rand_percent" in config:
        cmd.extend(["--rand-percent", str(config["rand_percent"])])
    if "eta" in config:
        cmd.extend(["--eta", str(config["eta"])])
    if "layer_idx" in config:
        cmd.extend(["--layer_idx", str(config["layer_idx"])])
    
    print(f"\n{'='*80}")
    print(f"[{experiment_num}/{total}] {config['name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    sys.stdout.flush()  # Force flush before subprocess
    
    if dry_run:
        return {"name": config["name"], "status": "dry_run", "config": config}
    
    try:
        # Run with live output streaming
        print(f"⏳ Executing experiment (this may take several minutes)...\n")
        sys.stdout.flush()
        
        result = subprocess.run(cmd, check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
        sys.stdout.flush()
        
        return {
            "name": config["name"],
            "status": "success",
            "config": config,
            "elapsed_time": elapsed,
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        sys.stdout.flush()
        
        return {
            "name": config["name"],
            "status": "failed",
            "config": config,
            "error": str(e),
            "elapsed_time": elapsed,
        }
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        raise


def get_full_suite(seeds: List[int]) -> List[Dict[str, Any]]:
    """Complete experimental suite (all hypotheses)."""
    configs = []
    
    # H1: Configuration matching (baseline comparison)
    base_configs = [
        {"clients": 5, "batch": 32},
        {"clients": 10, "batch": 32},
        {"clients": 5, "batch": 128},
        {"clients": 10, "batch": 128},
    ]
    
    for base in base_configs:
        # Original FedALA (no rand_percent parameter)
        configs.append({
            "name": f"FedALA_C{base['clients']}_B{base['batch']}",
            "num_clients": base["clients"],
            "batch_size": base["batch"],
            "num_rounds": 100,
            "seeds": seeds,
            "dirichlet_alpha": 1.0,
        })
        
        # FedALA+ with different rand_percent values
        for rp in [20, 40, 60, 80]:
            configs.append({
                "name": f"FedALA+_C{base['clients']}_B{base['batch']}_RP{rp}",
                "num_clients": base["clients"],
                "batch_size": base["batch"],
                "num_rounds": 100,
                "rand_percent": rp,
                "seeds": seeds,
                "dirichlet_alpha": 1.0,
            })
    
    # H2: Heterogeneity impact
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        # Original FedALA
        configs.append({
            "name": f"FedALA_Hetero_alpha{alpha}",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 100,
            "seeds": seeds,
            "dirichlet_alpha": alpha,
        })
        
        # FedALA+ with low and high rand_percent
        for rp in [20, 80]:
            configs.append({
                "name": f"FedALA+_Hetero_alpha{alpha}_RP{rp}",
                "num_clients": 10,
                "batch_size": 128,
                "num_rounds": 100,
                "rand_percent": rp,
                "seeds": seeds,
                "dirichlet_alpha": alpha,
            })
    
    # H3: Convergence analysis (longer training)
    for rp in [20, 40, 60, 80]:
        configs.append({
            "name": f"FedALA+_Convergence_RP{rp}",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 200,  # Double rounds
            "rand_percent": rp,
            "seeds": seeds,
            "dirichlet_alpha": 1.0,
        })
    
    # H4: Scalability (high client count)
    for num_clients in [20, 50]:
        configs.append({
            "name": f"FedALA_Scale_C{num_clients}",
            "num_clients": num_clients,
            "batch_size": 128,
            "num_rounds": 100,
            "seeds": seeds,
            "dirichlet_alpha": 1.0,
        })
        
        for rp in [20, 80]:
            configs.append({
                "name": f"FedALA+_Scale_C{num_clients}_RP{rp}",
                "num_clients": num_clients,
                "batch_size": 128,
                "num_rounds": 100,
                "rand_percent": rp,
                "seeds": seeds,
                "dirichlet_alpha": 1.0,
            })
    
    return configs


def get_minimal_suite(seeds: List[int]) -> List[Dict[str, Any]]:
    """Minimal test suite for quick validation."""
    configs = []
    
    # Key configuration: 10 clients, batch 128 (where you saw differences)
    configs.append({
        "name": "FedALA_baseline",
        "num_clients": 10,
        "batch_size": 128,
        "num_rounds": 100,
        "seeds": seeds,
        "dirichlet_alpha": 1.0,
    })
    
    # FedALA+ sweep on rand_percent
    for rp in [20, 40, 60, 80]:
        configs.append({
            "name": f"FedALA+_RP{rp}",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 100,
            "rand_percent": rp,
            "seeds": seeds,
            "dirichlet_alpha": 1.0,
        })
    
    # Test heterogeneity impact (3 levels)
    for alpha in [0.1, 1.0, 10.0]:
        configs.append({
            "name": f"FedALA_hetero{alpha}",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 100,
            "seeds": seeds,
            "dirichlet_alpha": alpha,
        })
        
        configs.append({
            "name": f"FedALA+_hetero{alpha}_RP80",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 100,
            "rand_percent": 80,
            "seeds": seeds,
            "dirichlet_alpha": alpha,
        })
    
    return configs


def get_heterogeneity_suite(seeds: List[int]) -> List[Dict[str, Any]]:
    """Focus on data heterogeneity experiments."""
    configs = []
    
    # Sweep heterogeneity levels
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        # Original FedALA
        configs.append({
            "name": f"FedALA_alpha{alpha}",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 100,
            "seeds": seeds,
            "dirichlet_alpha": alpha,
        })
        
        # FedALA+ with optimal rand_percent
        configs.append({
            "name": f"FedALA+_alpha{alpha}_RP80",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 100,
            "rand_percent": 80,
            "seeds": seeds,
            "dirichlet_alpha": alpha,
        })
    
    return configs


def get_config_suite(seeds: List[int]) -> List[Dict[str, Any]]:
    """Focus on configuration (clients/batch) experiments."""
    configs = []
    
    clients_list = [5, 10, 20]
    batch_list = [32, 64, 128]
    
    for clients in clients_list:
        for batch in batch_list:
            # Original FedALA
            configs.append({
                "name": f"FedALA_C{clients}_B{batch}",
                "num_clients": clients,
                "batch_size": batch,
                "num_rounds": 100,
                "seeds": seeds,
                "dirichlet_alpha": 1.0,
            })
            
            # FedALA+ with best rand_percent
            configs.append({
                "name": f"FedALA+_C{clients}_B{batch}_RP80",
                "num_clients": clients,
                "batch_size": batch,
                "num_rounds": 100,
                "rand_percent": 80,
                "seeds": seeds,
                "dirichlet_alpha": 1.0,
            })
    
    return configs


def get_convergence_suite(seeds: List[int]) -> List[Dict[str, Any]]:
    """Focus on convergence speed experiments."""
    configs = []
    
    # Extended training to see convergence patterns
    for rp in [20, 40, 60, 80]:
        configs.append({
            "name": f"FedALA+_LongTrain_RP{rp}",
            "num_clients": 10,
            "batch_size": 128,
            "num_rounds": 300,  # 3x normal
            "rand_percent": rp,
            "seeds": seeds,
            "dirichlet_alpha": 1.0,
        })
    
    # Baseline for comparison
    configs.append({
        "name": "FedALA_LongTrain",
        "num_clients": 10,
        "batch_size": 128,
        "num_rounds": 300,
        "seeds": seeds,
        "dirichlet_alpha": 1.0,
    })
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive FedALA vs FedALA+ comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "minimal", "heterogeneity", "config", "convergence"],
        default="minimal",
        help="Experiment suite to run (default: minimal)"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to use (default: 0 1 2 3 4)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments without running"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fedala_comparison",
        help="Directory for results (default: results/fedala_comparison)"
    )
    
    args = parser.parse_args()
    
    # Select experiment suite
    if args.mode == "full":
        configs = get_full_suite(args.seeds)
    elif args.mode == "minimal":
        configs = get_minimal_suite(args.seeds)
    elif args.mode == "heterogeneity":
        configs = get_heterogeneity_suite(args.seeds)
    elif args.mode == "config":
        configs = get_config_suite(args.seeds)
    elif args.mode == "convergence":
        configs = get_convergence_suite(args.seeds)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print(f"\n{'='*80}")
    print(f"FedALA vs FedALA+ Comparison Suite")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Total experiments: {len(configs)}")
    print(f"Seeds per experiment: {len(args.seeds)}")
    print(f"Total runs: {len(configs)} experiments × {len(args.seeds)} seeds")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("DRY RUN - Experiments that would be executed:\n")
        for i, cfg in enumerate(configs, 1):
            print(f"{i}. {cfg['name']}")
            print(f"   Clients: {cfg.get('num_clients', 'N/A')}, "
                  f"Batch: {cfg.get('batch_size', 'N/A')}, "
                  f"Rounds: {cfg.get('num_rounds', 'N/A')}")
            print(f"   Dirichlet α: {cfg.get('dirichlet_alpha', 'N/A')}, "
                  f"Rand %: {cfg.get('rand_percent', 'N/A (original FedALA)')}")
            print()
        print(f"Total: {len(configs)} experiments")
        return 0
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comparison_{args.mode}_{run_id}.jsonl"
    
    print(f"Results will be saved to: {results_file}\n")
    
    results = []
    total_start = time.time()
    
    with results_file.open("w") as f:
        for i, config in enumerate(configs, 1):
            result = run_experiment(config, dry_run=False, experiment_num=i, total=len(configs))
            results.append(result)
            
            # Save incrementally
            f.write(json.dumps(result) + "\n")
            f.flush()
            
            # Progress summary
            elapsed = time.time() - total_start
            avg_time = elapsed / i
            remaining = avg_time * (len(configs) - i)
            
            print(f"\n{'─'*80}")
            if result["status"] == "success":
                print(f"✓ [{i}/{len(configs)}] SUCCESS: {config['name']}")
            else:
                print(f"✗ [{i}/{len(configs)}] FAILED: {config['name']}")
            
            print(f"Progress: {i}/{len(configs)} ({100*i/len(configs):.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
            print(f"{'─'*80}\n")
            sys.stdout.flush()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*80}\n")
    
    # Create summary file
    summary_file = output_dir / f"summary_{args.mode}_{run_id}.json"
    summary = {
        "mode": args.mode,
        "total_experiments": len(results),
        "successful": success,
        "failed": failed,
        "seeds": args.seeds,
        "timestamp": run_id,
        "results_file": str(results_file),
    }
    
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())