"""Run Table 6 Graph-FL experiments from table6_configs.

This repo already has `run_experiments.py` which runs a *single* flat YAML experiment
(configs like `configs/proteins/proteins_fedavg.yaml`).

The files under `table6_configs/` are *sweep/meta* configs (datasets + algorithm lists).
This script expands those sweeps into flat experiments and executes them using the
existing `run_experiments.run_experiment` / `run_experiments.run_experiments` helpers.

Usage examples:
  # Preview what will run
  python run_table6_graphfl.py --dry-run

  # Run full sweep
  python run_table6_graphfl.py

  # Override seeds / simulation
  python run_table6_graphfl.py --seeds 0 1 2 3 4 --simulation-mode graph_fl_label_skew

Notes:
  - `run_experiments.py` uses data root `./data`. If your datasets live in `./dataset`,
    make a symlink once: `ln -s dataset data`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

# Import supported algorithms for validation.
import openfgl.config as openfgl_config

# Reuse existing experiment execution code.
import run_experiments


@dataclass(frozen=True)
class SweepDefaults:
    task: str = "graph_cls"
    scenario: str = "graph_fl"

    # === UPDATED TO MATCH APPENDIX A.7 ===
    # Paper uses Label Distribution Skew with Dirichlet α=1
    simulation_mode: str = "graph_fl_label_skew"  # Changed from topology_skew!

    # Graph-FL training defaults
    lr: float = 1e-3           # Paper: 1e-3
    weight_decay: float = 5e-4  # Paper: 5e-4
    dropout: float = 0.5        # Paper: 0.5
    optim: str = "adam"         # Paper: Adam
    batch_size: int = 128       # Paper: 128 (was 32!)
    num_rounds: int = 100       # Paper: 100 (was 50!)
    local_epochs: int = 1       # Paper: 1 epoch per round
    num_clients: int = 10       # Paper: 10-client (was 5!)

    # Label-skew related defaults - IMPORTANT!
    skew_alpha: float = 1.0       # Paper: α=1 for Dirichlet
    dirichlet_alpha: float = 1.0  # Paper: α=1 (was 10.0!)


_ALGO_NAME_MAP = {
    # Table6 naming -> openfgl naming
    "Local": "isolate",
    "FedAvg": "fedavg",
    "FedProx": "fedprox",
    "Scaffold": "scaffold",
    "MOON": "moon",
    "FedProto": "fedproto",
    "FedTGP": "fedtgp",
    "FedStar": "fedstar",
    "GCFL+": "gcfl_plus",
    "FedSage+": "fedsage_plus",
    "FedALA": "fedala",
    "FedALA+": "fedala_plus",
    "FedALARegularized": "fedala_regularized",
    "FedALA+Regularized": "fedala_plus_regularized",
    "FedALARC": "fedalarc",
    # Keep placeholders (may be unsupported in this fork)
    "FedNH": "fednh",
}

_MODEL_NAME_MAP = {
    "GIN": "gin",
    "SAGPooling": "global_sag",
    "EdgePooling": "global_edge",
    "PANPooling": "global_pan",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected dict YAML at {path}, got {type(loaded).__name__}")
    return loaded


def _to_openfgl_algorithm(name: str) -> str:
    return _ALGO_NAME_MAP.get(name, name).lower()


def _to_openfgl_model(name: str) -> str:
    return _MODEL_NAME_MAP.get(name, name).lower()


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _supported_algorithms_set() -> set:
    # openfgl_config defines this at module scope.
    supported = getattr(openfgl_config, "supported_fl_algorithm", None)
    if not supported:
        # Fallback: rely on argparse choices embedded in openfgl_config.args
        return set()
    return set(supported)


def _build_experiment(
    *,
    name: str,
    dataset: str,
    algorithm: str,
    model: str,
    simulation_mode: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    optim: str,
    dirichlet_alpha: float,
    skew_alpha: float,
    layer_idx: int = None,
    lambda_graph: float = 0.0,        # NEW
    graph_reg_type: str = "laplacian", # NEW
    defaults: SweepDefaults,
    # === NEW: FedALA+ parameters ===
    eta: float = 1.0,
    rand_percent: int = 80,
    threshold: float = 0.1,
    num_pre_loss: int = 10,
    use_disagreement: bool = False,
    selection_frequency: int = 1,
    min_disagreement_samples: Optional[int] = None,
) -> Dict[str, Any]:
    exp: Dict[str, Any] = {
        "name": name,
        "dataset": [dataset],
        "task": defaults.task,
        "scenario": defaults.scenario,
        "algorithm": algorithm,
        "model": [model],
        "simulation_mode": simulation_mode,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "optim": optim,
        "metrics": ["accuracy"],
        "dirichlet_alpha": dirichlet_alpha,
        "skew_alpha": skew_alpha,
    }

    # Only used by label_skew sims.
    exp["skew_alpha"] = skew_alpha
    if layer_idx is not None:
        exp["layer_idx"] = layer_idx
    if lambda_graph is not None:
        exp["lambda_graph"] = lambda_graph
    if graph_reg_type is not None:
        exp["graph_reg_type"] = graph_reg_type
    # === NEW: Add FedALA/FedALA+ specific parameters ===
    if algorithm in ["fedala", "fedala_plus", "fedala_regularized", "fedala_plus_regularized", "fedalarc"]:
        exp["eta"] = eta
        exp["layer_idx"] = layer_idx
        exp["rand_percent"] = rand_percent
        exp["threshold"] = threshold
        exp["num_pre_loss"] = num_pre_loss
        
        # FedALA+ specific (disagreement sampling)
        if "plus" in algorithm:
            exp["use_disagreement"] = use_disagreement
            exp["selection_frequency"] = selection_frequency
            exp["min_disagreement_samples"] = min_disagreement_samples

    return exp


def _iter_table6_experiments(
    *,
    repo_root: Path,
    table6_config_path: Path,
    defaults: SweepDefaults,
    simulation_mode: str,
    seeds: Sequence[int],
    only_groups: Optional[Sequence[str]],
    only_datasets: Optional[Sequence[str]],
    strict: bool,
    overrides: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (experiments, warnings)."""

    cfg = _load_yaml(table6_config_path)
    includes = _ensure_list(cfg.get("include"))

    # Load included YAMLs relative to config directory.
    config_dir = table6_config_path.parent
    included: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (config_dir / inc).resolve()
        included[inc] = _load_yaml(inc_path)

    datasets_yaml = included.get("datasets.yaml", {})
    datasets = _ensure_list(datasets_yaml.get("datasets"))

    local_yaml = included.get("local.yaml", {}).get("local", {})
    fl_yaml = included.get("fl.yaml", {}).get("fl", {})
    fgl_yaml = included.get("fgl.yaml", {}).get("fgl", {})

    fedala_yaml = included.get("fedala.yaml", {}).get("fedala", {})
    fedala_reg_yaml = included.get("fedala_reg.yaml", {}).get("fedala_reg", {})
    fedala_plus_yaml = included.get("fedala_plus.yaml", {}).get("fedala_plus", {})
    fedala_plus_regularized_yaml = included.get("fedala_plus_regularized.yaml", {}).get("fedala_plus_regularized", {})
    fedala_regularized_yaml = included.get("fedala_regularized.yaml", {}).get("fedala_regularized", {})
    fedalarc_yaml = included.get("fedalarc.yaml", {}).get("fedalarc", {})

    selected_groups = set(only_groups or ["local", "fl", "fgl"])
    if only_groups is not None and not selected_groups:
        raise ValueError("--groups provided but empty")

    if only_datasets is not None:
        dataset_filter = set(only_datasets)
        datasets = [d for d in datasets if d in dataset_filter]

    supported_algos = _supported_algorithms_set()

    warnings: List[str] = []
    experiments: List[Dict[str, Any]] = []

    def add_group(group_name: str, group_cfg: Dict[str, Any]) -> None:
        group_algos = _ensure_list(group_cfg.get("algorithms"))
        group_common = group_cfg.get("common", {}) or {}

        # Model(s)
        if group_name == "local":
            models = _ensure_list(group_cfg.get("models"))
            if not models:
                models = ["GIN"]
        else:
            models = _ensure_list(group_cfg.get("model"))
            if not models:
                models = ["GIN"]

        for dataset in datasets:
            for algo_display in group_algos:
                algo = _to_openfgl_algorithm(str(algo_display))

                if supported_algos and algo not in supported_algos:
                    msg = f"Skipping unsupported algorithm '{algo_display}' -> '{algo}'"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                    continue

                for model_display in models:
                    model = _to_openfgl_model(str(model_display))

                    # Common hyperparams from table6 (with CLI overrides)
                    num_clients = int(overrides.get("num_clients", group_common.get("num_clients", 10)))
                    num_rounds = int(overrides.get("num_rounds", group_common.get("rounds", defaults.num_rounds)))  # 100
                    local_epochs = int(overrides.get("local_epochs", group_common.get("local_steps", defaults.local_epochs)))  # 1
                    batch_size = int(overrides.get("batch_size", group_common.get("batch_size", defaults.batch_size)))  # 128

                    lr = float(overrides.get("lr", group_common.get("lr", defaults.lr)))  # 1e-3
                    weight_decay = float(overrides.get("weight_decay", group_common.get("weight_decay", defaults.weight_decay)))  # 5e-4
                    dropout = float(overrides.get("dropout", group_common.get("dropout", defaults.dropout)))  # 0.5
                    optim = str(overrides.get("optim", group_common.get("optimizer", defaults.optim)))  # adam

                    dirichlet_alpha = float(overrides.get("dirichlet_alpha", defaults.dirichlet_alpha))
                    skew_alpha = float(overrides.get("skew_alpha", defaults.skew_alpha))

                    layer_idx = int(overrides.get("layer_idx")) if overrides.get("layer_idx") is not None else None

                    lambda_graph = float(overrides.get("lambda_graph", group_common.get("lambda_graph", 0.0)))
                    graph_reg_type = str(overrides.get("graph_reg_type", group_common.get("graph_reg_type", "laplacian")))

                    # === NEW: FedALA+ parameters ===
                    eta = float(overrides.get("eta", group_common.get("eta", 1.0)))
                    rand_percent = int(overrides.get("rand_percent", group_common.get("rand_percent", 80)))
                    threshold = float(overrides.get("threshold", group_common.get("threshold", 0.1)))
                    num_pre_loss = int(overrides.get("num_pre_loss", group_common.get("num_pre_loss", 10)))
                    
                    use_disagreement = bool(overrides.get("use_disagreement", group_common.get("use_disagreement", False)))
                    selection_frequency = int(overrides.get("selection_frequency", group_common.get("selection_frequency", 1)))
                    min_disagreement_samples = overrides.get("min_disagreement_samples", group_common.get("min_disagreement_samples"))

                    exp_name = f"Table6_{dataset}_{group_name}_{algo}_{model}"

                    experiments.append(
                        _build_experiment(
                            name=exp_name,
                            dataset=dataset,
                            algorithm=algo,
                            model=model,
                            simulation_mode=simulation_mode,
                            num_clients=num_clients,
                            num_rounds=num_rounds,
                            local_epochs=local_epochs,
                            batch_size=batch_size,
                            lr=lr,
                            weight_decay=weight_decay,
                            dropout=dropout,
                            optim=optim,
                            dirichlet_alpha=dirichlet_alpha,
                            skew_alpha=skew_alpha,
                            layer_idx=layer_idx,
                            lambda_graph=lambda_graph,
                            graph_reg_type=graph_reg_type,
                            defaults=defaults,
                            # === NEW: FedALA+ params ===
                            eta=eta,
                            rand_percent=rand_percent,
                            threshold=threshold,
                            num_pre_loss=num_pre_loss,
                            use_disagreement=use_disagreement,
                            selection_frequency=selection_frequency,
                            min_disagreement_samples=min_disagreement_samples,
                        )
                    )

    if "local" in selected_groups:
        add_group("local", local_yaml)
    if "fl" in selected_groups:
        add_group("fl", fl_yaml)
    if "fgl" in selected_groups:
        add_group("fgl", fgl_yaml)
    if "fedala" in selected_groups:
        add_group("fedala", fedala_yaml)
    if "fedala_reg" in selected_groups:
        add_group("fedala_reg", fedala_reg_yaml)
    if "fedala_plus" in selected_groups:
        add_group("fedala_plus", fedala_plus_yaml)
    if "fedala_regularized" in selected_groups:
        add_group("fedala_regularized", fedala_regularized_yaml)
    if "fedala_plus_regularized" in selected_groups:
        add_group("fedala_plus_regularized", fedala_plus_regularized_yaml)
    if "fedalarc" in selected_groups:
        add_group("fedalarc", fedalarc_yaml)    

    return experiments, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Table 6 Graph-FL sweep configs")

    parser.add_argument(
        "--table6-config",
        type=str,
        default="table6_configs/graph_fl_experiments.yaml",
        help="Path to Table6 meta-config YAML",
    )

    parser.add_argument(
        "--simulation-mode",
        type=str,
        default="graph_fl_label_skew"
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds")
    parser.add_argument("--groups", type=str, nargs="+", default=None, help="Subset: local fl fgl")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Run only these datasets")

    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--skew-alpha", type=float, default=None)

    parser.add_argument("--strict", action="store_true", help="Fail on unsupported algorithms")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only")

    # FedALA hyperparameters (per paper Table 5/6)
    parser.add_argument("--eta", type=float, default=1.0, help="Learning rate for ALA weights")
    parser.add_argument("--layer_idx", type=int, default=1, help="p=1: adapt only last layer (paper default)")
    parser.add_argument("--rand_percent", type=int, default=80, help="s=80: percentage of data for ALA")
    parser.add_argument("--threshold", type=float, default=0.1, help="Convergence threshold for ALA")
    parser.add_argument("--num_pre_loss", type=int, default=10, help="Window size for convergence check")

    parser.add_argument("--layer-idx", type=int, default=None, help="FedALA layer_idx param")

    parser.add_argument("--lambda-graph", type=float, default=None, 
                        help="Graph regularization strength (0=disabled)")
    parser.add_argument("--graph-reg-type", type=str, default=None,
                        choices=["laplacian", "dirichlet"],
                        help="Type of graph regularization")
    # === NEW: FedALA+ specific ===
    parser.add_argument("--use-disagreement", action="store_true", default=None, help="Use disagreement-based sampling")
    parser.add_argument("--no-use-disagreement", dest="use_disagreement", action="store_false", help="Disable disagreement sampling")
    parser.add_argument("--selection-frequency", type=int, default=None, help="Recompute disagreement every N rounds")
    parser.add_argument("--min-disagreement-samples", type=int, default=None, help="Minimum disagreement samples before fallback")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    table6_path = (repo_root / args.table6_config).resolve()
    simulation_mode = args.simulation_mode

    defaults = SweepDefaults()

    # Pull seeds from meta-config if not overridden.
    cfg = _load_yaml(table6_path)
    meta_seeds = (
        _ensure_list(((cfg.get("experiment") or {}).get("seeds")))
        or _ensure_list(cfg.get("seeds"))
        or [0, 1, 2, 3, 4]
    )
    seeds: Sequence[int] = args.seeds if args.seeds is not None else [int(s) for s in meta_seeds]

    overrides: Dict[str, Any] = {}
    if args.num_clients is not None:
        overrides["num_clients"] = args.num_clients
    if args.num_rounds is not None:
        overrides["num_rounds"] = args.num_rounds
    if args.local_epochs is not None:
        overrides["local_epochs"] = args.local_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.optim is not None:
        overrides["optim"] = args.optim
    if args.dirichlet_alpha is not None:
        overrides["dirichlet_alpha"] = args.dirichlet_alpha
    if args.skew_alpha is not None:
        overrides["skew_alpha"] = args.skew_alpha
    if args.layer_idx is not None:
        overrides["layer_idx"] = args.layer_idx
    if args.lambda_graph is not None:
        overrides["lambda_graph"] = args.lambda_graph
    if args.graph_reg_type is not None:
        overrides["graph_reg_type"] = args.graph_reg_type
    if args.use_disagreement is not None:
        overrides["use_disagreement"] = args.use_disagreement
    if args.selection_frequency is not None:
        overrides["selection_frequency"] = args.selection_frequency
    if args.min_disagreement_samples is not None:
        overrides["min_disagreement_samples"] = args.min_disagreement_samples
    if args.eta is not None:
        overrides["eta"] = args.eta
    if args.layer_idx is not None:
        overrides["layer_idx"] = args.layer_idx
    if args.rand_percent is not None:
        overrides["rand_percent"] = args.rand_percent
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.num_pre_loss is not None:
        overrides["num_pre_loss"] = args.num_pre_loss        

    experiments, warnings = _iter_table6_experiments(
        repo_root=repo_root,
        table6_config_path=table6_path,
        defaults=defaults,
        simulation_mode=simulation_mode,
        seeds=seeds,
        only_groups=args.groups,
        only_datasets=args.datasets,
        strict=args.strict,
        overrides=overrides,
    )

    # Ensure data root exists (run_experiments hardcodes ./data)
    if not (repo_root / "data").exists() and (repo_root / "dataset").exists():
        warnings.append(
            "Data root './data' not found but './dataset' exists. Consider: ln -s dataset data"
        )

    for w in warnings:
        print(f"[warn] {w}")

    print(f"Planned runs: {len(experiments)} experiments x {len(seeds)} seeds")
    print(f"simulation_mode={simulation_mode}")

    if args.dry_run:

        for exp in experiments[:50]:
                extra_params_str = ""
                if 'rand_percent' in exp:
                    extra_params_str = f" | s={exp['rand_percent']}%"
                    if exp.get('use_disagreement'):
                        extra_params_str += f" disagr=✓"
                    else:
                        extra_params_str += f" disagr=✗"
                print(f"- {exp['name']} | {exp['dataset'][0]} | {exp['algorithm']} | {exp['model'][0]} | {extra_params_str}")
        if len(experiments) > 50:
            print(f"... ({len(experiments) - 50} more)")
        return 0

    # Where to write aggregated results
    output_dir = (repo_root / (cfg.get("experiment") or {}).get("output_dir", "results/graph_fl/")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"table6_graphfl_{run_id}.jsonl"

    print(f"Writing aggregated results to: {jsonl_path}")

    with jsonl_path.open("w") as out:
        for idx, exp in enumerate(experiments, start=1):
            print("\n" + "#" * 80)
            print(f"[{idx}/{len(experiments)}] {exp['name']}")
            print("#" * 80)

            result = run_experiments.run_experiments(exp, seeds=list(seeds))

            row = {
                # Identification
                "name": result["name"],
                "dataset": exp["dataset"][0],
                "group": exp["name"].split("_")[2] if "_" in exp["name"] else "N/A",
                "algorithm": exp["algorithm"],
                "model": exp["model"][0],
                
                # Simulation settings
                "simulation_mode": exp["simulation_mode"],
                "num_clients": exp["num_clients"],
                "dirichlet_alpha": exp.get("dirichlet_alpha"),
                
                # Training settings
                "num_rounds": exp["num_rounds"],
                "local_epochs": exp["local_epochs"],
                "batch_size": exp["batch_size"],
                "lr": exp["lr"],
                "weight_decay": exp["weight_decay"],
                "dropout": exp["dropout"],

                # FedALA-specific params
                "layer_idx": exp.get("layer_idx"),

                "lambda_graph": exp.get("lambda_graph"),
                "graph_reg_type": exp.get("graph_reg_type"),
                    
                # === NEW: ALA+-specific metrics ===
                "rand_percent": exp.get("rand_percent"), #sample percent s
                "use_disagreement": exp.get("use_disagreement"),
                "selection_frequency": exp.get("selection_frequency"),
                "min_disagreement_samples": exp.get("min_disagreement_samples"),
    
                # ALA+ runtime metrics (if available from result)
                "mean_selection_time": result.get("mean_selection_time"),
                "mean_ala_training_time": result.get("mean_ala_training_time"),
                "cache_hit_rate": result.get("cache_hit_rate"),
                "fallback_count": result.get("fallback_count"),
                
                # === PRIMARY RESULTS ===
                # Test accuracy
                "mean_metric": result["mean_metric"],
                "std_metric": result["std_metric"],
                
                # NEW: Validation accuracy (useful for detecting overfitting)
                "mean_val_metric": result.get("mean_val_metric"),
                "std_val_metric": result.get("std_val_metric"),
                
                # NEW: Convergence info (which round achieved best result)
                "mean_best_round": result.get("mean_best_round"),
                "std_best_round": result.get("std_best_round"),
                
                # Time
                "mean_time": result["mean_time"],
                "std_time": result["std_time"],
                
                # Communication
                "mean_comm": result["mean_comm"],
                "std_comm": result["std_comm"],
                
                # === PER-SEED DETAILS (for reproducibility) ===
                "seeds": list(seeds),
                "test_metrics": result.get("test_metrics", []),
                "val_metrics": result.get("val_metrics", []),
                "best_rounds": result.get("best_rounds", []),
                "times": result.get("times", []),
                "comm_costs": result.get("comm_costs", []),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
