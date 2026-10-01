#!/usr/bin/env python3
"""Main experiment runner: data splitting, baseline training, GNN training, ablations, and plotting."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compilerlens.data import (
    GraphData,
    TpuTileDataset,
    create_graph_disjoint_splits,
    load_npz_graph,
)
from compilerlens.evaluate import evaluate_predictions, save_evaluation_report
from compilerlens.features import FeaturePreprocessor
from compilerlens.models import (
    ConfigConditionedGNN,
    GBTBaseline,
    GraphFreeModel,
    GraphSAGEModel,
    RandomRankingBaseline,
)
from compilerlens.predict import export_test_predictions, predict_graph_candidates
from compilerlens.train import Trainer, get_device, set_seed
from compilerlens.visualize import (
    plot_ablation_comparison,
    plot_baseline_comparison,
    plot_case_studies,
    plot_computation_graph_sample,
    plot_dataset_distributions,
    plot_model_architecture,
    plot_pipeline_diagram,
    plot_predicted_vs_observed,
    plot_regret_distribution,
    plot_topk_regret_curve,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiment")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CompilerLens experiment.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config YAML.")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports.")
    parser.add_argument("--figures-dir", type=str, default="assets/figures", help="Output directory for figures.")
    parser.add_argument("--run-baselines", action="store_true", default=True, help="Run baseline models.")
    parser.add_argument("--run-ablations", action="store_true", default=True, help="Run ablation models.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(config.get("device", "auto"))
    logger.info(f"Using compute device: {device}")

    # 1. Discover and partition dataset files
    raw_dir = Path(config["data"]["raw_dir"]) / "npz" / "tile" / config["data"]["backend"]
    train_files = sorted(list((raw_dir / "train").glob("*.npz")))
    val_files = sorted(list((raw_dir / "valid").glob("*.npz")))
    test_files = sorted(list((raw_dir / "test").glob("*.npz")))

    logger.info(f"Discovered files: {len(train_files)} train, {len(val_files)} valid, {len(test_files)} test")

    # Limit graph counts based on config profile
    max_train = config["data"].get("max_train_graphs")
    if max_train is not None and max_train < len(train_files):
        train_files = train_files[:max_train]

    max_val = config["data"].get("max_val_graphs")
    if max_val is not None and max_val < len(val_files):
        val_files = val_files[:max_val]

    max_test = config["data"].get("max_test_graphs")
    if max_test is not None and max_test < len(test_files):
        test_files = test_files[:max_test]

    logger.info(f"Active experiment subset: {len(train_files)} train, {len(val_files)} valid, {len(test_files)} test")

    # Graph-disjoint development split from official train
    val_ratio = config["data"].get("val_ratio", 0.2)
    train_dev_files, val_dev_files = create_graph_disjoint_splits(train_files, val_ratio=val_ratio, seed=seed)
    logger.info(f"Partitioned train into: {len(train_dev_files)} train_dev, {len(val_dev_files)} val_dev")

    # 2. Fit preprocessor STRICTLY on train_dev
    logger.info("Loading train_dev graphs to fit feature preprocessor...")
    train_dev_graphs = [load_npz_graph(p, split="train_dev") for p in train_dev_files]
    preprocessor = FeaturePreprocessor(
        node_feat_dim=config["model"]["node_feat_dim"],
        config_feat_dim=config["model"]["config_feat_dim"],
    )
    preprocessor.fit(train_dev_graphs)
    preprocessor.save(output_dir / "preprocessor_stats.json")
    logger.info("Fitted and saved preprocessor statistics.")

    # Load official valid graphs for final evaluation
    logger.info("Loading valid graphs for final evaluation...")
    valid_graphs = [load_npz_graph(p, split="valid") for p in val_files]

    # Create PyTorch datasets
    max_configs = config["data"].get("max_configs_per_graph", 32)
    train_dataset = TpuTileDataset(train_dev_files, split_name="train_dev", max_configs=max_configs, seed=seed)
    val_dev_dataset = TpuTileDataset(val_dev_files, split_name="val_dev", max_configs=None, seed=seed)
    official_val_dataset = TpuTileDataset(val_files, split_name="valid", max_configs=None, seed=seed)

    experiment_results: Dict[str, Any] = {
        "config": config,
        "splits": {
            "train_dev_count": len(train_dev_files),
            "val_dev_count": len(val_dev_files),
            "official_valid_count": len(val_files),
            "test_count": len(test_files),
        },
        "models": {},
        "ablations": {},
    }

    # 3. Baselines Evaluation
    if args.run_baselines:
        logger.info("=== Running Baseline A: Random Ranking ===")
        rand_baseline = RandomRankingBaseline(num_draws=50, seed=seed)
        draws = rand_baseline.evaluate_all(valid_graphs)
        # Average across draws
        rand_top1 = []
        rand_top5 = []
        rand_top10 = []
        for draw_preds in draws:
            summary = evaluate_predictions(draw_preds, valid_graphs, top_k_list=[1, 5, 10], bootstrap_rounds=50, seed=seed)
            rand_top1.append(summary["top1_regret_mean"])
            rand_top5.append(summary["best_of_topk"][5]["mean"])
            rand_top10.append(summary["best_of_topk"][10]["mean"])

        rand_summary = {
            "top1_regret_mean": float(np.mean(rand_top1)),
            "top1_regret_median": float(np.median(rand_top1)),
            "top1_regret_ci95": [float(np.percentile(rand_top1, 2.5)), float(np.percentile(rand_top1, 97.5))],
            "spearman_mean": 0.0,
            "spearman_ci95": [0.0, 0.0],
            "best_of_topk": {
                1: {"mean": float(np.mean(rand_top1))},
                5: {"mean": float(np.mean(rand_top5))},
                10: {"mean": float(np.mean(rand_top10))},
            },
            "param_count": 0,
        }
        experiment_results["models"]["Random"] = rand_summary
        logger.info(f"Random Baseline Top-1 Regret: {rand_summary['top1_regret_mean']:.4f}")

        logger.info("=== Running Baseline B: Gradient-Boosted Trees ===")
        gbt = GBTBaseline(max_iter=60, seed=seed)
        gbt.fit(train_dev_graphs, preprocessor=preprocessor, max_configs_per_graph=max_configs)
        gbt_preds = gbt.predict_all(valid_graphs)
        gbt_summary = evaluate_predictions(gbt_preds, valid_graphs, top_k_list=[1, 5, 10], bootstrap_rounds=200, seed=seed)
        gbt_summary["param_count"] = gbt.param_count
        gbt_summary["fit_time_seconds"] = gbt.fit_time_seconds
        experiment_results["models"]["GBT"] = gbt_summary
        logger.info(f"GBT Baseline Top-1 Regret: {gbt_summary['top1_regret_mean']:.4f}, Spearman: {gbt_summary.get('spearman_mean') or 0.0:.4f}")

        logger.info("=== Running Baseline C: Graph-Free Neural Model ===")
        gf_model = GraphFreeModel(
            node_feat_dim=config["model"]["node_feat_dim"],
            config_feat_dim=config["model"]["config_feat_dim"],
            num_opcodes=config["model"]["num_opcodes"],
            opcode_embed_dim=config["model"]["opcode_embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            dropout=config["model"]["dropout"],
        )
        gf_trainer = Trainer(
            gf_model, train_dataset, val_dev_dataset, preprocessor, config, device=device
        )
        gf_trainer.num_epochs = max(2, config["training"]["num_epochs"] // 2)
        gf_trainer.fit()
        gf_eval = gf_trainer.evaluate(official_val_dataset)
        gf_eval["param_count"] = sum(p.numel() for p in gf_model.parameters())
        experiment_results["models"]["GraphFree"] = gf_eval
        logger.info(f"GraphFree Top-1 Regret: {gf_eval['top1_regret_mean']:.4f}, Spearman: {gf_eval.get('spearman_mean') or 0.0:.4f}")

        logger.info("=== Running Baseline D: GraphSAGE ===")
        sage_model = GraphSAGEModel(
            node_feat_dim=config["model"]["node_feat_dim"],
            config_feat_dim=config["model"]["config_feat_dim"],
            num_opcodes=config["model"]["num_opcodes"],
            opcode_embed_dim=config["model"]["opcode_embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
        sage_trainer = Trainer(
            sage_model, train_dataset, val_dev_dataset, preprocessor, config, device=device
        )
        sage_trainer.num_epochs = max(2, config["training"]["num_epochs"] // 2)
        sage_trainer.fit()
        sage_eval = sage_trainer.evaluate(official_val_dataset)
        sage_eval["param_count"] = sum(p.numel() for p in sage_model.parameters())
        experiment_results["models"]["GraphSAGE"] = sage_eval
        logger.info(f"GraphSAGE Top-1 Regret: {sage_eval['top1_regret_mean']:.4f}, Spearman: {sage_eval.get('spearman_mean') or 0.0:.4f}")

    # 4. Main Model: Configuration-Conditioned GNN
    logger.info("=== Training Main Model: CompilerLens Config-Conditioned GNN ===")
    main_model = ConfigConditionedGNN(
        node_feat_dim=config["model"]["node_feat_dim"],
        config_feat_dim=config["model"]["config_feat_dim"],
        num_opcodes=config["model"]["num_opcodes"],
        opcode_embed_dim=config["model"]["opcode_embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        conditioning=config["model"].get("conditioning", "film"),
        bidirectional=config["model"].get("bidirectional", True),
    )
    main_trainer = Trainer(
        main_model, train_dataset, val_dev_dataset, preprocessor, config, device=device
    )

    # Pilot epoch profiling
    pilot_info = main_trainer.profile_pilot_epoch()
    experiment_results["pilot_profiling"] = pilot_info

    # Train model
    history = main_trainer.fit(resume_from=args.resume)
    experiment_results["training_history"] = history

    # Evaluate on official valid
    main_eval = main_trainer.evaluate(official_val_dataset)
    main_eval["param_count"] = sum(p.numel() for p in main_model.parameters())
    experiment_results["models"]["CompilerLens GNN"] = main_eval
    logger.info(f"CompilerLens GNN Top-1 Regret: {main_eval['top1_regret_mean']:.4f}, Spearman: {main_eval.get('spearman_mean') or 0.0:.4f}")

    # 5. Ablations
    if args.run_ablations:
        logger.info("=== Running Ablation Studies ===")
        ablations_to_run = [
            ("No Graph Edges", {"use_graph_edges": False}),
            ("No Opcode Embeddings", {"use_opcode_embeddings": False}),
            ("Regression-Only Loss", {"loss_mode": "regression_only"}),
            ("2 Message-Passing Layers", {"num_layers": 2}),
        ]

        ablation_regrets = {"Full CompilerLens GNN": main_eval["top1_regret_mean"]}

        for abl_name, abl_kwargs in ablations_to_run:
            logger.info(f"Running Ablation: {abl_name}...")
            abl_model = ConfigConditionedGNN(
                node_feat_dim=config["model"]["node_feat_dim"],
                config_feat_dim=config["model"]["config_feat_dim"],
                num_opcodes=config["model"]["num_opcodes"],
                opcode_embed_dim=config["model"]["opcode_embed_dim"],
                hidden_dim=config["model"]["hidden_dim"],
                num_layers=abl_kwargs.get("num_layers", config["model"]["num_layers"]),
                dropout=config["model"]["dropout"],
                conditioning=config["model"].get("conditioning", "film"),
                bidirectional=config["model"].get("bidirectional", True),
                use_opcode_embeddings=abl_kwargs.get("use_opcode_embeddings", True),
                use_graph_edges=abl_kwargs.get("use_graph_edges", True),
            )
            abl_config = dict(config)
            if "loss_mode" in abl_kwargs:
                abl_config["training"] = dict(config["training"])
                abl_config["training"]["loss_mode"] = abl_kwargs["loss_mode"]

            abl_trainer = Trainer(
                abl_model, train_dataset, val_dev_dataset, preprocessor, abl_config, device=device
            )
            abl_trainer.num_epochs = max(2, config["training"]["num_epochs"] // 2)
            abl_trainer.fit()
            abl_eval = abl_trainer.evaluate(official_val_dataset)
            ablation_regrets[abl_name] = abl_eval["top1_regret_mean"]
            experiment_results["ablations"][abl_name] = abl_eval
            logger.info(f"Ablation '{abl_name}' Top-1 Regret: {abl_eval['top1_regret_mean']:.4f}")

        experiment_results["ablation_summary"] = ablation_regrets

    # 6. Generate predictions map for plotting
    main_preds: Dict[str, np.ndarray] = {}
    for g in valid_graphs:
        scores = predict_graph_candidates(main_model, g, preprocessor, device=device)
        main_preds[g.graph_id] = scores

    # 7. Generate all publication figures (PNG and SVG)
    logger.info("=== Generating Publication Visualizations ===")
    plot_pipeline_diagram(figures_dir / "compilerlens_pipeline")
    plot_model_architecture(figures_dir / "gnn_architecture")
    if valid_graphs:
        plot_computation_graph_sample(valid_graphs[0], figures_dir / "computation_graph_example")
    plot_dataset_distributions(train_dev_graphs, valid_graphs, figures_dir / "dataset_distributions")
    plot_training_curves(history, figures_dir / "training_curves")
    plot_predicted_vs_observed(main_eval, valid_graphs, main_preds, figures_dir / "predicted_vs_observed")
    plot_regret_distribution(main_eval, figures_dir / "regret_distribution")
    if args.run_baselines:
        plot_baseline_comparison(experiment_results["models"], figures_dir / "baseline_comparison")
        plot_topk_regret_curve(experiment_results["models"], figures_dir / "topk_regret_curve")
    if args.run_ablations:
        plot_ablation_comparison(experiment_results["ablation_summary"], figures_dir / "ablation_study")
    plot_case_studies(main_eval, valid_graphs, main_preds, figures_dir / "case_studies")

    # 8. Export test predictions if test graphs available
    if test_files:
        logger.info("Exporting unlabeled test split predictions...")
        test_graphs = [load_npz_graph(p, split="test", validate=False) for p in test_files]
        test_csv_path = output_dir / "test_predictions.csv"
        export_test_predictions(main_model, test_graphs, preprocessor, test_csv_path, device=device)
        logger.info(f"Exported test predictions to {test_csv_path}")

    # 9. Save final experiment summary JSON
    results_json_path = output_dir / "experiment_results.json"
    save_evaluation_report(experiment_results, results_json_path)
    logger.info(f"Experiment completed successfully! Results saved to {results_json_path}")


if __name__ == "__main__":
    main()
