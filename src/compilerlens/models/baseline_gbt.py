"""Gradient-boosted decision tree baseline using graph summaries and configuration features."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from compilerlens.data import GraphData
from compilerlens.features import FeaturePreprocessor, extract_graph_summary_features


class GBTBaseline:
    """Gradient-Boosted Tree baseline predicting compiler configuration runtimes.

    Features: Concatenation of fixed-length graph summary features and candidate configuration features.
    Target: Log normalized runtime.
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            loss="squared_error",
        )
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.fit_time_seconds: float = 0.0
        self.param_count: int = max_iter * max_leaf_nodes

    def _build_dataset(
        self,
        graphs: Sequence[GraphData],
        preprocessor: Optional[FeaturePreprocessor] = None,
        max_configs_per_graph: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct tabular feature matrix and target vector from graph collection."""
        x_list = []
        y_list = []

        for g_idx, g in enumerate(graphs):
            if g.log_normalized_runtimes is None:
                continue

            summary = extract_graph_summary_features(g, preprocessor=preprocessor)  # (411,)
            configs = g.config_feat  # (C, 24)
            targets = g.log_normalized_runtimes  # (C,)

            if preprocessor is not None and preprocessor.is_fit:
                configs = preprocessor.transform_config_feat(configs)

            num_c = configs.shape[0]
            if max_configs_per_graph is not None and num_c > max_configs_per_graph:
                rng = np.random.RandomState(self.seed + g_idx)
                sel_idx = rng.choice(num_c, size=max_configs_per_graph, replace=False)
                configs = configs[sel_idx]
                targets = targets[sel_idx]
                num_c = len(sel_idx)

            # Broadcast summary across all configurations of this graph
            summary_repeated = np.repeat(summary[np.newaxis, :], num_c, axis=0)  # (num_c, 411)
            x_g = np.concatenate([summary_repeated, configs], axis=1)  # (num_c, 435)

            x_list.append(x_g)
            y_list.append(targets)

        if not x_list:
            raise ValueError("No labeled configurations available to build GBT dataset.")

        X = np.concatenate(x_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0).astype(np.float32)
        return X, y

    def fit(
        self,
        training_graphs: Sequence[GraphData],
        preprocessor: Optional[FeaturePreprocessor] = None,
        max_configs_per_graph: Optional[int] = 64,
    ) -> GBTBaseline:
        """Fit GBT regressor on training graphs."""
        self.preprocessor = preprocessor
        start_time = time.time()
        X_train, y_train = self._build_dataset(
            training_graphs,
            preprocessor=preprocessor,
            max_configs_per_graph=max_configs_per_graph,
        )
        self.model.fit(X_train, y_train)
        self.fit_time_seconds = time.time() - start_time
        return self

    def predict_graph(self, graph: GraphData) -> np.ndarray:
        """Predict log normalized runtimes for all candidate configurations of a single graph."""
        summary = extract_graph_summary_features(graph, preprocessor=self.preprocessor)
        configs = graph.config_feat

        if self.preprocessor is not None and self.preprocessor.is_fit:
            configs = self.preprocessor.transform_config_feat(configs)

        num_c = configs.shape[0]
        summary_repeated = np.repeat(summary[np.newaxis, :], num_c, axis=0)
        X_g = np.concatenate([summary_repeated, configs], axis=1).astype(np.float32)

        return self.model.predict(X_g).astype(np.float32)

    def predict_all(self, graphs: Sequence[GraphData]) -> Dict[str, np.ndarray]:
        """Predict candidates for all graphs in collection."""
        preds = {}
        for g in graphs:
            preds[g.graph_id] = self.predict_graph(g)
        return preds
