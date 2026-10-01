"""Random ranking baseline evaluated across repeated seeded draws."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from compilerlens.data import GraphData


class RandomRankingBaseline:
    """Predicts uniformly random scores to establish an empirical lower-bound baseline."""

    def __init__(self, num_draws: int = 50, seed: int = 42) -> None:
        self.num_draws = num_draws
        self.seed = seed
        self.param_count = 0

    def predict_graph(self, graph: GraphData, draw_seed: int) -> np.ndarray:
        """Generate random ranking scores for a graph's candidate configurations.

        Returns:
            np.ndarray of shape (C,) representing pseudo-scores (lower is predicted faster).
        """
        rng = np.random.RandomState(draw_seed)
        return rng.rand(graph.num_configs).astype(np.float32)

    def evaluate_all(
        self, graphs: Sequence[GraphData]
    ) -> List[Dict[str, np.ndarray]]:
        """Run repeated evaluations across all graphs and return prediction sets per draw."""
        draw_predictions = []
        for draw in range(self.num_draws):
            draw_seed = self.seed + draw * 1000
            preds = {}
            for g_idx, g in enumerate(graphs):
                g_seed = draw_seed + g_idx
                preds[g.graph_id] = self.predict_graph(g, g_seed)
            draw_predictions.append(preds)
        return draw_predictions
