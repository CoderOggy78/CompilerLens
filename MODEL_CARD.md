# Model Card for CompilerLens

## Model Details
- **Model Name**: CompilerLens Configuration-Conditioned GNN
- **Model Architecture**: Direction-Aware Graph Neural Network with Feature-wise Configuration Modulation (FiLM / Concatenation)
- **Primary Objective**: Offline learned compiler cost modeling and configuration ranking
- **Version**: 0.1.0
- **License**: Apache-2.0
- **Date**: October 2026

## Intended Use
- **Primary Intended Use**: Given an XLA High-Level Optimizer (HLO) computation graph and a discrete set of candidate tile configurations, rank the configurations from fastest to slowest and recommend top-$k$ candidates for autotuning.
- **Primary Research Question**: Does a configuration-conditioned graph neural network select faster compiler configurations for unseen program graphs than models that ignore graph connectivity?
- **Out-of-Scope and Prohibited Uses**:
  - CompilerLens is **not** an end-to-end compiler; it does not generate binary machine code or verify operational semantics.
  - CompilerLens does **not** accelerate live compiler passes directly unless integrated into an autotuning feedback loop with candidate generation.
  - The model is trained exclusively on TPU v3 execution measurements for XLA fused tile configurations. It must **not** be assumed to generalize to CPU, NVIDIA/AMD GPU, or mobile accelerator runtimes without retraining or fine-tuning.

## Model Architecture Overview
1. **Node Opcode Embedding**: Look-up table mapping discrete XLA opcodes ($0 \dots 127$) to a $D$-dimensional continuous space ($D=128$).
2. **Numeric Feature Projection**: Multi-Layer Perceptron (MLP) projecting $140$-dimensional continuous node attributes into $D$ dimensions, normalized via LayerNorm.
3. **Configuration Encoder**: MLP mapping $24$-dimensional tile configuration vectors into $D$ dimensions.
4. **Configuration Conditioning**: Injects configuration context into node representations before message passing via feature-wise linear modulation (FiLM) or concatenation projection.
5. **Direction-Aware Message Passing**: 3 layers of bidirectional message passing separating forward dataflow edges from reverse consumer-to-producer edges, with residual connections and layer normalization.
6. **Graph Readout & Scoring**: Mean + Max global graph pooling, concatenated with the configuration embedding, passed into an MLP regressor predicting scalar $\hat{y} = \log(\text{normalized\_runtime})$. Lower score indicates faster predicted runtime.

## Training Data & Target
- **Dataset**: TpuGraphs Tile XLA dataset (`npz_tile_xla`).
- **Target**: Scalar log normalized runtime $\log(\text{config\_runtime} / \text{config\_runtime\_normalizers})$.
- **Objective Function**: Joint regression and pairwise ranking loss:
  $$\mathcal{L} = \mathcal{L}_{\text{Huber}}(\hat{y}, y) + \lambda_{\text{rank}} \mathcal{L}_{\text{pairwise}}(\hat{y}_i, \hat{y}_j)$$
  where $\mathcal{L}_{\text{pairwise}} = \log(1 + \exp(\hat{y}_i - \hat{y}_j))$ for pairs where candidate $i$ is faster than candidate $j$ ($y_i < y_j$).

## Evaluation Metrics & Protocols
- **Graph-Disjoint Splits**: Models are evaluated strictly on graphs unseen during training.
- **Ranking Metrics**:
  - **Top-1 Regret**: $\frac{r(\text{predicted best})}{r(c^*)} - 1$, where $c^*$ is the fastest configuration in the candidate set.
  - **Best-of-top-$k$ Regret** ($k \in \{1, 5, 10\}$): $\min_{c \in \text{top-}k} \frac{r(c)}{r(c^*)} - 1$.
  - **Spearman Rank Correlation ($\rho$)**: Per-graph rank correlation between predicted scores and observed runtimes.
- **Regression Metrics**: Log runtime MAE and RMSE.
- **Statistical Significance**: 95% bootstrap confidence intervals computed over graphs (1,000 resamples).

## Limitations and Ethical Considerations
- **Hardware Specificity**: Predictions reflect physical TPU v3 timing characteristics and thermal/frequency steady states.
- **Candidate Quality Ceiling**: The model can only select among the supplied candidate configurations; it cannot synthesize novel configurations outside the candidate generator's space.
- **Evaluation Candidate Set Definition**: The "oracle" baseline $c^*$ is strictly the best configuration among the evaluated candidates in the dataset, not a globally optimal configuration over all possible parameter spaces.
