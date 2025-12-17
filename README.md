# JL+LSH Experimental Setup

This repository contains the experimental infrastructure for evaluating Johnson-Lindenstrauss (JL) projection combined with Locality-Sensitive Hashing (LSH) for Approximate Nearest Neighbor (ANN) search.

## Project Structure

```
project/
├── data/                           # Dataset directory (not included)
│   ├── sift/
│   │   ├── sift_base.fvecs
│   │   ├── sift_query.fvecs
│   │   └── sift_groundtruth.ivecs
│   ├── gist/
│   │   ├── gist_base.fvecs
│   │   ├── gist_query.fvecs
│   │   └── gist_groundtruth.ivecs
│   └── deep1b/
│       ├── base.10M.fbin               # 10M subset
│       ├── query.public.10k.fbin
│       └── groundtruth.public.10k.ibin
│
├── evaluator.py                    # ANNEvaluator class for consistent metrics
├── datasets.py                     # Dataset loading utilities
├── 01_baseline_knn.ipynb          # Baseline experiments (this file)
│
├── results/                        # Experiment results (auto-created)
│   ├── baseline_knn_sift.json
│   ├── baseline_knn_gist.json
│   └── baseline_knn_summary.json
│
└── README.md                       # This file
```

## Installation

### Requirements

```bash
pip install numpy scikit-learn matplotlib seaborn psutil jupyter
```

### Optional (for future notebooks)
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install hnswlib
```

## Datasets

Download the benchmark datasets from:

- **SIFT-1M**: http://corpus-texmex.irisa.fr/
- **GIST-1M**: http://corpus-texmex.irisa.fr/
- **Deep1B**: http://sites.skoltech.ru/compvision/noimi/

Place the `.fvecs` and `.ivecs` files in the corresponding `data/` subdirectories.

**Note**: Deep1B is very large (1 billion vectors). Use the 10M subset (`deep10M_base.fvecs`) for initial experiments.

## Usage

### 1. Baseline KNN Experiments

The baseline notebook establishes exact k-NN performance:

```bash
jupyter notebook 01_baseline_knn.ipynb
```

**What it does:**
- Loads SIFT, GIST, and Deep1B datasets
- Computes ground truth using brute-force k-NN
- Measures query time, build time, memory usage, and recall
- Saves results to `results/` directory

### 2. Using the Evaluator Class

The `ANNEvaluator` class provides a consistent interface for evaluating any ANN algorithm:

```python
from evaluator import ANNEvaluator
from datasets import DatasetLoader

# Load dataset
loader = DatasetLoader(data_dir="data")
X_train, X_test = loader.load_sift(n_train=100000, n_test=1000)

# Initialize evaluator
evaluator = ANNEvaluator(X_train, X_test, k=10)

# Compute ground truth
evaluator.compute_ground_truth()

# Define your algorithm
def my_index_builder(X_train, **params):
    # Build your index here
    return index

def my_query_func(index, X_test, k):
    # Query your index here
    return indices, distances

# Evaluate
results = evaluator.evaluate(
    index_builder=my_index_builder,
    query_func=my_query_func,
    method_name="MyMethod",
    param1=value1,
    param2=value2
)

# Save results
evaluator.save_results(results, "results/my_method.json")
```

### 3. Metrics Computed

The evaluator automatically computes:

- **Recall@K**: Fraction of true k-nearest neighbors retrieved
- **Query time**: Average milliseconds per query
- **Build time**: Index construction time in seconds
- **Memory**: Memory footprint in MB
- **Recall statistics**: Mean, std, min, max across queries

## Experimental Plan

### Week 1-2: Baseline and JL Projection
- [x] Baseline KNN (this notebook)
- [ ] Implement JL projection (Achlioptas sparse random matrices)
- [ ] Test JL with various target dimensions k ∈ {50, 75, 100, 150, 200, 300}

### Week 3: LSH Implementation
- [ ] Implement E2LSH (p-stable hashing)
- [ ] Tune hyperparameters (K, L, w)
- [ ] Compare with FAISS and HNSW baselines

### Week 4: JL+LSH Composition
- [ ] Combine JL projection with LSH indexing
- [ ] Analyze collision probability preservation
- [ ] Generate space-time-accuracy Pareto curves
- [ ] Write final report

## Key Results to Track

For each method, track:

1. **Space-Time Trade-off**: Index size vs query time
2. **Accuracy-Time Trade-off**: Recall vs query time
3. **Dimensionality Scaling**: How performance changes with d
4. **Optimal k**: Best target dimension for JL projection

## Example: Comparing Multiple Methods

```python
# Evaluate multiple methods
methods = [
    ("KNN-Brute", knn_builder, knn_query),
    ("JL+KNN k=100", jl_knn_builder, jl_knn_query),
    ("E2LSH", lsh_builder, lsh_query),
]

all_results = []
for name, builder, query in methods:
    results = evaluator.evaluate(builder, query, method_name=name)
    all_results.append(results)

# Compare
evaluator.compare_methods(all_results)
```

## Tips for Next Notebooks

### 02_jl_projection.ipynb (Next)
- Implement Achlioptas sparse random projection
- Test multiple target dimensions: k ∈ {50, 75, 100, 150, 200, 300}
- Measure recall degradation vs dimensionality reduction
- Plot space savings vs accuracy loss

### 03_lsh.ipynb
- Implement E2LSH hash family
- Test amplification parameters (K, L)
- Compare with FAISS (Product Quantization)
- Compare with HNSW (graph-based)

### 04_jl_lsh.ipynb (Main Contribution)
- Combine JL + LSH
- Analyze collision probability preservation (Lipschitz analysis)
- Generate final Pareto curves
- Derive practical guidelines

## Troubleshooting

### Dataset Loading Issues
- Ensure `.fvecs` and `.ivecs` files are in correct directories
- Check file permissions
- For Deep1B, use 10M subset first

### Memory Issues
- Reduce `n_train` and `n_test` in configuration
- Process queries in smaller batches
- Use swap space for very large datasets

### Slow Performance
- Use `n_jobs=-1` to enable parallel processing
- Reduce test set size for faster iteration
- Start with SIFT (smallest dataset)

## Citation

If you use this code, please cite:

```bibtex
@article{jl-lsh-composition,
  title={Johnson-Lindenstrauss + LSH Composition via Lipschitz Analysis},
  author={Goyal, Dev and Castillo, Alessandro and Khajanchi, Anoushka},
  year={2025}
}
```

## References

- Johnson & Lindenstrauss (1984): Original JL Lemma
- Achlioptas (2001): Sparse random projections
- Indyk & Motwani (1998): LSH framework
- Datar et al. (2004): E2LSH family
- Andoni & Indyk (2008): LSH survey

## Contact

Alessandro Castillo: agc2166@columbia.edu
Anoushka Khajanchi: 
Dev Goyal: dg3513@columbia.edu

---
