"""
Evaluator class for ANN algorithm comparison.
Provides consistent metrics across all experiments.
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Any
from sklearn.neighbors import NearestNeighbors


class ANNEvaluator:
    """
    Evaluator for Approximate Nearest Neighbor algorithms.
    
    Metrics computed:
    - Recall@K: Fraction of true nearest neighbors retrieved
    - Query time: Average time per query (ms)
    - Build time: Time to construct the index
    - Memory footprint: Index size in memory
    - Index size: Size on disk (if applicable)
    """
    
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, k: int = 10):
        """
        Initialize evaluator.
        
        Args:
            X_train: Training data (n_train, d)
            X_test: Test queries (n_test, d)
            k: Number of nearest neighbors to retrieve
        """
        self.X_train = X_train
        self.X_test = X_test
        self.k = k
        self.ground_truth = None
        self.n_train, self.d = X_train.shape
        self.n_test = X_test.shape[0]
        
        print(f"Evaluator initialized:")
        print(f"  Training points: {self.n_train:,}")
        print(f"  Test queries: {self.n_test:,}")
        print(f"  Dimensions: {self.d}")
        print(f"  k (neighbors): {self.k}")
    
    def compute_ground_truth(self, force_recompute: bool = False):
        """
        Compute exact k-NN using brute force (ground truth for recall).
        Uses sklearn's efficient ball tree implementation.
        
        Args:
            force_recompute: If True, recompute even if ground truth exists
        """
        if self.ground_truth is not None and not force_recompute:
            print("Ground truth already computed. Use force_recompute=True to recompute.")
            return
        
        print(f"Computing ground truth k-NN (k={self.k}) using brute force...")
        start = time.time()
        
        # Use ball_tree for efficiency with high-dimensional data
        knn = NearestNeighbors(
            n_neighbors=self.k,
            algorithm='brute',  # Exact search
            metric='euclidean',
            n_jobs=-1  # Use all CPU cores
        )
        knn.fit(self.X_train)
        
        _, self.ground_truth = knn.kneighbors(self.X_test)
        
        elapsed = time.time() - start
        print(f"Ground truth computed in {elapsed:.2f}s")
        print(f"Ground truth shape: {self.ground_truth.shape}")
    
    def evaluate(self, 
                 index_builder: callable,
                 query_func: callable,
                 method_name: str = "Unknown",
                 **build_params) -> Dict[str, Any]:
        """
        Evaluate an ANN algorithm.
        
        Args:
            index_builder: Function that builds the index
                Signature: index_builder(X_train, **build_params) -> index
            query_func: Function that queries the index
                Signature: query_func(index, X_test, k) -> (indices, distances)
                Returns: (n_test, k) arrays of neighbor indices and distances
            method_name: Name of the method being evaluated
            **build_params: Additional parameters for index building
        
        Returns:
            Dictionary with all metrics
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not computed. Call compute_ground_truth() first.")
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")
        
        results = {
            'method': method_name,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'd': self.d,
            'k': self.k,
            'build_params': build_params
        }
        
        # 1. Build time
        print(f"Building index...")
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        build_start = time.time()
        index = index_builder(self.X_train, **build_params)
        build_time = time.time() - build_start
        
        mem_after = process.memory_info().rss / 1024**2  # MB
        
        results['build_time_s'] = build_time
        print(f"  Build time: {build_time:.2f}s")
        
        # 2. Memory footprint
        memory_used = mem_after - mem_before
        results['memory_mb'] = memory_used
        print(f"  Memory used: {memory_used:.2f} MB")
        
        # 3. Query time and retrieve results
        print(f"Querying {self.n_test:,} test points...")
        query_times = []
        all_indices = []
        
        # Query in batches for timing accuracy
        batch_size = min(100, self.n_test)
        n_batches = (self.n_test + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.n_test)
            batch = self.X_test[start_idx:end_idx]
            
            query_start = time.time()
            indices, _ = query_func(index, batch, self.k)
            query_time = time.time() - query_start
            
            query_times.append(query_time / len(batch))  # Per query
            all_indices.append(indices)
        
        predicted_neighbors = np.vstack(all_indices)
        avg_query_time = np.mean(query_times) * 1000  # Convert to ms
        
        results['avg_query_time_ms'] = avg_query_time
        print(f"  Avg query time: {avg_query_time:.3f} ms")
        
        # 4. Recall@K
        recall = self._compute_recall(predicted_neighbors, self.ground_truth)
        results['recall@k'] = recall
        print(f"  Recall@{self.k}: {recall:.4f} ({recall*100:.2f}%)")
        
        # 5. Per-query recall distribution
        per_query_recall = np.array([
            len(np.intersect1d(pred, true)) / self.k 
            for pred, true in zip(predicted_neighbors, self.ground_truth)
        ])
        results['recall_mean'] = np.mean(per_query_recall)
        results['recall_std'] = np.std(per_query_recall)
        results['recall_min'] = np.min(per_query_recall)
        results['recall_max'] = np.max(per_query_recall)
        
        print(f"  Recall stats: mean={results['recall_mean']:.4f}, "
              f"std={results['recall_std']:.4f}, "
              f"min={results['recall_min']:.4f}, "
              f"max={results['recall_max']:.4f}")
        
        print(f"{'='*60}\n")
        
        return results
    
    def _compute_recall(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute Recall@K.
        
        Args:
            predicted: (n_queries, k) array of predicted neighbor indices
            ground_truth: (n_queries, k) array of true neighbor indices
        
        Returns:
            Recall@K averaged over all queries
        """
        assert predicted.shape == ground_truth.shape
        
        recalls = []
        for pred, true in zip(predicted, ground_truth):
            # Count how many predicted neighbors are in ground truth
            intersection = len(np.intersect1d(pred, true))
            recalls.append(intersection / self.k)
        
        return np.mean(recalls)
    
    def compare_methods(self, results_list: List[Dict[str, Any]]) -> None:
        """
        Print comparison table of multiple methods.
        
        Args:
            results_list: List of result dictionaries from evaluate()
        """
        print(f"\n{'='*80}")
        print("COMPARISON OF METHODS")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'Recall@K':<12} {'Query Time':<15} {'Build Time':<15} {'Memory':<12}")
        print(f"{'':<20} {'':<12} {'(ms)':<15} {'(s)':<15} {'(MB)':<12}")
        print(f"{'-'*80}")
        
        for result in results_list:
            print(f"{result['method']:<20} "
                  f"{result['recall@k']:<12.4f} "
                  f"{result['avg_query_time_ms']:<15.3f} "
                  f"{result['build_time_s']:<15.2f} "
                  f"{result['memory_mb']:<12.2f}")
        
        print(f"{'='*80}\n")
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save results to JSON file.
        
        Args:
            results: Result dictionary from evaluate()
            filepath: Path to save JSON file
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.integer):
                results_serializable[key] = int(value)
            elif isinstance(value, np.floating):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {filepath}")