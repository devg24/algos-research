"""
Dataset loading utilities for ANN benchmarks.

Supports:
- SIFT-1M (128-dim)
- GIST-1M (960-dim)  
- Deep1B (96-dim)
"""

import numpy as np
import os
from typing import Tuple, Optional


class DatasetLoader:
    """
    Loader for standard ANN benchmark datasets.
    
    Expected directory structure:
        data/
        ├── sift/
        │   ├── sift_base.fvecs
        │   ├── sift_query.fvecs
        │   └── sift_groundtruth.ivecs
        ├── gist/
        │   ├── gist_base.fvecs
        │   ├── gist_query.fvecs
        │   └── gist_groundtruth.ivecs
        └── deep1b/
            ├── deep1b_base.fvecs (or subset)
            ├── deep1b_query.fvecs
            └── deep1b_groundtruth.ivecs
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Root directory containing datasets
        """
        self.data_dir = data_dir
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory '{data_dir}' does not exist!")
    
    @staticmethod
    def _read_fvecs(filepath: str, count: Optional[int] = None) -> np.ndarray:
        """
        Read .fvecs file format (used by SIFT, GIST, Deep1B).
        
        Format: [d, v1, v2, ..., vd] repeated for each vector
        where d is dimension (int32) and vi are float32 values
        
        Args:
            filepath: Path to .fvecs file
            count: Number of vectors to read (None = all)
        
        Returns:
            Array of shape (n, d)
        """
        with open(filepath, 'rb') as f:
            # Read dimension from first vector
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(0)
            
            # Calculate number of vectors (use Python int to avoid overflow)
            filesize = os.path.getsize(filepath)
            vector_size = 4 + int(d) * 4  # 4 bytes for dim + d * 4 bytes for floats
            n_vectors = int(filesize // vector_size)
            
            if count is not None:
                n_vectors = min(n_vectors, count)
            
            # Read vectors
            data = np.zeros((n_vectors, int(d)), dtype=np.float32)
            
            for i in range(n_vectors):
                dim = np.fromfile(f, dtype=np.int32, count=1)[0]
                assert dim == d, f"Dimension mismatch at vector {i}"
                data[i] = np.fromfile(f, dtype=np.float32, count=int(d))
            
            return data
    
    @staticmethod
    def _read_fbin(filepath: str, count: Optional[int] = None) -> np.ndarray:
        """
        Read .fbin file format (Deep1B format).
        
        Format: [n, d, vector_data...]
        where n = number of vectors (uint32)
              d = dimension (uint32)
              vector_data = n*d float32 values
        
        Args:
            filepath: Path to .fbin file
            count: Number of vectors to read (None = all)
        
        Returns:
            Array of shape (n, d)
        """
        with open(filepath, 'rb') as f:
            # Read header
            n_total = np.fromfile(f, dtype=np.uint32, count=1)[0]
            d = np.fromfile(f, dtype=np.uint32, count=1)[0]
            
            # Convert to Python int to avoid overflow
            n_total = int(n_total)
            d = int(d)
            
            # Debug output
            print(f"  [_read_fbin] File header: n={n_total:,}, d={d}")
            
            # Determine how many vectors to actually read
            n_read = n_total if count is None else min(n_total, count)
            
            # Read exactly n_read vectors
            data = np.fromfile(f, dtype=np.float32, count=n_read * d)
            
            # Verify we got the expected amount of data
            expected_size = n_read * d
            actual_size = data.size
            print(f"  [_read_fbin] Read {actual_size:,} values (expected {expected_size:,})")
            
            if actual_size != expected_size:
                raise ValueError(
                    f"Expected to read {expected_size} values but got {actual_size}. "
                    f"File may be corrupted or truncated."
                )
            
            # Reshape to (n_read, d)
            data = data.reshape(n_read, d)
            print(f"  [_read_fbin] Final shape: {data.shape}")
            
            return data
    
    @staticmethod
    def _read_ibin(filepath: str, count: Optional[int] = None) -> np.ndarray:
        """
        Read .ibin file format (Deep1B ground truth format).
        
        Format: [n, k, indices...]
        where n = number of queries (uint32)
              k = number of neighbors (uint32)
              indices = n*k uint32 values
        
        Args:
            filepath: Path to .ibin file
            count: Number of queries to read (None = all)
        
        Returns:
            Array of shape (n_queries, k)
        """
        with open(filepath, 'rb') as f:
            # Read header
            n = np.fromfile(f, dtype=np.uint32, count=1)[0]
            k = np.fromfile(f, dtype=np.uint32, count=1)[0]
            
            # Convert to Python int to avoid overflow
            n = int(n)
            k = int(k)
            
            if count is not None:
                n = min(n, count)
            
            # Read indices
            indices = np.fromfile(f, dtype=np.uint32, count=n*k)
            indices = indices.reshape(n, k)
            
            return indices
    
    @staticmethod
    def _read_ivecs(filepath: str, count: Optional[int] = None) -> np.ndarray:
        """
        Read .ivecs file format (used for ground truth).
        
        Format: [k, idx1, idx2, ..., idxk] repeated for each query
        where k is number of neighbors (int32) and idxi are int32 indices
        
        Args:
            filepath: Path to .ivecs file
            count: Number of queries to read (None = all)
        
        Returns:
            Array of shape (n_queries, k)
        """
        with open(filepath, 'rb') as f:
            # Read k from first entry
            k = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(0)
            
            # Calculate number of queries (use Python int to avoid overflow)
            filesize = os.path.getsize(filepath)
            entry_size = 4 + int(k) * 4  # 4 bytes for k + k * 4 bytes for indices
            n_queries = int(filesize // entry_size)
            
            if count is not None:
                n_queries = min(n_queries, count)
            
            # Read indices
            data = np.zeros((n_queries, int(k)), dtype=np.int32)
            
            for i in range(n_queries):
                num_neighbors = np.fromfile(f, dtype=np.int32, count=1)[0]
                assert num_neighbors == k, f"Neighbor count mismatch at query {i}"
                data[i] = np.fromfile(f, dtype=np.int32, count=int(k))
            
            return data
    
    def load_sift(self, 
                  n_train: Optional[int] = None,
                  n_test: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load SIFT-1M dataset.
        
        Args:
            n_train: Number of training vectors to load (None = all, max 1M)
            n_test: Number of test queries to load (None = all, max 10K)
        
        Returns:
            (X_train, X_test) tuple of arrays
        """
        print("Loading SIFT-1M dataset...")
        base_path = os.path.join(self.data_dir, "sift", "sift_base.fvecs")
        query_path = os.path.join(self.data_dir, "sift", "sift_query.fvecs")
        
        X_train = self._read_fvecs(base_path, count=n_train)
        X_test = self._read_fvecs(query_path, count=n_test)
        
        print(f"  Training: {X_train.shape} (dtype: {X_train.dtype})")
        print(f"  Test: {X_test.shape} (dtype: {X_test.dtype})")
        
        return X_train, X_test
    
    def load_gist(self,
                  n_train: Optional[int] = None,
                  n_test: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load GIST-1M dataset.
        
        Args:
            n_train: Number of training vectors to load (None = all, max 1M)
            n_test: Number of test queries to load (None = all, max 1K)
        
        Returns:
            (X_train, X_test) tuple of arrays
        """
        print("Loading GIST-1M dataset...")
        base_path = os.path.join(self.data_dir, "gist", "gist_base.fvecs")
        query_path = os.path.join(self.data_dir, "gist", "gist_query.fvecs")
        
        X_train = self._read_fvecs(base_path, count=n_train)
        X_test = self._read_fvecs(query_path, count=n_test)
        
        print(f"  Training: {X_train.shape} (dtype: {X_train.dtype})")
        print(f"  Test: {X_test.shape} (dtype: {X_test.dtype})")
        
        return X_train, X_test
    
    def load_deep1b(self,
                    n_train: Optional[int] = None,
                    n_test: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Deep1B dataset (10M subset with .fbin format).
        
        Args:
            n_train: Number of training vectors to load (None = all, max 10M for subset)
            n_test: Number of test queries to load (None = all, max 10K)
        
        Returns:
            (X_train, X_test) tuple of arrays
        """
        print("Loading Deep1B-10M dataset...")
        base_path = os.path.join(self.data_dir, "deep1b", "base.10M.fbin")
        query_path = os.path.join(self.data_dir, "deep1b", "query.public.10K.fbin")
        
        X_train = self._read_fbin(base_path, count=n_train)
        X_test = self._read_fbin(query_path, count=n_test)
        
        print(f"  Training: {X_train.shape} (dtype: {X_train.dtype})")
        print(f"  Test: {X_test.shape} (dtype: {X_test.dtype})")
        
        return X_train, X_test
    
    def load_dataset(self, 
                     name: str,
                     n_train: Optional[int] = None,
                     n_test: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset by name.
        
        Args:
            name: Dataset name ('sift', 'gist', or 'deep1b')
            n_train: Number of training vectors to load
            n_test: Number of test queries to load
        
        Returns:
            (X_train, X_test) tuple of arrays
        """
        name = name.lower()
        
        if name == 'sift':
            return self.load_sift(n_train, n_test)
        elif name == 'gist':
            return self.load_gist(n_train, n_test)
        elif name in ['deep1b', 'deep']:
            return self.load_deep1b(n_train, n_test)
        else:
            raise ValueError(f"Unknown dataset: {name}. Use 'sift', 'gist', or 'deep1b'")


def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors (useful for cosine similarity).
    
    Args:
        X: Array of shape (n, d)
    
    Returns:
        Normalized array of shape (n, d)
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return X / norms


def compute_pairwise_distances(X: np.ndarray, Y: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """
    Compute pairwise Euclidean distances in batches (memory efficient).
    
    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)
        batch_size: Process in batches to save memory
    
    Returns:
        Distance matrix of shape (n, m)
    """
    n, m = X.shape[0], Y.shape[0]
    distances = np.zeros((n, m), dtype=np.float32)
    
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            
            # Compute batch distances
            diff = X[i:end_i, None, :] - Y[None, j:end_j, :]
            distances[i:end_i, j:end_j] = np.sqrt(np.sum(diff**2, axis=2))
    
    return distances


# Quick dataset info
DATASET_INFO = {
    'sift': {
        'name': 'SIFT-1M',
        'dimension': 128,
        'train_size': 1_000_000,
        'test_size': 10_000,
        'description': 'SIFT image descriptors'
    },
    'gist': {
        'name': 'GIST-1M',
        'dimension': 960,
        'train_size': 1_000_000,
        'test_size': 1_000,
        'description': 'GIST global image features'
    },
    'deep1b': {
        'name': 'Deep1B-10M',
        'dimension': 96,
        'train_size': 10_000_000,  # 10 million (subset)
        'test_size': 10_000,
        'description': 'Deep learning image features (10M subset)'
    }
}


def print_dataset_info(name: str):
    """Print information about a dataset."""
    info = DATASET_INFO.get(name.lower())
    if info:
        print(f"\n{info['name']} Dataset Info:")
        print(f"  Description: {info['description']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Training size: {info['train_size']:,}")
        print(f"  Test size: {info['test_size']:,}\n")
    else:
        print(f"Unknown dataset: {name}")