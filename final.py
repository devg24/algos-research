import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors
import sys
import os

# Add parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DatasetLoader
# Import the E2LSH class you defined (assuming it's in src/lsh.py or similar)
# If it's in the same file, you can paste the class here. 
# For this script, I will include the E2LSH class definition to make it self-contained.

# ==========================================
# 1. THE E2LSH CLASS (Corrected for Euclidean)
# ==========================================
from collections import defaultdict

class E2LSH:
    def __init__(self, num_tables, hash_bits, dim, r, random_state=42):
        self.num_tables = num_tables
        self.hash_bits = hash_bits
        self.dim = dim
        self.r = r
        self.rng = np.random.RandomState(random_state)
        self.projections = [self.rng.randn(hash_bits, dim) for _ in range(num_tables)]
        self.offsets = [self.rng.uniform(0, r, size=(hash_bits, 1)) for _ in range(num_tables)]
        self.tables = None
        self.X_train = None

    def _compute_hash(self, X, table_idx):
        if X.ndim == 1: X = X.reshape(1, -1)
        proj = X @ self.projections[table_idx].T
        return np.floor((proj + self.offsets[table_idx].T) / self.r).astype(np.int32)

    def fit(self, X_train):
        self.X_train = X_train
        self.tables = [defaultdict(list) for _ in range(self.num_tables)]
        n = len(X_train)
        print(f"  Building Index: {n} points...")
        for table_idx in range(self.num_tables):
            # optimization: process in one batch for speed in this script
            hashes = self._compute_hash(X_train, table_idx)
            for i, h_row in enumerate(hashes):
                self.tables[table_idx][tuple(h_row)].append(i)
        return self

    def query(self, X_test, k=1):
        n_queries = len(X_test)
        found_flags = np.zeros(n_queries, dtype=bool)
        
        # We only care if we found the neighbor (Recall@1)
        # We need the ground truth indices passed in to verify success strictly
        return found_flags # Placeholder, logic implemented in main loop

# ==========================================
# 2. THEORETICAL HARDNESS (Track A)
# ==========================================
def f2(t):
    return np.sqrt(2 / np.pi) * np.exp(-t**2 / 2)

def calculate_hardness(c, r):
    return calculate_relative_hardness(c, r)

def p_collision_func(c, r):
    # Standard E2LSH probability formula (integral 0 to r)
    def integrand(t):
        return (1/c) * f2(t/c) * (1 - t/r)
    val, _ = quad(integrand, 0, r)
    return val

def calculate_relative_hardness(c, r):
    if c == 0: return 0.0
    
    # 1. Calculate P(c)
    p_val = p_collision_func(c, r)
    if p_val < 1e-9: return 100.0 # Cap max hardness for impossible points
    
    # 2. Calculate Derivative (same as before)
    def integrand_deriv(t):
        term1 = (t**2 / c**2) - 1
        term2 = 1 - (t / r)
        term3 = f2(t / c)
        return (1/c**2) * term1 * term2 * term3
    deriv, _ = quad(integrand_deriv, 0, r)
    
    # 3. Relative Hardness = (|p'| * c) / p
    return (abs(deriv) * c) / p_val

# ==========================================
# 3. MAIN EXPERIMENT
# ==========================================
def run_hardness_experiment():
    # --- CONFIGURATION ---
    DATA_DIR = "data"
    N_TEST = 1000  # Number of queries to plot (1000 is enough for a trend)
    
    # THE "SWEET SPOT" PARAMETERS (tuned for d=32)
    TARGET_DIM = 32
    LSH_R = 250.0
    LSH_L = 50     # Num Tables
    LSH_K = 4      # Hash Bits (Adjust to 4 or 8 if recall is 100% or 0%)
    
    print("="*60)
    print("RUNNING LIPSCHITZ HARDNESS VERIFICATION")
    print(f"Target Dim: {TARGET_DIM}")
    print(f"LSH Params: L={LSH_L}, k={LSH_K}, r={LSH_R}")
    print("="*60)

    # 1. Load Data
    loader = DatasetLoader(data_dir=DATA_DIR)
    try:
        X_train_orig, X_test_orig = loader.load_sift(n_test=N_TEST)
    except Exception as e:
        print(f"Error loading SIFT: {e}")
        return

    # 2. Compute Ground Truth (in 128D) & Hardness Scores (Track A)
    print("\n[Track A] Computing Theoretical Hardness (128D)...")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean').fit(X_train_orig)
    true_dists, true_indices = nbrs.kneighbors(X_test_orig)
    
    hardness_scores = []
    print("  Calculating derivatives...")
    for c in true_dists.flatten():
        h = calculate_hardness(c, LSH_R)
        hardness_scores.append(h)
    
    hardness_scores = np.array(hardness_scores)
    print(f"  Hardness Range: [{min(hardness_scores):.4f}, {max(hardness_scores):.4f}]")

    # 3. Run Empirical Pipeline (Track B)
    print("\n[Track B] Running JL + LSH Experiment (32D)...")
    
    # A. JL Projection
    print(f"  Projecting 128D -> {TARGET_DIM}D...")
    jl = GaussianRandomProjection(n_components=TARGET_DIM, eps=0.1, random_state=42)
    X_train_proj = jl.fit_transform(X_train_orig)
    X_test_proj = jl.transform(X_test_orig)
    
    # B. Build Index
    lsh = E2LSH(num_tables=LSH_L, hash_bits=LSH_K, dim=TARGET_DIM, r=LSH_R)
    lsh.fit(X_train_proj)
    
    # C. Query & Verify
    print("  Querying & Measuring Recall...")
    success_flags = [] # 1 if found, 0 if missed
    
    for i in range(N_TEST):
        # We need the raw query logic here to get boolean result
        candidates = set()
        query_vec = X_test_proj[i]
        
        # Collect candidates
        for table_idx in range(LSH_L):
            h_row = lsh._compute_hash(query_vec, table_idx)[0]
            candidates.update(lsh.tables[table_idx].get(tuple(h_row), []))
        
        # Check if TRUE neighbor is in candidates
        # (Note: We use the index from 128D ground truth)
        true_idx = true_indices[i][0]
        
        if true_idx in candidates:
            success_flags.append(1)
        else:
            success_flags.append(0)
            
    success_flags = np.array(success_flags)
    global_recall = np.mean(success_flags)
    print(f"  Global Empirical Recall: {global_recall:.2%}")
    
    if global_recall > 0.98 or global_recall < 0.02:
        print("\nWARNING: Global recall is too extreme (near 0% or 100%).")
        print("The correlation plot might look flat. Adjust LSH_K to fix.")

    # 4. Correlation Analysis & Plotting
    print("\n[Analysis] Generating Hardness vs. Recall Plot...")
    
    # Sort data by hardness
    sorted_indices = np.argsort(hardness_scores)
    sorted_hardness = hardness_scores[sorted_indices]
    sorted_success = success_flags[sorted_indices]
    
    # Binning (Deciles)
    n_bins = 10
    bins = np.array_split(np.arange(len(sorted_hardness)), n_bins)
    
    bin_avg_hardness = []
    bin_avg_recall = []
    
    for b_idx in bins:
        bin_avg_hardness.append(np.mean(sorted_hardness[b_idx]))
        bin_avg_recall.append(np.mean(sorted_success[b_idx]))
        
    # PLOT
    plt.figure(figsize=(10, 6))
    
    # Scatter of bins
    plt.plot(bin_avg_hardness, bin_avg_recall, 'o-', color='#d62728', linewidth=2, markersize=8, label='Binned Recall')
    
    # Trend line fit
    z = np.polyfit(bin_avg_hardness, bin_avg_recall, 1)
    p = np.poly1d(z)
    plt.plot(bin_avg_hardness, p(bin_avg_hardness), "k--", alpha=0.5, label=f'Trend (Slope={z[0]:.2f})')
    
    plt.title(f"Novelty Verification: Local Lipschitz Hardness vs. Recall\n(SIFT-1M, JL 128->32, r={LSH_R})", fontsize=14)
    plt.xlabel("Theoretical Hardness Score (Derived from Math)", fontsize=12)
    plt.ylabel("Empirical Recall Probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotation explaining the metric
    plt.annotate('Stable Queries\n(Low Derivative)', xy=(min(bin_avg_hardness), max(bin_avg_recall)), 
                 xytext=(min(bin_avg_hardness), max(bin_avg_recall)-0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.annotate('Unstable Queries\n(High Derivative)', xy=(max(bin_avg_hardness), min(bin_avg_recall)), 
                 xytext=(max(bin_avg_hardness)-0.5, min(bin_avg_recall)+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    save_path = "results/hardness_correlation_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Ensure results dir exists
    if not os.path.exists("results"):
        os.makedirs("results")
        
    run_hardness_experiment()