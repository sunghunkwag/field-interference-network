"""
benchmarks/benchmark_fin.py
Empirical complexity benchmark: dense O(n²) vs sparse O(n·k) coupling.
Reports only measured wall-clock timings. No fabricated speedup claims.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from fin.gnn.sparse_coupler import SparseResonanceCoupler


def dense_coupling(phi):
    """Reference O(n²) dense Kuramoto coupling."""
    diff = phi.unsqueeze(0) - phi.unsqueeze(1)
    return torch.sin(diff).mean(dim=1)


def timed(fn, repeats=20):
    for _ in range(3): fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'N':>6} {'k':>4} {'dense_ms':>12} {'sparse_ms':>12} {'ratio':>8}")
    for n in [64, 128, 256, 512]:
        torch.manual_seed(0)
        phi = torch.randn(n, device=device)
        pos = torch.randn(n, 3, device=device)
        k   = 8
        coupler = SparseResonanceCoupler(n, k_neighbors=k).to(device)
        coupler.build_graph(pos)
        dense_t  = timed(lambda: dense_coupling(phi)) * 1000
        sparse_t = timed(lambda: coupler(phi, pos))   * 1000
        ratio    = dense_t / max(sparse_t, 1e-9)
        print(f"{n:>6} {k:>4} {dense_t:>12.4f} {sparse_t:>12.4f} {ratio:>8.2f}x")
