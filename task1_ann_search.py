"""Task 1 - Exact vs Approximate Search (NumPy vs FAISS Flat vs FAISS HNSW)"""
import time
import numpy as np
import faiss
import matplotlib.pyplot as plt

N, D, K = 50_000, 128, 10
rng = np.random.default_rng(42)
xb = rng.random((N, D), dtype=np.float32)       # database vectors
xq = rng.random((100, D), dtype=np.float32)     # 100 queries -> stable timings
q1 = xq[:1]                                      # single query (as in the task)


def timeit(fn, repeat=20):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return out, (time.perf_counter() - t) / repeat * 1000  # ms per call


# ---------- 1. Exact search with NumPy ----------
def numpy_search(q):
    d = np.sum((xb - q) ** 2, axis=1)            # squared L2 to every vector
    idx = np.argpartition(d, K)[:K]
    return idx[np.argsort(d[idx])]

truth_single, t_np = timeit(lambda: numpy_search(q1[0]), repeat=5)

# ---------- 2. FAISS IndexFlatL2 (exact) ----------
flat = faiss.IndexFlatL2(D)
flat.add(xb)
(_, I_flat), t_flat = timeit(lambda: flat.search(q1, K))

# ground truth for all queries (used for recall)
_, gt = flat.search(xq, K)


def recall_at_k(pred, truth):
    return np.mean([len(set(p) & set(t)) / K for p, t in zip(pred, truth)])


# ---------- 3. FAISS HNSW ----------
def build_hnsw(M, efC=200):
    t = time.perf_counter()
    idx = faiss.IndexHNSWFlat(D, M)
    idx.hnsw.efConstruction = efC
    idx.add(xb)
    return idx, time.perf_counter() - t

hnsw, build_t = build_hnsw(M=32)
hnsw.hnsw.efSearch = 64
(_, I_h), t_hnsw = timeit(lambda: hnsw.search(q1, K))
_, I_all = hnsw.search(xq, K)
rec_hnsw = recall_at_k(I_all, gt)

print(f"Exact (NumPy)      : {t_np:8.3f} ms")
print(f"Exact (IndexFlatL2): {t_flat:8.3f} ms  recall@10 = 1.00")
print(f"HNSW  (M=32, ef=64): {t_hnsw:8.3f} ms  recall@10 = {rec_hnsw:.2f}  (build {build_t:.1f}s)")
print(f"Exact: {t_np:.1f} ms | HNSW: {t_hnsw:.2f} ms with recall@10 = {rec_hnsw:.2f}")

# ---------- 4. Vary M and efSearch -> recall vs time ----------
results = []  # (M, ef, ms_per_query, recall)
for M in (8, 16, 32, 64):
    idx, _ = build_hnsw(M)
    for ef in (8, 16, 32, 64, 128, 256):
        idx.hnsw.efSearch = ef
        _, ms_batch = timeit(lambda: idx.search(xq, K), repeat=5)
        _, I = idx.search(xq, K)
        results.append((M, ef, ms_batch / len(xq), recall_at_k(I, gt)))

print("\nM   efSearch  ms/query  recall@10")
for M, ef, ms, r in results:
    print(f"{M:<3} {ef:<9} {ms:<9.4f} {r:.3f}")

plt.figure(figsize=(8, 5))
for M in (8, 16, 32, 64):
    pts = [(ms, r) for m, ef, ms, r in results if m == M]
    plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"M={M}")
    for (m, ef, ms, r) in [x for x in results if x[0] == M]:
        plt.annotate(str(ef), (ms, r), fontsize=7, xytext=(2, 3), textcoords="offset points")
plt.xlabel("Search time per query (ms)")
plt.ylabel("recall@10")
plt.title("HNSW: recall vs time (labels = efSearch)")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig("recall_vs_time.png", dpi=150)
print("Saved recall_vs_time.png")

# ---------- 5. Memory estimate for the trade-off table ----------
raw_mb = xb.nbytes / 1e6
hnsw_mb = raw_mb + N * 32 * 2 * 4 / 1e6   # vectors + ~2*M links/node (int32), rough
print(f"\nMemory: Flat ~ {raw_mb:.0f} MB | HNSW(M=32) ~ {hnsw_mb:.0f} MB (approx.)")
