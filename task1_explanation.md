# Task 1 - Write-up

## How HNSW finds neighbours quickly (5 lines)
1. HNSW builds a multi-layer graph: the bottom layer (0) holds every vector, each higher layer holds a random, exponentially smaller subset.
2. Each node links to its M closest neighbours in every layer it belongs to, so the graph is "small-world" (short paths between any two points).
3. Search starts at the entry point in the top, sparsest layer and greedily hops to whichever neighbour is closest to the query.
4. When no neighbour is closer, it drops one layer down and continues from that node - long jumps first, fine-grained steps later.
5. On layer 0 it explores `efSearch` candidates with a best-first search and returns the top-k, touching only a tiny fraction of the 50,000 vectors (~O(log N)).

## Exact vs ANN trade-off

| Aspect   | Exact (Flat / NumPy)             | ANN (HNSW)                                         |
|----------|----------------------------------|----------------------------------------------------|
| Accuracy | 100% recall (ground truth)       | ~90-99% recall, tunable via `efSearch`, `M`        |
| Speed    | O(N*D) per query, slows linearly | ~O(log N), often 10-100x faster                    |
| Memory   | Only raw vectors (N*D*4 bytes)   | Raw vectors + graph links (extra ~2*M*4 B/vector)  |
| Build    | None                             | Slow index build (`efConstruction`)                |
| Best for | Small data, need perfect results | Large data, low latency, "good enough" results     |
