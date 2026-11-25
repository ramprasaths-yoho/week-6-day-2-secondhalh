"""Bonus - two collections, two embedding models, compare retrieval."""
from sentence_transformers import SentenceTransformer
from common import *

client = get_client()
MODELS = {
    "docs_minilm": ("all-MiniLM-L6-v2", 384),
    "docs_mpnet":  ("all-mpnet-base-v2", 768),
}
loaded = {}
for coll, (mname, dim) in MODELS.items():
    loaded[coll] = SentenceTransformer(mname)
    recreate_collection(client, coll, dim)
    client.upsert(coll, points=build_points(loaded[coll]))

queries = ["How many vacation days do I get?",
           "What if someone steals my laptop?",
           "When do I get paid?",
           "How fast must a data leak be reported?"]

for q in queries:
    print("\n" + "=" * 70 + f"\nQUERY: {q}")
    for coll, (mname, _) in MODELS.items():
        v = loaded[coll].encode(q, normalize_embeddings=True).tolist()
        pts = client.query_points(coll, query=v, limit=3, with_payload=True).points
        print(f"\n  [{mname}]")
        for p in pts:
            print(f"    {p.score:.3f}  ({p.payload['category']}) {p.payload['text']}")
