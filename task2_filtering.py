"""Task 2 - filtering + collection management."""
from sentence_transformers import SentenceTransformer
from qdrant_client.models import (Filter, FieldCondition, MatchValue, Range,
                                  FilterSelector, PayloadSchemaType)
from common import *

client = get_client()
model = SentenceTransformer(MODEL_NAME)
recreate_collection(client, COLLECTION, DIM)
client.upsert(COLLECTION, points=build_points(model))

def search(q, flt=None, limit=10):
    v = model.encode(q, normalize_embeddings=True).tolist()
    return client.query_points(COLLECTION, query=v, query_filter=flt, limit=limit, with_payload=True).points

# 1. payload filter: category == 'HR'
hr_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value="HR"))])
hr = search("leave and employee policy", hr_filter)
show(hr, "Filtered search (category == 'HR')")
print(f"Filtered search (category=HR): {len(hr)} results")

# 2. range filter: page >= 3
page_filter = Filter(must=[FieldCondition(key="page", range=Range(gte=3))])
pg = search("security and compliance rules", page_filter, limit=5)
show(pg, "Range filter (page >= 3)")

# combined: category == 'IT' AND page >= 3
both = Filter(must=[FieldCondition(key="category", match=MatchValue(value="IT")),
                    FieldCondition(key="page", range=Range(gte=3))])
show(search("protect my laptop", both, limit=5), "Combined (IT AND page >= 3)")

# 3. re-upsert SAME ids with changed text -> count must not grow
before = client.count(COLLECTION, exact=True).count
client.upsert(COLLECTION, points=build_points(model, suffix=" (updated 2026)"))
after = client.count(COLLECTION, exact=True).count
print(f"\nPoints before re-upsert: {before} | after: {after}")
print("Sample updated payload:", client.retrieve(COLLECTION, ids=[1])[0].payload)

# 4. delete points by filter, verify count
ops_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value="Ops"))])
n_ops = client.count(COLLECTION, count_filter=ops_filter, exact=True).count
client.delete(COLLECTION, points_selector=FilterSelector(filter=ops_filter), wait=True)
left = client.count(COLLECTION, exact=True).count
print(f"\nDeleted category=Ops ({n_ops} points). Count now: {left} (expected {after - n_ops})")

# 5. payload index on category
client.create_payload_index(COLLECTION, field_name="category",
                            field_schema=PayloadSchemaType.KEYWORD, wait=True)
client.create_payload_index(COLLECTION, field_name="page",
                            field_schema=PayloadSchemaType.INTEGER, wait=True)
print("\nPayload indexes created:", list(client.get_collection(COLLECTION).payload_schema.keys()))
print("""
Why index 'category'? Without an index Qdrant must scan each candidate's payload to
check the filter. A payload index gives a fast lookup of which points match, lets the
HNSW search plan filtering efficiently (pre-filter / filterable HNSW), and keeps latency
low and recall high as the collection grows. Index only fields you filter on.""")
