"""Task 2 - pgvector in PostgreSQL (+ bonus: cosine vs L2 vs inner product)"""
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np

DSN = dict(host="localhost", port=5432, dbname="vectordb", user="postgres", password="postgres")

SENTENCES = [
    # tech
    ("Python is a popular programming language for data science.", "tech"),
    ("Neural networks learn patterns from large datasets.", "tech"),
    ("PostgreSQL is a powerful open source relational database.", "tech"),
    ("Docker containers make deployment consistent across machines.", "tech"),
    ("Vector databases store embeddings for semantic search.", "tech"),
    ("FastAPI makes building REST APIs quick and easy.", "tech"),
    # food
    ("Chennai is famous for its crispy dosa and filter coffee.", "food"),
    ("Biryani is cooked with fragrant basmati rice and spices.", "food"),
    ("Fresh mangoes are sweet and juicy in the summer.", "food"),
    ("A good sambar needs tamarind, lentils and fresh vegetables.", "food"),
    ("Pizza with extra cheese is a favourite among kids.", "food"),
    # sports
    ("Cricket is the most followed sport in India.", "sports"),
    ("The football team scored a late winning goal.", "sports"),
    ("She trains every morning for the marathon.", "sports"),
    ("Chennai Super Kings won the IPL final last season.", "sports"),
    # travel
    ("The Marina Beach in Chennai is one of the longest urban beaches.", "travel"),
    ("Mountains in Ooty are covered with tea gardens.", "travel"),
    ("Booking flights early usually saves a lot of money.", "travel"),
    ("The train journey through the Western Ghats is scenic.", "travel"),
    ("Kerala backwaters are best explored on a houseboat.", "travel"),
]

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dimensions
conn = psycopg2.connect(**DSN)
conn.autocommit = True
cur = conn.cursor()

# --- setup ---
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)
cur.execute("DROP TABLE IF EXISTS documents;")
cur.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL,
        embedding vector(384)
    );
""")

# --- insert 20 embedded sentences (parameterised) ---
texts = [s for s, _ in SENTENCES]
embs = model.encode(texts, normalize_embeddings=True)
for (text, cat), emb in zip(SENTENCES, embs):
    cur.execute("INSERT INTO documents (content, category, embedding) VALUES (%s, %s, %s)",
                (text, cat, emb))
print(f"Inserted {len(SENTENCES)} rows")

query = "Which database should I use to store embeddings?"
qv = model.encode(query, normalize_embeddings=True)


def explain(sql, params):
    cur.execute("EXPLAIN ANALYZE " + sql, params)
    print("\n".join(r[0] for r in cur.fetchall()))


# --- 3 nearest rows, parameterised ---
SQL_TOP3 = "SELECT id, content, category, embedding <=> %s AS dist FROM documents ORDER BY embedding <=> %s LIMIT 3"
cur.execute(SQL_TOP3, (qv, qv))
print("\nTop 3 (cosine):")
for r in cur.fetchall():
    print(f"  {r[0]:>2} | {r[3]:.4f} | [{r[2]}] {r[1]}")

# --- EXPLAIN ANALYZE before index ---
print("\n=== EXPLAIN ANALYZE (no index) ===")
explain("SELECT id FROM documents ORDER BY embedding <=> %s LIMIT 3", (qv,))

# --- create HNSW index and re-run ---
cur.execute("CREATE INDEX documents_embedding_hnsw ON documents "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);")
cur.execute("ANALYZE documents;")
cur.execute("SET enable_seqscan = off;")   # tiny table: force planner to show the index
print("\n=== EXPLAIN ANALYZE (HNSW index) ===")
explain("SELECT id FROM documents ORDER BY embedding <=> %s LIMIT 3", (qv,))
cur.execute("RESET enable_seqscan;")

# --- filtered search ---
SQL_FILT = ("SELECT id, content, embedding <=> %s AS dist FROM documents "
            "WHERE category = %s ORDER BY embedding <=> %s LIMIT 3")
cur.execute(SQL_FILT, (qv, "tech", qv))
print("\nTop 3 in category='tech':")
for r in cur.fetchall():
    print(f"  {r[0]:>2} | {r[2]:.4f} | {r[1]}")
print("\n=== EXPLAIN ANALYZE (filtered) ===")
cur.execute("SET enable_seqscan = off;")
explain(SQL_FILT, (qv, "tech", qv))
cur.execute("RESET enable_seqscan;")

# --- BONUS: cosine (<=>) vs L2 (<->) vs inner product (<#>) ---
print("\n=== BONUS: distance operators ===")
for name, op in [("cosine <=>", "<=>"), ("L2 <->", "<->"), ("inner product <#>", "<#>")]:
    cur.execute(f"SELECT id, content, embedding {op} %s AS d FROM documents ORDER BY embedding {op} %s LIMIT 3",
                (qv, qv))
    print(f"\n{name}")
    for r in cur.fetchall():
        print(f"  {r[0]:>2} | {r[2]: .4f} | {r[1]}")
print("\nNote: <#> returns the NEGATIVE inner product (so ORDER BY ASC works). "
      "With normalised vectors all three give the same ranking.")

cur.close()
conn.close()
