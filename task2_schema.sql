-- Task 2 - pgvector (run inside psql / PyCharm / pgAdmin)
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT NOT NULL,
    category  TEXT NOT NULL,
    embedding vector(384)
);

-- Query 1: 3 nearest rows (cosine distance). %s is bound by psycopg2 in Python
-- SELECT id, content, embedding <=> %s AS distance
-- FROM documents ORDER BY embedding <=> %s LIMIT 3;

-- Before index
EXPLAIN ANALYZE
SELECT id, content FROM documents
ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = 1)
LIMIT 3;

-- HNSW index
CREATE INDEX documents_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- After index (note: with only 20 rows the planner may still pick Seq Scan;
-- force index use for the demo)
SET enable_seqscan = off;
EXPLAIN ANALYZE
SELECT id, content FROM documents
ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = 1)
LIMIT 3;
RESET enable_seqscan;

-- Filtered search
EXPLAIN ANALYZE
SELECT id, content FROM documents
WHERE category = 'tech'
ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = 1)
LIMIT 3;
