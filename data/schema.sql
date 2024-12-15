-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create sample table
CREATE TABLE abstracts (
    pmid INT PRIMARY KEY,
    text VARCHAR(4096) NOT NULL,
    -- Embedding size from Sentence-Transformers is 384
    embedding vector(384)
);
