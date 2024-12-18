from pathlib import Path
import os

from dotenv import load_dotenv
import psycopg
from tqdm import tqdm
from rag_utils import get_embeddings, DATABASE_URL


load_dotenv()

QUERY = """
    INSERT INTO abstracts (pmid, text, embedding)
    VALUES (%s, %s, %s::vector)
    ON CONFLICT DO NOTHING
"""
CHAR_LIMIT = 4096


def load_abstracts_to_db(folder_path, chunk_size=10_000, conn=None):
    """
    Load abstracts from text files into the database.

    Args:
        file_path (str): Path to the directory containing the text files.
        chunk_size (int): Number of abstracts to insert into the database at a time.
        conn (psycopg.connection): Connection to the database.
    """
    folder_path = Path(folder_path)
    files = list(folder_path.glob('**/*.txt'))

    total = len(files) // chunk_size
    for i in tqdm(range(0, len(files), chunk_size), desc='Loading abstracts', total=total):
        chunk = files[i:i + chunk_size]
        pmids, texts = [], []

        # Prepare data for insertion
        for file in chunk:
            pmid = int(file.stem)
            text = file.read_text()
            pmids.append(pmid)
            texts.append(text[:CHAR_LIMIT])

        embeddings = get_embeddings(texts)
        zipped = zip(pmids, texts, embeddings)
        data = [(pmid, text, list(embedding)) for pmid, text, embedding in zipped]

        try:
            with conn.cursor() as cur:
                cur.executemany(QUERY, data)  # Batch insert using executemany
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error inserting data: {e}")


if __name__ == "__main__":
    N = 50
    abstracts_folder = range(N)

    for i in abstracts_folder:
        with psycopg.connect(DATABASE_URL) as conn:
            print(f"Loading abstracts from folder data/pubmed/{i}")
            load_abstracts_to_db(f'data/pubmed/{i}', conn=conn)
