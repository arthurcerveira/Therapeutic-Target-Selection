import json
import psycopg
from dotenv import load_dotenv
import gradio as gr

from rag_utils import (
    DATABASE_URL,
    get_embeddings, 
    get_embeddings_model, 
    SYSTEM_PROMPT, 
    PROMPT_TEMPLATE,
    get_gemini_completion
)


load_dotenv()

RETRIEVE_QUERY = """
    SELECT
        pmid,
        text,
        embedding <=> cast(%s::vector as vector) as distance
    FROM abstracts
    ORDER BY embedding <=> cast(%s::vector as vector)
    LIMIT %s
"""


def retrieve_similar_abstracts(query, limit, conn, model=None, query_extension="therapeutic targets"):
    # Generate embedding for search query
    query = f"{query} AND {query_extension}"

    query_embedding = list(get_embeddings(query, model))
    data = (query_embedding, query_embedding, limit)

    # Perform vector distance search
    try:
        with conn.cursor() as cur:
            cur.execute(RETRIEVE_QUERY, data)
            results = cur.fetchall()
    except Exception as e:
        print(f"Error retrieving data: {e}")
        results = []

    return results


def rag_pipeline(disease, limit=5):
    embeddings_model = get_embeddings_model()

    with psycopg.connect(DATABASE_URL) as conn:
        results = retrieve_similar_abstracts(disease, limit, conn, embeddings_model)

    abstracts = list()

    for pmid, text, distance in results:        
        # print(f"PMID: {pmid} (Distance: {distance:.2f})")
        # print(f"{text}\n")
        abstracts.append(text)

    chunks = "\n".join([f"{chunk}\n" for chunk in abstracts])
    prompt = PROMPT_TEMPLATE.replace("{disease}", disease).replace("{abstracts}", chunks)

    response = get_gemini_completion(prompt, SYSTEM_PROMPT, json_format=True)
    text = response.text

    # Convert JSON if valid
    try:
        text_dict = json.loads(text)
        text = json.dumps(text_dict, indent=4)
    except json.JSONDecodeError:
        pass

    return text, chunks

if __name__ == "__main__":
    query = "Alzheimer's disease"

    demo = gr.Interface(
        fn=rag_pipeline,
        inputs=[gr.Textbox(lines=1, label="Disease", placeholder=query), 
                gr.Slider(1, 20, step=1, value=5, label="Abstracts Limit")],
        outputs=[gr.Textbox(lines=5, label="Disease targets"),
                 gr.Textbox(lines=20, label="Retrieved Abstracts")],
        title="Therapeutic Targets Identification",
        description="Identify therapeutic targets for a given disease query."
    )

    demo.launch()