import os
import yaml

from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Defaults
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'
LLM_CLIENT = "gemini-1.5-flash"

with open("prompt_template.yml", "r") as file:
    prompts = yaml.safe_load(file)

SYSTEM_PROMPT = prompts["System_Prompt"]
PROMPT_TEMPLATE = prompts["Prompt"]

genai.configure(api_key=os.getenv('GEMNI_API_KEY'))


def get_embeddings_model(model_name=EMBEDDINGS_MODEL):
    return SentenceTransformer(model_name)


def get_llm_model(model_name=LLM_CLIENT):
    return genai.GenerativeModel(model_name)


def get_embeddings(sentences, model=None):
    if model is None:
        model = get_embeddings_model()

    return model.encode(sentences)


def get_gemini_completion(prompt, system_prompt=None, json_format=False):
    client = get_llm_model()

    # Recommended format for system prompt in Gemini
    # https://www.googlecloudcommunity.com/gc/AI-ML/Gemini-Pro-Context-Option/m-p/684704/highlight/true#M4159
    if system_prompt:
        chat = client.start_chat(
            history=[
            {
                "role": "user",
                "parts": [{"text": "System prompt: " + system_prompt}]
            },
            {
                "role": "model",
                "parts": [{"text": "Understood."}]
            }
        ])
    else:
        chat = client.start_chat()

    generation_config = genai.GenerationConfig(
        temperature=0.1
    ) 
    
    if json_format:
        generation_config.response_mime_type = "application/json"

    # response = client.generate_content(prompt, generation_config=generation_config)
    response = chat.send_message(prompt, generation_config=generation_config)

    return response
