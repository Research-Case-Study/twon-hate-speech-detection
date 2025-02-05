# import libraries
import trace

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from collections import Counter
import requests
import json
import os
from tqdm import tqdm
import pandas as pd
import json
from qdrant_client import QdrantClient
import torch
from sentence_transformers import SentenceTransformer

df = pd.read_csv("Data/TEST_DF.csv", index_col=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
EMBEDDINGS_MODEL = SentenceTransformer(
    "dunzhang/stella_en_1.5B_v5",
    trust_remote_code=True,
    device="cpu"
)# .cuda()


qdrant_client = QdrantClient(
    url="https://7ef18c4d-2ef6-4fb0-9243-0ac62546593c.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="BR8zsNr5lEYrqJPL4EknUj2oRska2JO1nHwPFawlFMqZIrYMuGZ0Wg",
)

print(qdrant_client.get_collections())

index_name = "HateXplain_gpu_stella_0"


def fetch_explanations_rag(user_input, top_k):

    embedding = EMBEDDINGS_MODEL.encode(user_input)
    search_result = qdrant_client.search(
        collection_name=index_name, query_vector=embedding, limit=top_k
    )

    result = [
        {
            "text": item.payload["tweet_text"],
            "explanation": item.payload["Response"],
            "key_features": item.payload["key_features"],
            # "score": item.score,
            "label": item.payload["label"]
        }
        for item in search_result
    ]
    return result


# Define LLM inference function to use later
# API_URL = "https://inf.cl.uni-trier.de/chat/"
API_URL = "http://localhost:11434/api/chat"

def llm(model_name, system_prompt, input_query, rag_context):
    rag_context = f"Additional Context: {rag_context}"
    # Construct the request payload
    payload = {
        "messages": [
            {"content": system_prompt, "role": "system"},
            {"content": rag_context, "role": "system"},
            {"content": input_query, "role": "user"}
        ],
        "model": model_name,
        # "options": {"stream": False},
        "stream": False
    }

    # Set the request headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve response. Status code: {response.status_code}, Details: {response.text}")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while connecting to the API: {e}")
        return None

system_prompt_classification = """
Lets classify the following text for hate speech.

if its hatespeech your reply will be: "hate speech"
if its not hatespeech your reply will be : "normal"
if its offensive and not hatespeech, your reply will be : "offensive"

You are also given some related examples and their decision along with explainations why the decision was took. If those examples help make an unbiast decision contextually.
Focus more on userinput.


Here is user input and reply with only one word ONLY such as [hate speech, normal, or offensive] 
# Only Reply with one word """


model_name = "Hermes-2-70B-i1-IQ2_XXS:latest"
classification_column = f"RAG_{model_name}"

# Ensure column exists
df[classification_column] = None  

# Process each tweet in DataFrame
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Tweets"):
    try:
        user_input = "Userinput: " + row["tweet_text"]
        
        # Fetch RAG results
        rag_results = fetch_explanations_rag(user_input, top_k=2)
        if rag_results is None:
            df.at[idx, classification_column] = None
            continue

        rag_results_str = json.dumps(rag_results, indent=4)
        rag_results_str += "\nOnly Reply with one word"
        # Get LLM classification
        response = llm(model_name, system_prompt_classification, user_input, rag_results_str)
        df.at[idx, classification_column] = response['message']['content'] # ['response']

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        df.at[idx, classification_column] = None  # Store NaN if an error occurs

# Save updated DataFrame
df.to_csv("classification_experiments/rag/classified_test_df_Hermes-2.csv")

print("Classification completed and saved to 'classified_test_df.csv'")

