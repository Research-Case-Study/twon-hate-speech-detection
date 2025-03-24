import json
from collections import Counter
import requests
import json
import os
import pandas as pd
import requests
import json
from tqdm import tqdm
import time

# models = [
#     "mistral:7b-instruct-v0.2-q8_0",
#     "llama3.3:70b-instruct-q6_K",
#     "Hermes-3-Llama-3.1-70B-Q5_K_S:latest"
# ]
model = "Hermes-3-Llama-3.1-70B-Q5_K_S:latest"

df = pd.read_csv("Data/dataset.csv", index_col=0)
df = df.iloc[15001:]
columns_to_drop = [
    "annotator_1_label", "annotator_1_target",
    "annotator_2_label", "annotator_2_target",
    "annotator_3_label", "annotator_3_target"
]
df["Row Number"] = df.index
# Drop the specified columns
df = df.drop(columns=columns_to_drop)

data = df.to_dict(orient='records')

few_shot_CoT_prompt = """
As an AI agent specialized in text analysis, your task is to classify text into one of three categories: hate speech, offensive, or normal. 
Based on this classification, provide a concise yet comprehensive explanation of why the text falls into the specified category.

Available data:
- tweet_text: The text to be analyzed.
- key_features: Crucial words or phrases that influenced the classification.
- target: The intended audience or group targeted by the text.
- label: The classification of the text (hate speech, offensive, or neutral).

Procedure:
1. Identify and briefly summarize the label's implication for the text.
2. Discuss how key features support this classification.
3. Evaluate the text's impact on or reference to the target group.
4. Assess the context and tone to gauge the user's intent.
5. Provide a concise conclusion that justifies the classification based on the analysis.

Guidelines:
- Keep explanations concise and focused.
- Emphasize key factors that support the classification.
- If the label is neutral but the text could be seen as offensive or hate speech, explain the rationale for its neutral classification.
- For an offensive label, detail the nature of the offense and the contribution of key features.

Example analyses:

1. **Hate Speech**:
   - tweet_text: "A negro moslem doctor got struck off for negligence when a white <number> year old kid with down syndrome died; the guardian's response is to help promote a crowdfunding campaign for her."
   - Key points include discriminatory terms and the sensitive context, targeting racial, religious, and disabled groups. The usage of derogatory language in describing an ethnic minority and religious group in a negative incident categorizes this as hate speech.

2. **Offensive**:
    - tweet_text: "as much as i appreciate a format of open discussion i really just want a facebook that i can say beaner on?"
    - The term "beaner" is a derogatory slur used to target individuals of Mexican descent. Its use in the tweet trivializes the harm caused by such language and perpetuates ethnic stereotypes.
    - This tweet is classified as offensive due to its use of explicit derogatory language aimed at a specific ethnic group.

3. **Neutral**:
   - tweet_text: "What’s everyone’s favorite food?"
   - This lacks controversial content or targeted language, making it neutral. The friendly tone and general nature affirm its classification, posing no harm or offense to any group.

Please analyze the given text using this streamlined reasoning framework.
"""

# prompts = [
#     {"prompt": zero_shot_prompt, "prompt_name": "zero_shot_prompt"},
#     {"prompt": few_shot_prompt, "prompt_name": "few_shot_prompt"},
#     {"prompt": few_shot_CoT_prompt, "prompt_name": "few_shot_CoT_prompt"},
# ]
prompt = few_shot_CoT_prompt
# Define LLM inference function to use later
API_URL = "http://localhost:11434/api/chat"


def llm(model_name, system_prompt, input_query, stream=False):
    # Construct the request payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_query}
        ],
        "stream": stream  # Ensure streaming behavior matches API expectations
    }

    # Set the request headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        # Send the POST request
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

        # Check for errors
        if response.status_code == 200:
            # print("Response received successfully:")
            return response.json()  # Parse JSON response
        else:
            print(f"Failed to retrieve response. Status code: {response.status_code}, Details: {response.text}")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while connecting to the API: {e}")
        return None


# # Function to select a range of 10 rows
# def pick_inputs(data, start_row, num_inputs=10):
#     if start_row + num_inputs > len(data):
#         raise ValueError("Start row plus number of inputs exceeds dataset size.")
#     return list(enumerate(data[start_row:start_row + num_inputs], start=start_row + 1))

# inputs = pick_inputs(data, start_row=720, num_inputs=1000)
# print(inputs)

# Run all combinations with progress bar and retry logic
results = []

# Use tqdm for progress tracking
with tqdm(total=len(data), desc="Processing LLM responses") as pbar:
    # for model in models:
    for idx, input_dict in enumerate(data, start=1):
        attempt = 0
        time_taken = 0.0  # Initialize time taken
        # print(input_dict)
        # input_text = input_dict.get("Input")
        row_number = input_dict["Row Number"] # input_dict.get("Row Number", idx)

        # prompt = prompt_dict["prompt"]
        # prompt_name = prompt_dict["prompt_name"]
        while attempt < 3:  # Retry up to 3 times
            try:
                start_time = time.time()  # Start timing

                response = llm(model, prompt, json.dumps(input_dict))
                end_time = time.time()  # End timing
                time_taken = round(end_time - start_time, 2)  # Calculate time taken
                explanation = response["message"]["content"]  # .strip()
                results.append({
                    "Row Number": row_number,
                    # "Model": model,
                    # "Prompt Name": prompt_name,
                    "Input": input_dict,
                    "tweet_text": input_dict.get("tweet_text"),
                    "key_features": input_dict.get("key_features"),
                    "label": input_dict.get("label"),
                    "target": input_dict.get("target"),
                    "Response": explanation,
                    "Time Taken (s)": time_taken
                })
                break
            except Exception as e:
                attempt += 1
                if attempt == 3:
                    # Log error and continue
                    results.append({
                        "Row Number": row_number,
                        # "Model": model,
                        # "Prompt Name": prompt_name,
                        "Input": input_dict,

                        "Response": "error",
                        "Time Taken (s)": time_taken
                    })
                    print(f"Error after 3 retries: {e}")
        pbar.update(1)
        # Save to CSV every 1000 runs
        if idx % 1000 == 0:
            df_results = pd.DataFrame(results)
            # df_results = df_results.sort_values(by="Row Number", ascending=True)
            df_results.to_csv("Data/knowledge_base_data/explainations_CoT_Hermes_partial_5.csv")
            print(f"Progress saved after {idx} runs.")

# Convert results to DataFrame
df_results = pd.DataFrame(results)
# df_results = df_results.sort_values(by="Row Number", ascending=True)
# Save results to CSV
df_results.to_csv("Data/knowledge_base_data/explainations_CoT_Hermes_partial_5.csv")

