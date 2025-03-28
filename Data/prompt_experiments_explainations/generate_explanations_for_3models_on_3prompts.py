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

models = [
    "mistral:7b-instruct-v0.2-q8_0",
    "llama3.3:70b-instruct-q6_K",
    "Hermes-3-Llama-3.1-70B-Q5_K_S:latest"
]
# model = "Hermes-3-Llama-3.1-70B-Q5_K_S:latest"

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
few_shot_prompt = """
you are AI agent which is specialized in analyzing a text for 3 different classes which is hate speech, offensive and normal, and then provide and educated
explanation for why the text falls into any of these categories. 
In order for you to do that, you have the following key feautures available to you which helps you in your analysis and creating the explanation for it

key_features: 

{'post_id': ## post id . ignore it
 'tweet_text': #its the main user text that you need to write explaination on.
 'key_features': [] # list of the words that made a decision, important feature
 'target': # targetted audience. 
 'label': # either offensive, hate speech or normal.   
 }
below are some of the example tweet_text and their explanations for each categories, refer those and learn from them to generate your explanations :

Label -> Normal

1. "i know this is somewhat hypocritical but what the fuck is the deal with queer guys and slightly too small white tshirts"
Explanation: The tweet has been labeled normal because the user is solely expressing an observation. However the use of vulgar terms such as “fuck” may indicate that the user has a rather negative outlook towards topics concerning queers. Given that the context is not clear it may be understood as a neutral observation. The targets used within the dataset do not contain the word “queer” but the word homosexual. 


2. “wore a nipsey blue suit today only the niggers in the work place feel the energy”
Explanation:This tweet has been labeled normal because it seems to be a positive reference of paying tribute to the rapper nipsey hussle who is known for a specific shade of blue that is affiliated to the crib gang culture of LA.
The use of the term "niggers" is offensive and rooted in racism. Even if no direct harm or attack is evident, it perpetuates a derogatory term, which many systems and definitions include as hate speech. However, disagreement in labeling often arises due to contextual interpretations. Example for that could be Cultural or Social Context that goes Intended Meaning. In the tweet addition The phrase might be interpreted as describing a situation where the "energy" of the outfit is appreciated by a certain group, which could be seen as neutral, despite the offensive language.


3. rafiullah shaheed a young man from levi force was shot dead by unknown men in muslim bagh a levies officer was killed in a bomb blast in lashkar gah on sunday while militants were also killed on the same day three people including mullahaniyev were killed in chaman"
Explanation: The statement is a factual report of events without any explicit derogatory, hateful, or harmful language targeting a specific individual or group based on characteristics like race, religion, or ethnicity. It lacks the intention to incite hate or violence, which is key to labeling content as hate speech.

Label -> Offensive

1) “pocahontas jumps in bed with moslem terrorists interesting combo”
Explanation:This statement is labeled offensive because it uses derogatory language and stereotypes to insult and demean individuals or groups. Referring to someone as "Pocahontas" in a pejorative context is disrespectful, and associating them with "Muslim terrorists" perpetuates harmful stereotypes and biases, making the comment offensive and inappropriate.

2) iIslamophobic you for real and how is islam a race you half wit
Explanation: It is labled offensive beacuse it contains an insult.

3) treat the immigrants as drug traffickers they will soon have a change of mind
Explanation: This statement is offensive because it unjustly stereotypes immigrants as drug traffickers, promoting a harmful generalization that dehumanizes and criminalizes an entire group of people. Such language fosters prejudice, discrimination, and hostility toward immigrants. Furthermore it incitement to


Label -> Hate Speech

1)you niggas is fake asl that why you dont see none round me
Explanation:This statement is classified as hate speech because it uses a racial slur ("niggas"), which is historically and culturally offensive, particularly against Black individuals. Even when used in some contexts as reclaimed language, its use in this statement appears derogatory, contributing to a hostile and discriminatory tone. The inclusion of aggressive language further amplifies its hateful nature.


2) "online nazi haha guys the left can not meme <number> white privilege diversity is our strength race is just skin color the holocaust racist sexist bigot homophobe islamophobe anti semite"
Explanation: This statement is hate speech because it mocks and trivializes serious societal issues, such as racism, sexism, and discrimination, using a sarcastic tone to dismiss terms like "white privilege," "diversity is our strength," and "anti-Semite." By doing so, it undermines the significance of these issues and belittles those advocating for equality. References to "Nazi" and "the Holocaust" in a dismissive context further minimize historical atrocities and signal alignment with extremist ideologies. Overall, the statement ridicules efforts to address inequality and promotes hostility, making it hate speech.

3) "pro tip the niggers do not care if you never owned slaves granddaddy fought the natzis or great great grandad ground the confederacy into dust with the boot heel of the proto bolshevik yankee gubment force of weight and arms behind it they hate you because you are white and you exist altright"
Explanation:This statement is classified as hate speech because it uses the racial slur ("niggers"). The statement further vilifies an entire group by making sweeping generalizations, such as claiming that Black people "hate you because you are white and you exist." This kind of language fosters an "us vs. them" narrative, portraying Black people as inherently hostile toward white people, which promotes racial division and resentment. The inclusion of terms like "proto Bolshevik Yankee gubment force" and "altright" reflects alignment with extremist ideologies often associated with white nationalism and discriminatory rhetoric. Additionally, references to historical events such as "granddaddy fought the Nazis" and "great great grandad ground the Confederacy" are weaponized to evoke a sense of pride in white heritage while simultaneously vilifying Black individuals. 

# Notes:
1) the explnatation should be consice. 
2) your explanations should be analytical and starightforward. 
3) while generating the explanations you should also consider the possible context for the text which the user might have in mind when writing the tweet
4) The purpose of these explanations is to educate the user as to why certain text written by them falls into certain category and can harm a 
certain group of people.
Now you will be given user input and you have to write explainations.
"""

zero_shot_prompt = """
you are AI agent which is specialized in analyzing a text for 3 different classes which is hate speech, offensive and normal, and then provide and educated
explanation for why the text falls into any of these categories.
In order for you to do that, you have the following key feautures available to you which helps you in your analysis and creating the explanation for it

available data: 

{'post_id': ## post id . ignore it
 'tweet_text': #its the main user text that you need to write explaination on.
 'key_features': [] # list of the words that made a decision, important feature
 'target': # targetted audience. 
 'label': # either offensive, hate speech or normal.   
 }
Based on the the label present in the available data, write your explanation for each tweet text.

# Notes:
1) the explnatation should be consice. 
2) your explanations should be analytical and starightforward. 
3) while generating the explanations you should also consider the possible context for the text which the user might have in mind when writing the tweet
4) The purpose of these explanations is to educate the user as to why certain text written by them falls into certain category and can harm a 
certain group of people.
Now you will be given user input and you have to write explainations.
"""

prompts = [
    {"prompt": zero_shot_prompt, "prompt_name": "zero_shot_prompt"},
    {"prompt": few_shot_prompt, "prompt_name": "few_shot_prompt"},
    {"prompt": few_shot_CoT_prompt, "prompt_name": "few_shot_CoT_prompt"},
]
# prompt = few_shot_CoT_prompt
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
    for model in models:
        for prompt in prompts:
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
                            "Model": model,
                            "Prompt Name": prompt_name,
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
                    df_results.to_csv("Data/prompt_experiments_explainations/llm_results_final_prompt_001.csv")
                    print(f"Progress saved after {idx} runs.")

# Convert results to DataFrame
df_results = pd.DataFrame(results)
# df_results = df_results.sort_values(by="Row Number", ascending=True)
# Save results to CSV
df_results.to_csv("Data/prompt_experiments_explainations/llm_results_final_prompt_001.csv")

