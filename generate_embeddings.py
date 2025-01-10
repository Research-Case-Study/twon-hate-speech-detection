from sentence_transformers import SentenceTransformer
from collections import Counter
import requests
import json
import os
from tqdm import tqdm
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Change to the parent directory of the notebook



EMBEDDINGS_MODEL = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()
df = pd.read_csv("Data/dataset.csv", index_col=0)
print("generating embeddings...")
# Generate embeddings for each row in the 'tweet_text' column
tqdm.pandas()

# Generate embeddings with progress tracking
df['X_train'] = df['tweet_text'].progress_apply(lambda text: EMBEDDINGS_MODEL.encode(text))

df.to_csv("dataset_mit_embeddings.csv")