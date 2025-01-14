import weaviate
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import pickle
from skipgram import SkipGram
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

file_name = "yoochoose-clicks.dat"
pkl_file_name= "yoochoose_trigrams.pkl"
checkpoint_path = "finished_embedding_YooChooseEmbedding.pt"

columns = ["session_id", "ts", "item_id", "category_id"]

dtype_mapping = {
    "session_id": "UInt32",
    "ts": "str",
    "item_id": "UInt32",
    "category_id": "category"
}

context_size = 5

# # Load environment variables from .env file
load_dotenv()

# # Get the file path from the environment variable
file_path = os.getenv("PATH_TO_ORIGINAL_DATA")
model_path = os.getenv("PATH_TO_MODELS")

# Load checkpoint
checkpoint = torch.load(model_path + checkpoint_path, map_location=torch.device("cpu"))

# Check embedding dim
embedding_weights = checkpoint["model"]["embedding.weight"]
embedding_dim = embedding_weights.shape[1]

print(f"Embedding-Dimension: {embedding_dim}")

# Load the data
# Data Source: https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015
data = pd.read_csv(file_path + file_name, names=columns, dtype=dtype_mapping)

print(data.head())

# transform to timestamp (in seconds)
data.ts = data.ts.apply(lambda x: int(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()))
data.sort_values(by="ts", inplace=True)

#load vocabulary mapping from pickle file
with open(model_path + pkl_file_name, "rb") as f:
    pkl_model = pickle.load(f)

print(pkl_model.keys())
print(pkl_model.values())

# Extract the action mapping
if "actions_map" in pkl_model:
    action_mapping = pkl_model["actions_map"]
    #print("Action Mapping:", action_mapping)
else:
    print("Action Mapping not found.")

print(pkl_model)
embedding = SkipGram.create_from_checkpoint(model_path + checkpoint_path, action_mapping, embedding_dim, context_size)



# Mapping of the item_ids from the Yoochoose data with the action_mapping
data["item_id_mapped"] = data["item_id"].apply(lambda x: action_mapping.get(np.int32(x), len(action_mapping)))

# Function to vectorize items
def vectorize_items(item_ids, model):
    item_ids_tensor = torch.tensor(item_ids, dtype=torch.long)
    with torch.no_grad():
        embeddings = model.embed(item_ids_tensor)
    return embeddings.numpy()

# Calculate the vectors for the first 10 items
item_ids = data["item_id_mapped"].tolist()
item_vectors = vectorize_items(item_ids, embedding)

print(item_vectors)

example_item_id = data["item_id_mapped"].iloc[11]  # Beispiel-Item (numerische ID)
example_vector = item_vectors[example_item_id]  # Vektor des Items

# Ähnlichkeit berechnen
similarities = cosine_similarity([example_vector], item_vectors)

# IDs der ähnlichsten Produkte (sortiert nach Ähnlichkeit)
similar_indices = np.argsort(similarities[0])[::-1][:5]  # Top 5 ähnliche Produkte

# Rückführung der numerischen IDs in die ursprünglichen item_id
reverse_action_mapping = {v: k for k, v in action_mapping.items()}  # Mapping umkehren
# Check if the key exists in the dictionary before accessing it
similar_items = []
for idx in similar_indices:
    if idx in reverse_action_mapping:
        similar_items.append(reverse_action_mapping[idx])
    else:
        print(f"Key {idx} not found in reverse_action_mapping")

print(f"Ähnliche Produkte für Item {example_item_id}: {similar_items}")