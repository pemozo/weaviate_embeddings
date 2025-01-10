import weaviate
import os
import pandas as pd
from dotenv import load_dotenv

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
file_path = os.path.join(parent_path, ".env")

# Load environment variables from .env file
load_dotenv()

# Get the file path from the environment variable
file_path = os.getenv("FILE_PATH")
print(file_path)

# Load the data
data = pd.read_pickle(file_path)

print(data.head())



# print('Test Sandbox')

# client = weaviate.Client("http://localhost:8080")

# # Check if the client is ready
# try:
#     if client.is_ready():
#         print("Connected to Weaviate!")
#     else:
#         print("Connection failed!")
# except Exception as e:
#     print(f"Error: {e}")

# # # Schema with deactivated vectorizer
# # schema = {
# #     "class": "CustomObject",
# #     "description": "Ein Beispiel für benutzerdefinierte Vektorisierung",
# #     "properties": [
# #         {"name": "name", "dataType": ["text"]},
# #         {"name": "description", "dataType": ["text"]}
# #     ],
# #     "vectorizer": "none"  # Deactivate vectorizer
# # }

# # client.schema.create_class(schema)

# schema = client.schema.get()
# print("Cluster schema:", schema)
