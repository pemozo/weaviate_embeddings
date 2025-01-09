import weaviate
import os

print('Test Sandbox')

client = weaviate.Client("http://localhost:8080")

# Check if the client is ready
try:
    if client.is_ready():
        print("Connected to Weaviate!")
    else:
        print("Connection failed!")
except Exception as e:
    print(f"Error: {e}")

# # Schema with deactivated vectorizer
# schema = {
#     "class": "CustomObject",
#     "description": "Ein Beispiel für benutzerdefinierte Vektorisierung",
#     "properties": [
#         {"name": "name", "dataType": ["text"]},
#         {"name": "description", "dataType": ["text"]}
#     ],
#     "vectorizer": "none"  # Deactivate vectorizer
# }

# client.schema.create_class(schema)

schema = client.schema.get()
print("Cluster schema:", schema)
