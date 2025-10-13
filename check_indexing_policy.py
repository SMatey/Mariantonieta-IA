# check_indexing_policy.py
import os, json
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

load_dotenv()

client = CosmosClient(os.getenv("AZURE_COSMOS_ENDPOINT"), credential=os.getenv("AZURE_COSMOS_KEY"))
db = client.get_database_client(os.getenv("AZURE_COSMOS_DB"))
c  = db.get_container_client(os.getenv("AZURE_COSMOS_CONTAINER"))

props = c.read()
ip = props.get("indexingPolicy", {})
print("Has vectorEmbeddingPolicy:", "vectorEmbeddingPolicy" in props)
print("vectorIndexes:", json.dumps(ip.get("vectorIndexes", []), indent=2))
