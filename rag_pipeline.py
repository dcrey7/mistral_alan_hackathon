from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
from getpass import getpass

client = Mistral(api_key="TvlAaEIaJYB08ufulDTk4jiRj4HXGQGG")

# Read txt 
f = open('essay.txt', 'w')
f.write(text)
f.close()

#Split in smaller chunks 
## It’s more effective to identify and retrieve the most relevant information in the retrieval process later

chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Embeddings for each text chunk 
## For each text chunk, we then need to create text embeddings, which are numeric representations of the text in the vector space
def get_text_embedding(input):
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          inputs=input
      )
    return embeddings_batch_response.data[0].embedding


# Load into a vector database

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)