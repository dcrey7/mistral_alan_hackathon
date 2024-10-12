import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load JSON data
with open('FrenchMedMCQA/corpus/train.json', 'r') as f:
    data = f.read()
    questions = json.loads(data)

# Extract question text and options from JSON data
texts = []
options = []
for question in questions:
    texts.append(question['question'])
    options.append(list(question['answers'].values()))

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for questions
question_embeddings = model.encode(texts)

# Generate embeddings for options
option_embeddings = [model.encode(opt) for opt in options]

# Function to perform similarity search
def find_relevant_options(query, question_embeddings, option_embeddings, k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, question_embeddings)
    top_k_indices = similarities.argsort()[0][-k:]
    return [options[i] for i in top_k_indices]

# Function to generate response
def generate_response(query, relevant_options):
    prompt = f"Question: {query}\nOptions: {relevant_options}\nAnswer:"
    try:
        response = requests.post(
            'https://api.mistral.ai/v1/chat/completions',
            headers={'Authorization': 'TvlAaEIaJYB08ufulDTk4jiRj4HXGQGG'},
            json={'model': 'mistral-small', 'messages': [{'role': 'user', 'content': prompt}]}
        )
        response.raise_for_status()  # Raise an error for bad responses
        print(response.text)  # Print the response text
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None

# Example usage
query = "Une anémie est définie en pratique chez un adulte homme de moins de 65 ans par :"
relevant_options = find_relevant_options(query, question_embeddings, option_embeddings)
response = generate_response(query, relevant_options)
if response:
    print(response)