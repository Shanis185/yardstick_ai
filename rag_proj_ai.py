# Install required libraries
!pip install pinecone-client openai

import os
import pinecone
import openai
import time

# Replace these with your actual API keys
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"  # Change this to your Pinecone API key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"      # Change this to your OpenAI API key

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Define the index name using lowercase and hyphens
index_name = 'my-index'  # Changed to use hyphen instead of underscore

# Check if index exists
if index_name not in pc.list_indexes().names():
    # Create the index (ensure to specify a correct region supported by your plan)
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')  # Change region as needed
    )

# Connect to the index
index = pc.Index(index_name)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def get_openai_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Use the correct model name
    )
    return response['data'][0]['embedding']

def get_openai_embeddings_with_retry(text, retries=5):
    for i in range(retries):
        try:
            return get_openai_embeddings(text)
        except openai.error.RateLimitError:
            wait_time = 5 * (2 ** i)  # Increase wait time up to 5 seconds
            print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    raise Exception("Max retries exceeded for OpenAI API call")

# Example documents to embed and index
documents = [
    {"id": "1", "content": "This is a test document."},
    {"id": "2", "content": "This document is about machine learning."},
]

# Create a batch of embeddings
for doc in documents:
    try:
        embedding = get_openai_embeddings_with_retry(doc['content'])
        index.upsert(vectors=[(doc['id'], embedding)])
        print(f"Document {doc['id']} indexed successfully.")
    except Exception as e:
        print(f"Error processing document {doc['id']}: {e}")
