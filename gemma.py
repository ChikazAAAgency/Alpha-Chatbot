from huggingface_hub import login
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COLLECTION_NAME = "Alpha"
VECTOR_SIZE = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


genai.configure(api_key=GOOGLE_API_KEY)

# function for initializing gemma pipeline
# Initialize Gemma pipeline (do not hardcode tokens).
# Use `ACCESS_TOKEN` from environment variables and `login(ACCESS_TOKEN)`.

# function for generating embeddings

def get_embedding(text, model, task_type="retrieval_document"):
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding

#  function for connecting to Qdrant vector database

def connect_to_vectorDB():
    print("Connecting to Qdrant Cloud...")
    q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not q_client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    return q_client

# function to chunk text into smaller pieces

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])
    return chunks


# function to extract text from PDF files

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# function to generate embeddings and upload to Qdrant

def generate_and_upload_embeddings(model, q_client, chunks):
    points = []

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(text=chunk.page_content, model=model)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.page_content
                    # Add any other metadata you want to store
                },
            )
        )

    operation_info = q_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points,
    )

    print(f"Upsert operation completed: {operation_info}")
    print(f"Uploaded {len(points)} chunks to Qdrant collection '{COLLECTION_NAME}'.")


def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)

    return response.text



#function for querying RAG system

def query_rag(user_query, model, q_client, system_prompt = None):
    
    # 1. Generate embedding
    query_embedding = get_embedding(user_query, model, task_type="retrieval_query")

    # 2. Query Qdrant
    search_results = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=5
    )

    # 3. Extract retrieved text chunks
    context_texts = [point.payload["text"] for point in search_results.points]

    # 4. Build prompt
    context = " ".join(context_texts)
    prompt = f"System Instructions : {system_prompt}\n\nContext: {context}\n\nQuestion: {user_query}\n\nAnswer:"

    # 5. Generate answer using gemini
    answer = ask_gemini(prompt)

    return answer


def main():
    login(ACCESS_TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/embeddinggemma-300M"
    model = SentenceTransformer(model_id).to(device=device)

    q_client = connect_to_vectorDB()
    text = pdf_to_text(r"C:\Users\risha\OneDrive\Desktop\Project_Alpha\website_content.pdf")
    chunks = chunk_text(text)
    generate_and_upload_embeddings(model, q_client, chunks)

    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        answer = query_rag(user_query, model, q_client, system_prompt="You are a helpful assistant.")
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
