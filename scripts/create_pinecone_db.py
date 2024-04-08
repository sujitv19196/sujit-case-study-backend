import csv 
import sys
import argparse
import time
import pinecone
import os 

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

def upload_to_pinecone(db_name: str, data_file: str):
    # Initialize Pinecone client
    print("Initializing Pinecone client...")
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')

    if db_name not in pinecone.list_indexes():
        print("Creating Pinecone index...")
        pinecone.create_index(db_name, dimension=1536)

    print("Uploading documents and embeddings to Pinecone...")
    index = pinecone.Index(db_name)

    print("Loading CSV...")
    csv.field_size_limit(sys.maxsize)
    loader = CSVLoader(file_path=data_file,  csv_args = {
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['text', 'url', 'depth', 'title', 'model_num', 'ps_num']},
        source_column='url',
        metadata_columns=['depth', 'title', 'model_num', 'ps_num']) # TODO add metadata columns
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print("Creating embeddings...")    
    embeddings = generate_embedding([doc.page_content for doc in docs])  # Function to generate embedding

    # Upload documents and embeddings in batches
    batch_size = 1000
    total_documents = len(documents)
    for i in range(0, total_documents, batch_size):
        batch_documents = documents[i:i+batch_size]
        ids = [doc.metadata['url'] for doc in batch_documents]
        batch_embeddings = embeddings[i:i+batch_size]
        index.upsert(id=ids, vectors=batch_embeddings)
        # casandra.upload_documents(batch_documents) TODO 
        print(f"Uploaded {min(i+batch_size, total_documents)} / {total_documents} documents")

    print("Upload complete.")

def generate_embedding(text):
    print("Getting hf Embeddings...")
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'batch_size': 8}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    
    return embeddings.embed_documents(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI for uploading CSV data to Pinecone index')
    parser.add_argument('--db-name', type=str, help='Name of the Pinecone index')
    parser.add_argument('--data-file', type=str, help='Path to the CSV data file')
    args = parser.parse_args()
    if not args.db_name or not args.data_file:
        parser.error("Missing required arguments: --db-name and --data-file")
        exit(0)

    upload_to_pinecone(args.db_name, args.data_file)
