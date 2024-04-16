from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

import csv 
import sys
import argparse
import time

def create_faiss_db(db_name: str, data_file: str):    
    print("Getting hf Embeddings...")
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'batch_size': 8}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    print("Loading CSV...")
    csv.field_size_limit(sys.maxsize)
    loader = CSVLoader(file_path=data_file,  csv_args = {
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['text', 'url', 'depth', 'title', 'model_num', 'ps_num']},
        source_column='url',
        metadata_columns=['depth', 'title', 'model_num', 'ps_num']) 
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    print("Creating FAISS Index from documents...")
    start_time = time.time()
    total_length = len(docs)
    batch_size = 1000
    db = None 
    for batch_start in range(0, total_length, batch_size):
        batch_end = min(batch_start + batch_size, total_length)
        batch_texts = docs[batch_start:batch_end]
        if db is None:
            db = FAISS.from_documents(batch_texts, embeddings)
        else: 
            db.add_documents(documents=batch_texts, embeddings=embeddings)
        print(f"Inserted {batch_end}/{total_length} docs")
    print(f"Completed inserting {batch_start} docs")
    print(f"FAISS Index created in {time.time() - start_time} seconds")

    print("Saving FAISS Index...")
    db.save_local(db_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI for adding CSV data to FAISS index')
    parser.add_argument('--db-name', type=str, help='Name of the FAISS index')
    parser.add_argument('--data-file', type=str, help='Path to the CSV data file')
    args = parser.parse_args()
    if not args.db_name or not args.data_file:
        parser.error("Missing required arguments: --db-name and --data-file")
        exit(0)

    create_faiss_db(args.db_name, args.data_file)