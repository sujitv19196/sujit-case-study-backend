import argparse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

import csv 
import sys
import math
import time 


def main():
    parser = argparse.ArgumentParser(description='CLI for adding CSV data to FAISS index')
    parser.add_argument('--db-name', type=str, help='Name of the FAISS index')
    parser.add_argument('--data-file', type=str, help='Path to the CSV data file')
    parser.add_argument('--num-docs', type=int, help='Number of docs to add to the FAISS index', default= math.inf)
    parser.add_argument('--embeddings-model', type=str, help='Embeddings model to use')
    parser.add_argument('--resume-point', type=int, help='Number of docs to add to the FAISS index')
    args = parser.parse_args()

    if not args.db_name or not args.data_file or not args.embeddings_model:
        parser.error("Missing required arguments: --db-name and --data-file and --embeddings-model")

    if args.embeddings_model == "hf":
        print("Getting hf Embeddings...")
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'mps'}
        encode_kwargs = {'batch_size': 8}
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    elif args.embeddings_model == "openai":
        print("Getting OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    print("Loading FAISS Index...")
    db = FAISS.load_local(args.db_name, embeddings, allow_dangerous_deserialization=True)

    print("Loading CSV...")
    csv.field_size_limit(sys.maxsize)
    loader = CSVLoader(file_path=args.data_file,  csv_args = {
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['text', 'url']})
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    start_time = time.time()
    if args.embeddings_model == "openai":
        total_length = len(docs)
        batch_size = 250 
        docs_inserted = 0
        start = 0
        if args.resume_point:
            start = args.resume_point
        for batch_start in range(start, total_length, batch_size):
            if docs_inserted >= args.num_docs:
                break
            batch_end = min(batch_start + batch_size, total_length)
            batch_texts = docs[batch_start:batch_end]
            db.add_documents(documents=batch_texts, embeddings=embeddings)
            docs_inserted += len(batch_texts)
            print(f"Inserted {batch_end}/{total_length} docs")
        print(f"Completed inserting {batch_start} docs")
        print("Resume point: ", batch_start)
    else: 
        print("Inserting all docs using hf embeddings...")
        db.add_documents(documents=docs, embeddings=embeddings)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    print("Saving FAISS Index...")
    db.save_local(args.db_name)

if __name__ == "__main__":
    main()
