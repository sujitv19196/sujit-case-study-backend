import argparse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
import csv 
import sys
import math


def main():
    parser = argparse.ArgumentParser(description='CLI for adding CSV data to FAISS index')
    parser.add_argument('--db-name', type=str, help='Name of the FAISS index')
    parser.add_argument('--data-file', type=str, help='Path to the CSV data file')
    parser.add_argument('--num-docs', type=int, help='Number of docs to add to the FAISS index', default= math.inf)
    parser.add_argument('--resume-point', type=int, help='Number of docs to add to the FAISS index')
    args = parser.parse_args()

    if not args.db_name or not args.data_file:
        parser.error("Missing required arguments: --db-name and --data-file")


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

    total_length = len(docs)
    batch_size = 500 
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
    print(f"Completed inserting {total_length} docs")

    print("Saving FAISS Index...")
    db.save_local(args.db_name)

    print("Resume point: ", batch_start)

if __name__ == "__main__":
    main()
