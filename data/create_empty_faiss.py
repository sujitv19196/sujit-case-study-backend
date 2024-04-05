from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
import csv 
import sys


def create_faiss_db(db_name: str):
    # Define the texts you want to add to the FAISS instance
    texts = ["Init", "Test"]

    print("Getting OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_texts(texts, embeddings)

    print("Saving FAISS Index...")
    db.save_local(db_name)


if __name__ == "__main__":
    db_name = input("Enter the name of the FAISS database: ")
    create_faiss_db(db_name)