from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_faiss_db(db_name: str, embeddings_model_name: str):    
    embeddings = None
    if embeddings_model_name == "hf":
        print("Getting hf Embeddings...")
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'mps'}
        encode_kwargs = {'batch_size': 8}
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    elif embeddings_model_name == "openai":
        print("Getting OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    else: 
        print("Invalid embeddings model name")
        return
    
    texts = ["Init", "Test"]
    db = FAISS.from_texts(texts, embeddings)

    print("Saving FAISS Index...")
    db.save_local(db_name)
    

if __name__ == "__main__":
    db_name = input("Enter the name of the FAISS database: ")
    embed = input("Enter the name of the embedings: ")
    create_faiss_db(db_name, embed)