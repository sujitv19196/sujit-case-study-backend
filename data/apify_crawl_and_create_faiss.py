from langchain_community.utilities import ApifyWrapper
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
import os

#Set up your Apify API token and OpenAI API key
# os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"
# os.environ["APIFY_API_TOKEN"] = "Your Apify API token"

apify = ApifyWrapper()

#Run the Website Content Crawler on a website, wait for it to finish, and save
#its results into a LangChain document loader:
print("Running the Website Content Crawler...")
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://www.partselect.com/"}], 
               "maxCrawlDepth": 7},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)
print("Website Content Crawler finished.")

print("Loading documents...")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("Getting OpenAI Embeddings...")
embeddings = OpenAIEmbeddings()

print("FAISS Indexing...")
db = FAISS.from_documents(docs, embeddings)

print("Saving FAISS Index...")
db.save_local("faiss_index")