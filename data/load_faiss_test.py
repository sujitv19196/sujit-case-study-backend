from langchain_community.utilities import ApifyWrapper
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
import os

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
query = "What kind of parts does this website have information on?"
docs = db.similarity_search_with_score(query)
print(docs[0:5])