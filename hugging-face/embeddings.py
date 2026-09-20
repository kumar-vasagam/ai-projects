from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import create_metadata_tagger
from langchain_huggingface import HuggingFaceEndpointEmbeddings


model = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction"
)

file_path = "~/Downloads/mongodb.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
cleaned_pages = []
cleaned_docs = []
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(str.capitalize(page.page_content))
        
        
print(cleaned_pages[10])
print("--------------")
print(cleaned_pages[11])

embeddings = []
documents = ["Hello", "World", "Earth", "Super Duper", "March Madness", "Thailand"]
for doc in documents:
    cleaned_docs.append(doc)
    embeddings.append(hf.embed_documents(doc))

embeddings.append(hf.embed_documents("Hello!"))

# print("embeddings   ", embeddings)
print("embeddings length ", len(embeddings))
