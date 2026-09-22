# Use langchain & openrouter with milvus integration
#wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
from langchain_milvus import Milvus
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
import pypdf
from langchain_core.documents import Document


URI = "./demo.db"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Load Part
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)
vector_store = Milvus(embedding_function=embeddings, collection_name="example", connection_args={"uri":URI},  
                      drop_old=True, text_field="content")
files = ["../data/lyft_2021.pdf", "../data/uber_2021.pdf"]
data = []
for x, val in enumerate(files):
    file_path = val
    reader = pypdf.PdfReader(file_path)
    documents = []
    page_numbers = []
    for i, page in enumerate(reader.pages):
        page_content = page.extract_text()
        documents.append(Document(page_content=page_content, metadata={"page_num":i, "doc_owner":file_path}))
        page_numbers.append(str(i))
    vector_store.add_documents(documents=documents, ids=page_numbers)
print("file loaded okay, hopefully...")

# Query Part
results = vector_store.similarity_search(query="provide a comparison between Lyft and Uber's total revenues in 2021")
for idx, doc in enumerate (results):
    print(f"Result {idx+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("\n")