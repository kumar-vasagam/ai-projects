from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import create_metadata_tagger
from langchain_huggingface import HuggingFaceEndpointEmbeddings


llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=1.0
)

chat = ChatHuggingFace(llm=llm, verbose=True)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to Tamil."),
    ("human", "I love to read"),
]

response = chat.invoke(messages)
print(response.content)
print(response.response_metadata)

print("---------")
messages = [
    ("system", "You are a great personal finance specialist based in NYC"),
    ("human", "Advise a 23 year old living in NYC to get rich"),
]
res = chat.invoke(messages)
print(res.content)
print(res.response_metadata)
