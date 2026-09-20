import os
import getpass
from langchain_core.documents import Document
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
embedding_model="nvidia/nemotron-3-embed-1b:free"

documents = [Document(page_content="Dogs are great companion, they are known for their loyalty and friendliness", 
                      metadata={"source":"mammal-pets-dog-doc"}),

            Document(page_content="Cats are independent pers that often enjoy their freedom",
                     metadata={"source":"mammal-pets-cats-doc"})]

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

cer = client.embeddings.create(model=embedding_model, input=documents[0].page_content)
print(cer.model_dump_json())
cer = client.embeddings.create(model=embedding_model, input=documents[1].page_content)
print(cer.model_dump_json())
