from dotenv import load_dotenv
import os
from pinecone import Pinecone 
from pinecone import ServerlessSpec 
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
from langchain_pinecone import PineconeVectorStore
import src.helper
print(dir(src.helper))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_files(r"C:\Users\deepa\OneDrive\Desktop\Medical Chat bot\MLChatBot\data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)