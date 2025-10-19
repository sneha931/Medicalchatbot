from dotenv import load_dotenv
import os
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

from weaviate import connect_to_weaviate_cloud
from weaviate.classes.config import Property, DataType
from langchain_weaviate import WeaviateVectorStore

load_dotenv()


extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

weaviate_api_key = os.getenv("weaviate_api_key")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") 
weaviate_url = os.getenv("weaviate_url")

os.environ["WEAVIATE_API_KEY"] = weaviate_api_key
os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
os.environ["WEAVIATE_URL"] = weaviate_url


client = connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),          
    
    auth_credentials=os.getenv("WEAVIATE_API_KEY")  
    
)




collection_name = "MedicalChatbot"


existing_collections = client.collections.list_all()


if collection_name not in existing_collections:
    client.collections.create(
        name=collection_name,
        vectorizer_config=None,  
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT)
        ]
    )
    
else:
    print(f"ℹ️ Collection '{collection_name}' already exists")

collection = client.collections.get(collection_name)


docsearch = WeaviateVectorStore.from_documents(
    client=client,
    documents=text_chunks,
    embedding=embeddings,
    index_name="MedicalChatbot"
)




docsearch = WeaviateVectorStore(
    client=client,
    index_name="MedicalChatbot",
    text_key="text", 
    embedding=embeddings
)


