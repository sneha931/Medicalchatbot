
from flask import Flask, render_template, jsonify, request
from weaviate import connect_to_weaviate_cloud
from src.helpers import download_hugging_face_embeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)


load_dotenv()

weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") 
weaviate_url = os.getenv("WEAVIATE_URL")


os.environ["WEAVIATE_API_KEY"] = weaviate_api_key
os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
os.environ["WEAVIATE_URL"] = weaviate_url

embeddings = download_hugging_face_embeddings()



client = connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),          
    
    auth_credentials=os.getenv("WEAVIATE_API_KEY")  
    
)

docsearch = WeaviateVectorStore(
    client=client,
    index_name="MedicalChatbot",
    text_key="text", 
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chat_model = ChatOpenAI(
    model="mistralai/mistral-nemo:free",  
    temperature=0.2,
    max_tokens=1024,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_API_BASE")
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_query = msg.strip()

    
    classify_prompt = f"""
    You are a classifier. Determine if this question is related to medicine, health, diseases, drugs, or treatments.
    If yes, reply with "medical".
    If not, reply with "general".
    Question: "{user_query}"
    """

    classification = chat_model.invoke([
        {"role": "user", "content": classify_prompt}
    ])
    query_type = classification.content.strip().lower()

    print(f"🩺 Detected query type: {query_type}")

   
    if "medical" in query_type:
        response = rag_chain.invoke({"input": user_query})
        answer = response.get("answer") or response.get("result") or "I'm sorry, I couldn't find that."
        print("Response:", answer)
        return str(answer)

    else:
        casual_prompt = f"""
        You are a friendly assistant. Respond casually and conversationally.
        User: "{user_query}"
        """
        response = chat_model.invoke([{"role": "user", "content": casual_prompt}])
        print("Casual response:", response.content)
        return response.content


if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 8080, debug= True)

