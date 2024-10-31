from dotenv import load_dotenv
import fitz
import streamlit as st
from langchain import hub
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAI
from vector_store import VectorStore

load_dotenv(override=True)

def setup_vector_store():
    try:
        vector_store = VectorStore()
        return vector_store
    except Exception as e:
        st.write(f"Error setting up vector store: {e}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

vector_store = setup_vector_store()
#rag_system = RAGSystem(vector_store)

st.title("GDG-Hacktoberfest RAG-bot")

st.sidebar.title("PDF Management")
uploaded_files = st.sidebar.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)

pdf_names = []
if uploaded_files:
    documents = [{"text": extract_text_from_pdf(file)} for file in uploaded_files]
    store_response = vector_store.store_documents(documents)
    print(store_response)
    st.sidebar.write(f"{store_response['stored_count']} chunks stored from {store_response['original_count']} uploaded PDFs.")

prompt = hub.pull("rlm/rag-prompt")

st.write("### Ask a question based on the uploaded documents:")
query = st.text_input("Enter your question:", "")


llm = OpenAI()

vectorstore = Qdrant(
        client=vector_store.qdrant_client,
        collection_name=vector_store.collection_name,
        embeddings=vector_store.embeddings,
)

qa_chain = (
    {
#        "context": vector_store.db_client.as_retriever() | format_docs,
        "context": vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

st.write(qa_chain.invoke(query))

