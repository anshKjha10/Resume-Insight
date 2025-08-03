import streamlit as st
import fitz  # PyMuPDF
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import sys, pathlib, pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

st.set_page_config(page_title="Resume Reader Assistant", page_icon=":book:")
st.title("Resume Insight!")
st.write("Upload your resume to get insights!")

llm = ChatGroq(groq_api_key = groq_api_key, model = "gemma2-9b-it")

session_id = "default_session"
if 'store' not in st.session_state:
    st.session_state.store = {}

document = st.file_uploader("Upload your resume in pdf format", type = "pdf")

if document is not None:
    # Convert file-like object into bytes for pymupdf
    with fitz.open(stream=document.read(), filetype="pdf") as doc:
        text = "\n\n".join([page.get_text() for page in doc])

    # Show extracted text (optional for debug)
    with st.expander("Extracted Text from Resume (click to expand)", expanded=False):
        st.write(text)

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(
        chunks,
        embeddings,
    )
    
    retriever = vector_store.as_retriever()
    
    contextualize_q_system_prompt = (
        """
        Given the chat history and a follow-up question, rewrite the follow-up question to be a standalone question.
        Assume the context is based on a resume uploaded by the user.
        Do not answer the question — only rewrite it if needed.
        """
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
     
    resume_prompt =(
    """
    You are a resume expert.
    Given the following resume text, give:
    1. A score out of 10 based on formatting, relevance, clarity, and impact.
    2. 3 key suggestions to improve this resume.
    
    Resume Text:
    {context}
    
    """
    )
    
    resume_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", resume_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    resume_chain = create_stuff_documents_chain(
        llm = llm,
        prompt = resume_prompt_template
    )
    
    rag_chain = create_retrieval_chain(history_aware_retriever, resume_chain)
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if "store" not in st.session_state:
            st.session_state.store = {}
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    def go():
        response = conversational_rag_chain.invoke(
            {"input": text},  
            config={"configurable": {"session_id": session_id}}
        )
        analysis_result = response["answer"]
        st.session_state["analysis_result"] = analysis_result
        st.session_state["analyzed"] = True

    if st.button("Analyze Resume"):
        go()
        
    if st.session_state.get("analyzed", False):
        st.subheader("📊 Resume Analysis Result:")
        st.write(st.session_state["analysis_result"])

        st.subheader("💬 Ask any question, if you want:")
        user_query = st.text_input("What would you like to know?", key="resume_q")

        if user_query:
            followup_response = conversational_rag_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(f"**Answer:** {followup_response['answer']}")
        
else:
    st.warning("Please upload a PDF resume to get insights.")        