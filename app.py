import os
import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Setup App
st.title("📄 Conversational PDF Assistant With Chat History")
st.write("📂 Upload your PDFs and chat seamlessly with their content.")

## Input Groq API key 
api_key = st.text_input("🔑 Enter Groq API Key:", type="password")

## Check if API key is valid
if api_key: 
    llm = ChatGroq(model="Llama3-8b-8192", groq_api_key = api_key)
    
    session_id = st.text_input("Session ID", value="default")
    ## Manage Chat history
    if "store" not in st.session_state:
        st.session_state.store={}
        
    uploaded_files = st.file_uploader("📄 Upload PDF file(s)", type="pdf", accept_multiple_files=True)
    ##Process Uploaded Files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
                    
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
            
        ## Spliting and creating embedding for documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=400)
        splits = text_splitter.split_documents(documents)
        
        # Create a persistent directory for ChromaDB
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        vector_db = Chroma.from_documents(
            splits, 
            embedding=embeddings,
            persist_directory=persist_directory
        )
        retriever = vector_db.as_retriever()
        
        ### System Prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the lastest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat, Do not answer the question."
            "just reformulate it if needed and other wise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
        
        ## Answer Question prompt
        system_prompt = (
            "You are an assistant for question answering task"
            "use the following peices of retriever context to answer"
            "the question. you dont know the answer, say that you"
            "don't know. use three sentences maxmmium and keep the"
            "answer concise"
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (MessagesPlaceholder("chat_history")),
                ("human","{input}")
            ]
        )
        
        ### RAG Chain
        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
                
            return st.session_state.store[session_id]
        
        converstional_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key= "input",
            history_messages_key= "chat_history",
            output_messages_key="answer",
        )
        
        user_input = st.text_input("💬 Ask your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = converstional_rag_chain.invoke(
                {"input": user_input},
                config = {
                    "configurable":{"session_id":session_id}
                    
                }
            )
            
            st.write(st.session_state.store)
            st.write("Assistant: ",response["answer"])
            st.write("Chat History:",session_history.messages)
            
else:
    st.warning("⚠️ Please enter a valid Groq API key.")