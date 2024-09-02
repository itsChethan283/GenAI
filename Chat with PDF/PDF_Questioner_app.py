import streamlit as st
from langchain_ollama import OllamaLLM
import time
import pymupdf
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
# import ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.llms import ollama
from langchain_ollama import ChatOllama
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


genai.configure(api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ")
st.set_page_config(page_title="Chat PDF", page_icon="📕")
st.title("Chat Bot Validator")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["type"] == "question":
        st.markdown(f'''<p style="color: #000000; 
                    position: absolute; 
                    right: 0px;  
                    background-color: #ffffff; 
                    display: inline-block; 
                    border-radius: 10px;
                    padding: 5px 15px 5px 15px; 
                    text-align: right">{message["content"]}</p>''', unsafe_allow_html=True)
    if message["type"] == "answer":
        st.markdown(f'''<p style="color: #000000; 
                    position: relative;
                    margin-top: 5%;
                    margin-bottom: 5%;
                    background-color: #e0eee1; 
                    display: inline-block; 
                    border-radius: 10px;
                    padding: 5px 15px 5px 5px; 
                    text-align: right">{message["content"]}</p>''', unsafe_allow_html=True)
    # st.write("Appending: ",message["role"], message["content"])

def empty_(var):
    var.empty()

def wait_time(secs):
    time.sleep(secs)


def display_top_message(message):
    success = st.success("Read the file successfully")
    wait_time(5)
    success.empty()
 
if 'success_message' not in st.session_state:
    st.session_state.success_message = False

if st.session_state.success_message:
    display_top_message("Success! Your action was completed successfully.")
    st.session_state.success_message = False

def get_pdf_text(pdf_path):
    text = ""
    pdf_pages = PdfReader(pdf_path)
    for page in pdf_pages.pages:
        text += page.extract_text()
    return text



def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def indexing(chunks, embedding_model):
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    temp_dir = "M:\Chethan\GenAI\GenAI\GenAI\Chat with PDF\vectors".replace("\\", "/")
    st.write(temp_dir)
    vectorstore.save_local(fr"{temp_dir}\faiss_index")
    

def conversational_chain():
    prompt_template = """Answer the question in short with the provided context: {context}\n for the question: {question}
    Answer:
    """
    # model = OllamaLLM.chat(model="llama3.1")
    # model = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          temperature=0.3, api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ")
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return "chain"


def retrival_system(retriver, prompt, query):
    # st.write("Came in retrival system")
    res = RetrievalQA.from_chain_type(llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3, api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ"),
                                       retriever=retriver , chain_type="stuff", chain_type_kwargs={"prompt":prompt})
    # st.write("completed retrival system")
    response = res({"query": query})
    return response

def get_response(query, chain):
    # st.write("came in to get response")
    response = chain({"query": query})
    # st.write("got the response")
    return chain


def chatting(question, embedding_model):
    with st.spinner("Processiong....."):
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        relavent_chunks = vectorstore.similarity_search(question)
        retriver = vectorstore.as_retriever()
        prompt_template = """Answer the question concisely, drawing on the provided context:

Context: {context}

Question: {question}
        """
        prompt = PromptTemplate.from_template(prompt_template)
        # chain = conversational_chain()
        chain = retrival_system(retriver, prompt, question)
        # res = get_response(question, chain)
        print(chain)
        # response = chain({"input_documents": relavent_chunks, "question":question}, return_only_outputs=True)
        # st.write("reply", response["output_text"])
        # st.chat_message("user").write(question)
        # st.chat_message("ai").write(res)
        return chain["result"]

# def user_query(pr):
#     ques = st.chat_input(pr)
#     return ques


def main():
    uploaded_file = None
    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Upload the PDF file", type="pdf")
        if uploaded_file == None:
            st.write("Upload a file to view the results")
        elif uploaded_file:
            st.write("File Name:", uploaded_file.name)
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.toast("File Uploaded successfully")
    
    if 'success_message' not in st.session_state:
        st.session_state.success_message = False

    if st.session_state.success_message:
        display_top_message("Success! Your action was completed successfully.")
        st.session_state.success_message = False   
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ")
    with st.sidebar: 
        submit = st.button("Submit", key="submit")
        if submit and uploaded_file != None:
            with st.spinner("Processing....."):
                text = get_pdf_text(uploaded_file)
                # st.write("text")
                chunks = get_chunks(text)
                # st.write("chunk")
                index = indexing(chunks, embedding_model)
                # st.write("index")
        elif submit and uploaded_file == None:
            st.warning("please upload the file and hit submit")
            
    # embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ")
    # user_ques = st.text_input("Ask a question", )
    # user_ques = st.chat_input("Ask a question")
    # if user_ques != st.session_state.get("last_query", ""):
    #     response = chatting(user_ques, embedding_model)
    # while True:
    res = ""
    user_ques = st.chat_input("Ask any question")
    if user_ques != None:
        # st.write(user_ques)
        res = chatting(user_ques, embedding_model)
        # with st.write():
        st.markdown(f'''<p style="color: #000000; 
                    position: absolute; 
                    right: 0px;  
                    margin-bottom: 10px; 
                    background-color: #ffffff; 
                    display: inline-block; 
                    border-radius: 10px;
                    padding: 5px 15px 5px 15px; 
                    text-align: right">{user_ques}</p>''', unsafe_allow_html=True)
        st.session_state.messages.append({"type": "question", "content":user_ques})
        # with st.chat_message():
        # st.session_state.messages.append({"answer":res})
        st.markdown(f'''<p style="position: relative; 
                    background-color: #e0eee1; 
                    margin-top: 5%;
                    margin-bottom: 5%;
                    color: #0f1116; 
                    display: inline-block; 
                    border-radius: 10px; 
                    padding: 5px 15px 5px 15px; 
                    text-align: left">{res}</p>''', unsafe_allow_html=True)
        # st.write("Ok came in")
        st.session_state.messages.append({"type": "answer", "content":res})
        # st.write(st.session_state.messages.append({"answer":res}))
            
        
        # else:
        #     user_query("Please ask a question")



if __name__ == "__main__":
    main()