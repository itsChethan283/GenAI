import streamlit as st
import os 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(page_title="Chat PDF", page_icon="📕")
st.title("Chat with your PDF")


if "successs_messages" not in st.session_state:
    st.session_state.success_messages = False

if st.session_state.success_messages:
    st.success("Done processing!")

if "messages" not in st.session_state:
    st.session_state.messages = []
    

if "binary_val" not in st.session_state:
    st.session_state.binary_val= []

for message in st.session_state.messages:
    if message["type"] == "question":
        st.markdown(f'''<p style="color: #000000; 
                    position: absolute; 
                    right: 0px;  
                    background-color: #ffffff; 
                    display: inline-block; 
                    border-radius: 10px;
                    padding: 5px 15px 5px 15px; 
                    text-align: left">{message["content"]}</p>''', unsafe_allow_html=True)
    if message["type"] == "answer":
        st.markdown(f'''<p style="color: #000000; 
                    position: relative;
                    margin-top: 5%;
                    margin-bottom: 5%;
                    background-color: #e0eee1; 
                    display: inline-block; 
                    border-radius: 10px;
                    padding: 5px 15px 5px 15px; 
                    text-align: left">{message["content"]}</p>''', unsafe_allow_html=True)
        

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

def indexing(chunks, embedding_model, pdf_files_list, vector_store_db):
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    vectorstore.save_local()

def chain_retrival_system(retriver, prompt, query):
    chain = RetrievalQA.from_chain_type(llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3, api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ"),
                                       retriever=retriver , chain_type="stuff", chain_type_kwargs={"prompt":prompt})
    response = chain({"query": query})
    # print("response", response)
    return response

def chatting(question, embedding_model, vector_store_db):
    with st.spinner("Processiong....."):
        vectorstore = FAISS.load_local(vector_store_db, embedding_model, allow_dangerous_deserialization=True)
        relavent_chunks = vectorstore.similarity_search(question)
        retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        prompt_template = """
        You are an AI assistant specialised in analysing and providing answers based on the specific PDF documents. 
        Be professional with the answers and ensure that all the derieved information is referenced from the document.
        If there is no context of the question in the document, don't hallucinate instead say that the provided information is not avalilabe and be patient.
        If there are any abusive words, donot answer instead make a polite statement that the information is offensive and to change the approach of asking questions. 
        The provided context is given below, only use this for your reference.
                Context: {context}
                Question: {question}
        """
        prompt = PromptTemplate.from_template(prompt_template)
        response = chain_retrival_system(retriver, prompt, question)
        # print(response["result"])
        return response["result"]

def main():
    vector_store_db= "M:/Chethan/GenAI/GenAI/GenAI/vec_store_"
    pdf_files_folder = "M:/Chethan/GenAI/GenAI/GenAI/pdf_files/"
    uploaded_files = None
    file_names = []

    with st.sidebar:
        st.title("Chat PDF")
        uploaded_files = st.file_uploader("Upload your files", type="pdf", accept_multiple_files=True)
        if uploaded_files != None:
            if not os.path.exists(pdf_files_folder):
                os.makedirs(pdf_files_folder) 
            for file in uploaded_files:
                bin = file.getvalue()
                st.session_state.binary_val.append(bin)
                file_names.append(file.name)
                with open(os.path.join(pdf_files_folder, file.name), "wb") as f:
                    f.write(file.read())


        submit = st.button("Submit")
        embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyCH-FPn68zYhVAeYfepmxt-W5O6iWMrfDQ")
        if submit and uploaded_files != None:
            with st.status("Processing.....", expanded=True) as status:
                for i, pdf_file in enumerate(file_names):
                    pdf_file = "pdf_files" + "/" + pdf_file
                    text = get_pdf_text(pdf_file)
                    st.write("Text Extracted")
                    chunks = get_chunks(text)
                    st.write("Chunked")
                    if i == 0:
                        vectorstore = FAISS.from_texts(chunks, embedding_model)
                    else:
                        vec_i = FAISS.from_texts(chunks, embedding_model)
                        vectorstore.merge_from(vec_i)   
                    vectorstore.save_local("vec_store_")
                    st.write("Embedded and Indexed")
                    status.update(label="Processing Done!", state="complete", expanded=False)
            st.session_state.success_messages = True

        
    user_ques = st.chat_input("Ask any question")
    if user_ques != None:
        res = chatting(user_ques, embedding_model, vector_store_db)
        ress = res.split("\n")
        print(ress)
        print("length", len(ress))
        st.markdown(f'''<p style="color: #000000; 
                    position: absolute; 
                    right: 0px;  
                    margin-bottom: 10px; 
                    background-color: #ffffff; 
                    display: inline-block; 
                    border-radius: 10px;
                    padding: 5px 15px 5px 15px; 
                    text-align: left">{user_ques}</p>''', unsafe_allow_html=True)
        st.session_state.messages.append({"type": "question", "content":user_ques})
        for response in ress:
            if response != "":
                st.markdown(f'''<p style="position: relative; 
                            background-color: #e0eee1; 
                            margin-top: 5%;
                            margin-bottom: 5%;
                            color: #0f1116; 
                            display: inline-block; 
                            border-radius: 10px; 
                            padding: 5px 15px 5px 15px; 
                            text-align: left">{response}</p>''', unsafe_allow_html=True)
                st.session_state.messages.append({"type": "answer", "content":response})


if __name__ == "__main__":
    main()