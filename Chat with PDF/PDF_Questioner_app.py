import streamlit as st
from langchain_ollama import OllamaLLM
import time
import pymupdf
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import *

st.title("Chat Bot Validator")

def progress_bar(percentage, bar):
    percentage+=20
    bar.progress(percentage, text=progress_message)
    return percentage

def empty_(var):
    var.empty()

def wait_time(secs):
    time.sleep(secs)

def get_pdf_text(pdf_path, percentage):
    doc = PdfReader(pdf_path)
    percentage = progress_bar(percentage, bar)
    text = ""
    count = 0
    for page in doc.pages:
        text += page.extract_text()
        count += 1
    percentage = progress_bar(percentage, bar)
    return text, count, percentage

def display_top_message(message):
    success = st.success("Read the file successfully")
    wait_time(5)
    success.empty()

percentage = 0
progress_message = "Reading the file"


temp_file_path = ""
pdf_text = ""
uploaded_file = None
with st.sidebar:
    uploaded_file = st.file_uploader("Upload the PDF file", type="pdf")
    if uploaded_file == None:
        st.write("Upload a file to view the results")
    elif uploaded_file:
        bar = st.progress(0, progress_message)
        st.write("File Name:", uploaded_file.name)
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.toast("File Uploaded successfully")
        pdf_name = uploaded_file.name
        percentage = progress_bar(percentage, bar)
        pdf_text, count, percentage = get_pdf_text(pdf_name, percentage)
        percentage = progress_bar(percentage, bar)
        st.write(f"No. of Pages: :blue[{count}]")
        # bg_color = st.color_picker('选择文字颜色', '#ffffff',key=3)
        # st.write(f"你选择了{bg_color}")
        # text_color = st.color_picker('选择文字颜色', '#ffffff',key=1)
        # st.write(f"你选择了{text_color}")
        # col1, col2 = st.columns(2)
        # col1 = st.write("yuik")
        # col2 = st.write(f'<p style="background-color:#587a87;color:#b8c2c9;font-size:24px;border-radius:2%; text-align:center; background: linear-gradient(80deg, #ff0000 20%, #0000ff 80%);">{count}</p>', unsafe_allow_html=True)

# sub_but = st.button("Progress")
# if sub_but:

# percentage = progress_bar(percentage, bar)
# percentage = progress_bar(percentage, bar)
# percentage = progress_bar(percentage, bar)
# percentage = progress_bar(percentage, bar)
# if percentage == 100:
#     bar.empty()

# Initialize state variable
if 'success_message' not in st.session_state:
    st.session_state.success_message = False

# Display success message if state variable is True
if st.session_state.success_message:
    display_top_message("Success! Your action was completed successfully.")
    st.session_state.success_message = False  # Clear the message after display

pdf, chat = st.columns(2)
with pdf:
    if pdf_text:
        # display_text = st.write(pdf_text)
        display_text = pdf_viewer(input=uploaded_file.getvalue())
        percentage = progress_bar(percentage, bar)
        if percentage == 100:
            bar.empty()
            st.toast("Read the file successfully")
            if display_text != "":
                st.session_state.success_message = True

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with chat:
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
