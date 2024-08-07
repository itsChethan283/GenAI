from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient
import requests
import ollama

API_TOKEN = "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"
model_id = "HuggingFaceTB/SmolLM-135M"
end_point = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def getLLMresponse(input_text, no_words, blog_style):
    template = f"""Write a blog on {input_text} and it should be from a {blog_style} view. Make it crispa nd to the point. Please let this be in one paragraph and try to use less than {no_words} words."""
    prompt = {"inputs": template}
    response = requests.post(end_point, headers=headers, json=prompt)
    
    return response.json()[0]["generated_text"]
print(getLLMresponse("wef", "50", "sdf"))

# st.set_page_config(page_title="Generate Blogs",
#                    page_icon="⚓",
#                    layout="centered",
#                    initial_sidebar_state="collapsed")

# st.header("Generate Daily Blogs ⚓")

# input_text = st.text_input("Enter the Blog topic")

# col1, col2 = st.columns([5,5])

# with col1:
#     no_words = st.text_input("No. of words")

# with col2:
#     blog_style = st.selectbox("Writing the Blog for ", ("Researchers", "Data Scientist", "Common People"), index=0)

# submit = st.button("Generate")

# if submit:
#     st.write(getLLMresponse(input_text, no_words, blog_style))