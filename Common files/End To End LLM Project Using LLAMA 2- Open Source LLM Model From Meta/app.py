from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient


def getLLMresponse(input_text, no_words, blog_style):


    llm = CTransformers(model="C:/Users/cs25/OneDrive - Capgemini/GenAI/Simple RAG/simplerag/End To End LLM Project Using LLAMA 2- Open Source LLM Model From Meta/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={"max_new_tokens":256,
                                "temperature":0.01})
    
    template = """
                Write a blog on {input_text} and it should be from a {blog_style} view. Make it crispa nd to the point. Please let this be in one paragraph and try to use less than {no_words} words.
                """
    prompt = PromptTemplate.from_template(template=template)
    # prompt = PromptTemplate(input_variables=["input_text", "no_words", "blog_style"],
    #                         template=template)
    
    # response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"blog_style":blog_style, "input_text": input_text, "no_words":no_words})
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon="⚓",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Generate Daily Blogs ⚓")

input_text = st.text_input("Enter the Blog topic")

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No. of words")

with col2:
    blog_style = st.selectbox("Writing the Blog for ", ("Researchers", "Data Scientist", "Common People"), index=0)

submit = st.button("Generate")

if submit:
    st.write(getLLMresponse(input_text, no_words, blog_style))