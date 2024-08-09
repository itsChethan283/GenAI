from langchain.prompts import PromptTemplate
import streamlit as st
import requests
from twilio.rest import Client
from transformers import pipeline

API_TOKEN = "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"
model_id = "google/flan-t5-xxl"
end_point = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def whatsapp_message(message):
    account_sid = 'ACecd0ded9124a2b30dee4e894f1cc422c'
    auth_token = '19312865152bfc81b13673c7cfe8c793'
    client = Client(account_sid, auth_token)

    messager = client.messages.create(
        body= message,
        from_="whatsapp:+14155238886",
        to="whatsapp:+916383367510",
    )

def regenerate_func(rtemplate):
    rtemplate = "Regenerate " + rtemplate
    print(rtemplate)
    prompt = {"inputs": rtemplate}
    response = requests.post(end_point, headers=headers, json=prompt)
    return response.json()[0]["generated_text"]

def getLLMresponse(topic, no_words, blog_style, task):
    template = f"Write a blog on {topic} and it should be from a {blog_style} view with less than" + f" {no_words} words."
    prompt = {"inputs": template}
    generator = pipeline("text-generation", model = "Joshua8966/blog-writer_v31-8-2023")
    if task == "generate":
        # response = requests.post(end_point, headers=headers, json=prompt)
        response = generator(template)

    elif task == "regenerate":
        # response = regenerate_func(template)
        # return response 
        response = generator(template)
        
    elif task == "send":
        # response = requests.post(end_point, headers=headers, json=prompt)
        response = generator(template)
        whatsapp_message(response)
    
    return response

# print(getLLMresponse("Books", "50", "Common people"))

st.set_page_config(page_title="Whatsapp Blog", page_icon="⚓", layout="centered", initial_sidebar_state="collapsed")
st.header("Whatsapp Blog")

topic = st.text_input("Enter the topic in which the blog should be?")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input("Number of words")

with col2:
    blog_style = st.selectbox("Writing the Blog for ", ("Researchers", "Data Scientist", "Common People"), index=0)

g_col1, g_col2, g_col3 = st.columns([0.33, 0.33, 0.33])

display_text =""
with g_col1:
    if st.button("Generate"):
        display_text = getLLMresponse(topic, no_words, blog_style, "generate")

with g_col2:
    if st.button("Regenerate"):
        display_text = getLLMresponse(topic, no_words, blog_style, "regenerate")

with g_col3:
    if st.button("Generate and send"):
        display_text = getLLMresponse(topic, no_words, blog_style, "send")

st.write(display_text)