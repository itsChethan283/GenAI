# # import pywhatkit

# # phone_no = "+916383367510"
# # message = "What's up"

# # pywhatkit.sendwhatmsg_instantly(phone_no, message)


# from langchain.prompts import PromptTemplate
# import streamlit as st
# from langchain_community.llms import CTransformers
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from huggingface_hub import InferenceClient


# def getLLMresponse(input_text, no_words, blog_style):
    
# #     client = InferenceClient("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")
# #     for message in client.chat_completion(
# # 	messages=[{"role": "user", "content": "What is the capital of France?"}],
# # 	max_tokens=500,
# # 	stream=True,
# # ):print(message.choices[0].delta.content, end="")


#     import requests
#     API_URL = "https://api-inference.huggingface.co/models/gpt2"
#     API_TOKEN = "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"
#     headers = {"Authorization": f"Bearer {API_TOKEN}"}
#     def query(payload):
#         response = requests.post(API_URL, headers=headers, json=payload)
#         return response
#     data = query("Can you please let us know more details about your ")
#     # llm = CTransformers(model="C:/Users/cs25/OneDrive - Capgemini/GenAI/Simple RAG/simplerag/End To End LLM Project Using LLAMA 2- Open Source LLM Model From Meta/llama-2-7b-chat.ggmlv3.q8_0.bin",
#     #                     model_type="llama",
#     #                     config={"max_new_tokens":256,
#     #                             "temperature":0.01})
    
#     # template = """
#     #             Write a blog on {input_text} with {no_words} and it should be from a {blog_style} view. Make it crisp and to the point.
#     #             """
#     # prompt = PromptTemplate.from_template(template=template)

#     # user_inp = "write a blog on {input_text} with {no_words}.".format(input_text=input_text, no_words=no_words)
#     # # prompt = ChatPromptTemplate.from_messages([("system", "You are a {blog_style}"), ("user", "{user_inp}")])


#     # # prompt = PromptTemplate(input_variables=["input_text", "no_words", "blog_style"],
#     #                         # template=template)
    
#     # # prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
#     # # chain = prompt | llm
    
#     # output_parser = StrOutputParser()
#     # chain = prompt | llm | output_parser
#     # response = chain.invoke({"blog_style":blog_style, "input_text": input_text, "no_words":no_words})
#     # print(prompt, response)
#     return None

# getLLMresponse("MAchine learning", "50", "common people")

# import requests

# url = "https://api-inference.huggingface.co/models/nisten/Biggie-SmoLlm-0.15B-Base"
# # url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct/v1/chat/completions"
# token = "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"

# prompt = "What is the captial of india"
# headers = {"Authorization": f"Bearer {token}"}

# def quer(payload):
#     response = requests.post(url, headers=headers, json=payload)
#     return response.json()

# output = quer(prompt)
# print(output)

import requests

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM-135M"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output[0]["generated_text"])