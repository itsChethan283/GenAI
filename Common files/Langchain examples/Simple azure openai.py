from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json
import os

vals = json.loads(open("config.json").read())

<<<<<<< HEAD
os.environ["AZURE_OPENAI_API_KEY"] = vals["api_key"]
os.environ["AZURE_OPENAI_ENDPOINT"] = vals["endpoint"]

model = AzureChatOpenAI(
    openai_api_version = "2023-06-01",
    azure_deployment="pstestopenaidply-y377hh3qrrryi",

=======
# os.environ["AZURE_OPENAI_API_KEY"] = vals["api_key"]
# os.environ["AZURE_OPENAI_ENDPOINT"] = vals["endpoint"]

# model = AzureChatOpenAI(
#     deployment_na
# )   
model = AzureChatOpenAI(
    deployment_name="pstestopenaidply-dbkv3axne6hag",
    openai_api_version="2023-07-01-preview",
    openai_api_key="",
    azure_endpoint="https://https://pstestopenaidply-dbkv3axne6hag.openai.azure.com/openai.openai.azure.com/openai",
>>>>>>> 53b9adb9acf15d5cf37e1944e29b73f127d3cb2b
)

message = HumanMessage(
    content= "What is the biggest thing to achieve in the universe"
)

model.invoke([message])