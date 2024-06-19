from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json
import os

vals = json.loads(open("config.json").read())

os.environ["AZURE_OPENAI_API_KEY"] = vals["api_key"]
os.environ["AZURE_OPENAI_ENDPOINT"] = vals["endpoint"]

model = AzureChatOpenAI(
    openai_api_version = "2023-06-01",
    azure_deployment="pstestopenaidply-y377hh3qrrryi",

)

message = HumanMessage(
    content= "What is the biggest thing to achieve in the universe"
)

model.invoke([message])