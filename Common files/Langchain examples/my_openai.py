import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage

os.environ["OPENAI_API_KEY"] = ""   

llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-35-turbo", api_key="")
text = "What would be good company name for a company selling microgreens in bangalore and it should be in kannada and tamil"
messages = [HumanMessage(content=text)]

ll = llm.invoke(messages)
chat_model.invoke(messages)

print(ll)