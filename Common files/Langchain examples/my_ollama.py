from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

llm = Ollama(model="llama2")
chatmodel = ChatOllama()

text = "What would be good company name for a company selling microgreens in bangalore and it should be in kannada and tamil"
messages = [HumanMessage(content=text)]

ll = llm.invoke(messages)
chat = chatmodel.invoke(messages)