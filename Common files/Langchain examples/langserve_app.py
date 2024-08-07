from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate

system_message = "Translate the following text to {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_message), ("user", "{text}")]
)

model = AzureChatOpenAI(
        api_key="",
        azure_deployment="pstestopenaidply-k6h557oar4fu6",
        azure_endpoint= "https://pstestopenaidply-k6h557oar4fu6.openai.azure.com/",
        openai_api_version="2024-02-01",                            ## Find the api version in chat playground "view code"
    )

parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(
    title="Translation API",
    version="1.0",
    description="Translate text fromm english to the desired language"
)

add_routes(app, chain, path="/chain",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)