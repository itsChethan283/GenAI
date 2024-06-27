# 127.0.0.1:8000/openai/playground
# 127.0.0.1:8000/docs
"""
pip install "langserve[all]"
pip install fastapi
pip install langserve
pip install sse_starlette
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from dotenv import dotenv_values

config = dotenv_values(".env")
openai_api_key=config.get("OPENAI_API_KEY")
llm = ChatOpenAI(api_key="sk-proj-reWYfEUnvVdSs0BBtg32T3BlbkFJmRwRawDbtRKRlUHeQF4Y")

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


prompt = ChatPromptTemplate.from_template("Tell me a short story about {topic}")
chain = prompt | llm

add_routes(app, chain, path="/openai")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)