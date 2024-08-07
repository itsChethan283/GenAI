from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

embeddings_model = OpenAIEmbeddings(api_key="")
embeddings_model_cohere = CohereEmbeddings(cohere_api_key="")

embeddings_doc = embeddings_model_cohere.embed_documents(["Hello, world!", "Goodbye, world!"])

embeddings_sent = embeddings_model_cohere.embed_query("Hello, world!")

print(embeddings_sent, len(embeddings_sent))