from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

os.environ['PINECONE_API_KEY'] = ''

loader = TextLoader("C:/Users/cs25/OneDrive - Capgemini/GenAI/Langchain/office lap/office_lap/state_of_the_union.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents) 

embeddings = CohereEmbeddings(cohere_api_key="")

index_name = "same"
# PineconeVectorStore(pinecone_api_key="")

doc_search = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

query = "What did the president say about Ketanji Brown Jackson"
response = doc_search.similarity_search(query)
print(response[0].page_content)