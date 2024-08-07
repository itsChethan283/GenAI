from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant

print("1Yes")
raw_docs = TextLoader("C:/Users/cs25/OneDrive - Capgemini/GenAI/Langchain/office lap/office_lap/state_of_the_union.txt",  encoding='utf-8')
loaded = raw_docs.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(loaded)
db = Chroma.from_documents(docs, CohereEmbeddings(cohere_api_key=""))

query = "can you give the line where crystal word is present"
response = db.similarity_search(query)
print(response)
print("2Yes")

response_vec = db.similarity_search_by_vector(query)
# print(response_vec)
print("3Yes")

db = Qdrant.from_documents(docs, CohereEmbeddings(cohere_api_key=""), "http://localhost:6333")
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)