from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

raw_doc = TextLoader("C:/Users/cs25/OneDrive - Capgemini/GenAI/Simple RAG/simplerag/Sample.txt")
loaded_doc = raw_doc.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(loaded_doc)

# embedding_model = CohereEmbeddings(cohere_api_key="")
# embeddings = embedding_model.embed(split_docs)

embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = embedding_model.encode(split_docs)

print(embeddings)