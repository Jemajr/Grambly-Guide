import os
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
 
# Load environment variables
load_dotenv('grambly.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Verify the API key is loaded (optional)
print(f"API key loaded: {'Yes' if openai.api_key else 'No'}")

# Initialize the embedding function
embedding = OpenAIEmbeddings()
 
# Load and process the document
loader = Docx2txtLoader("STEP2.docx")
data = loader.load()

# Split the text into chunks
size = 500
overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=size,
    chunk_overlap=overlap,
    separators="\n"
)
questions = text_splitter.split_documents(data)

# Set up the Chroma vector database
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
persist_directory = os.path.join(notebook_dir, 'vector_db')
os.makedirs(persist_directory, exist_ok=True)
print(f"Vector database will be stored in: {persist_directory}")

# Create or load the vector database
vectordb = Chroma.from_documents(
    documents=questions,
    embedding=embedding,
    persist_directory=persist_directory
)


if __name__ == "__main__":
    # testing code
    print(f"Vector database created with {len(questions)} chunks.")
    #using the vectordb
    query = "What is Grambling State University?"
    results = vectordb.similarity_search(query, k=2)
    print(f"Sample query results: {results}")
