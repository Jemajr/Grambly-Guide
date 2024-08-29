import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from vector2 import vectordb  # Import vector database

  
# Load environment variables
load_dotenv('grambly.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Verify the API key is loaded
print(f"API key loaded: {'Yes' if openai.api_key else 'No'}")

app = Flask(__name__, template_folder='templates')
 
# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=1.5)
 
# Create a memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Build prompt
template = """You are Grambly Guide, an assistant for international students coming to Grambling State University. Use the following pieces of context (They are a set of questions and answers that new students to grambling may ask) to answer the question at the end. If you don't find the answer in the provided context, check if it can be found in your training data (you can also combine both the context and your training data if it will give a better and more precise overall answer), else, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Stay polite. Only respond in english. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

@app.route('/')
def home():
    return render_template('new.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json['question']
    #user_question = 'how are you?'
    #print(user_question)
    # Get the response from the QA chain
    result = qa_chain.invoke({"question": user_question})
    
    # Extract the answer and update the memory
    answer = result['answer']
    #print(answer)
    memory.chat_memory.add_user_message(user_question)
    memory.chat_memory.add_ai_message(answer)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
