import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationBufferMemory
from rag.generator import load_llm, guardrails
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Load environment variables
load_dotenv()

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm TherapyBot, a mental health assistant. How may I help you today?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
if 'llm' not in st.session_state:
    try:
        st.session_state['llm'] = load_llm()
    except ValueError as e:
        st.error(str(e))
        st.stop()

# Set page config
st.set_page_config(
    page_title="TherapyBot - Mental Health Assistant",
    page_icon="🤖",
    layout="wide"
)

# Sidebar contents
with st.sidebar:
    st.title('🤖 TherapyBot')
    st.markdown('''
    ## About
    TherapyBot is an AI-powered mental health assistant built using:
    - Streamlit
    - LangChain
    - HuggingFace Models
    - RAG Technology
    
    ⚠️ **Disclaimer**: This is not a replacement for professional mental health care.
    If you're experiencing a mental health emergency, please contact emergency services
    or a mental health crisis hotline immediately.
    
    **Emergency Resources:**
    - National Crisis Hotline: 988
    - Crisis Text Line: Text HOME to 741741
    ''')
    add_vertical_space(5)

# Load the document
@st.cache_resource
def process_pdf_document(pdf_path):
    """Process the PDF document and return necessary components for RAG"""
    if not os.path.exists(pdf_path):
        st.error(f"Error: Document not found at {pdf_path}")
        st.info("Please place your mental health guide PDF in the data/documents directory.")
        return None, None, None
        
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    return documents, parent_splitter, child_splitter

@st.cache_resource
def create_vectorstore(_embeddings):
    """Create and return the vector store components"""
    vectorstore = Chroma(embedding_function=_embeddings)
    store = InMemoryStore()
    return vectorstore, store

@st.cache_resource
def initialize_rag():
    """Initialize RAG components with caching"""
    documents, parent_splitter, child_splitter = process_pdf_document('data/documents/mental_health_guide.pdf')
    if not documents:
        return None
        
    vectorstore, store = create_vectorstore(st.session_state['embeddings'])
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(documents)
    return retriever

# Initialize RAG components
try:
    retriever = initialize_rag()
except Exception as e:
    st.error(f"Error initializing RAG system: {str(e)}")
    retriever = None

# Setup chat interface
st.title("💭 Chat with TherapyBot")
st.write("I'm here to provide information and support about mental health topics. Remember, I'm not a replacement for professional help.")

input_container = st.container()
colored_header(label='Chat History', description='', color_name='blue-30')
response_container = st.container()

# User input
def get_text():
    input_text = st.text_input(
        "Your message:",
        "",
        key="input",
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )
    return input_text

# Response generation
@st.cache_data(show_spinner=False)
def generate_response(user_input):
    if not guardrails(user_input):
        return "I apologize, but I cannot assist with content related to self-harm or harm to others. Please seek professional help if you're having such thoughts. Here are some resources:\n\n" + \
               "- National Crisis Hotline: 988\n" + \
               "- Crisis Text Line: Text HOME to 741741"
    
    try:
        # Use RAG for response generation
        if retriever:
            context_docs = retriever.get_relevant_documents(user_input)
            context = "\n".join([doc.page_content for doc in context_docs])
        else:
            context = ""
        
        response = st.session_state['llm'].predict(
            f"Context: {context}\n\nUser: {user_input}\n\nAssistant: Let me help you with that. Remember to be empathetic and supportive while providing accurate information based on the context. "
        )
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error. Please try again or rephrase your question. Error: {str(e)}"

# Main chat interface
with input_container:
    user_input = get_text()

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
