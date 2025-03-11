import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

def load_llm(repo_id="mistralai/Mistral-7B-Instruct-v0.2"):
    '''
    Load the LLM from the HuggingFace model hub

    Args:
        repo_id (str): The HuggingFace model ID

    Returns:
        llm (HuggingFaceEndpoint): The LLM model
    '''
    token = os.getenv('HUGGINGFACE_API_TOKEN')
    if not token:
        raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables. Please set it in your .env file.")

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.2,
        token=token
    )

    return llm

def guardrails(input_text):
    """
    Apply safety guardrails to the input text
    
    Args:
        input_text (str): The input text to check
        
    Returns:
        bool: True if text passes safety checks, False otherwise
    """
    # Add basic content filtering
    unsafe_keywords = ['suicide', 'self-harm', 'harm', 'kill']
    
    # Convert to lowercase for case-insensitive matching
    text_lower = input_text.lower()
    
    # Check for unsafe content
    for keyword in unsafe_keywords:
        if keyword in text_lower:
            return False
            
    return True
