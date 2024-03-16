import os
from langchain_community.llms import HuggingFaceEndpoint


# Set your HUGGINGFACEHUB_API_TOKEN in your environment variables for security
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


def load_llm(repo_id="mistralai/Mistral-7B-Instruct-v0.2", token='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'):
    '''
    Load the LLM from the HuggingFace model hub

    Args:
        repo_id (str): The HuggingFace model ID

    Returns:
        llm (HuggingFaceEndpoint): The LLM model
    '''

    repo_id = repo_id

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.2, token=token)

    return llm

def guardrails():
    return None
