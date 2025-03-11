# TherapyBot 🤖💭

A mental health-focused chatbot powered by advanced language models and RAG (Retrieval-Augmented Generation) technology. This application provides therapeutic conversations while leveraging evidence-based mental health resources.

## Features

- 🧠 Conversational AI with mental health expertise
- 📚 RAG-based responses using verified mental health documentation
- 💬 Interactive chat interface built with Streamlit
- 🔄 Conversation memory for contextual responses
- 📑 Document processing with PDF support

## Technical Architecture

- **Frontend**: Streamlit-based interactive web interface
- **Language Model**: Integration with HuggingFace models
- **RAG System**: Custom implementation using:
  - Document processing (PyMuPDF)
  - Vector storage (Chroma)
  - Embedding generation (HuggingFace)
  - Memory management (LangChain)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Meggison/therapybot.git
cd therapybot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501` by default.

## Project Structure

- `streamlit_app.py`: Main application file with UI and chat logic
- `rag/`: RAG implementation components
  - `generator.py`: LLM loading and response generation
  - `retriever.py`: Document processing and retrieval logic
- `model/`: Model-related configurations
- `preprocessing/`: Text preprocessing utilities
- `prompts/`: System and user prompt templates

## Dependencies

Key dependencies include:
- Streamlit and extensions for UI
- LangChain for RAG implementation
- HuggingFace libraries for embeddings and models
- ChromaDB for vector storage
- PyMuPDF for PDF processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
