# Blog-Assistance-Using-LLM
 Blog Assistant using Azure OpenAI and HuggingFace embeddings. Ask questions and get answers grounded strictly in your uploaded documents—no outside info, just accurate, document-based responses. Easy setup and secure.

Blog Assistant is a Streamlit-based chat application that lets you ask questions about your own documents and get AI-powered answers strictly based on their content. It uses Azure OpenAI for language generation and HuggingFace embeddings for semantic search, ensuring responses are always grounded in your uploaded files.

Features

Conversational Q&A: Chat with an AI assistant that answers using only your document knowledge base.

Document-grounded responses: No hallucinations or outside information—answers are always based on your files.

Easy setup: Configure credentials and knowledge base paths in YAML.

Robust error handling: Clear feedback for missing configuration or files.

Installation

Clone this repository.

Install dependencies:

bash
pip install -r requirements.txt
Add your credentials to config/credentials.yaml.

Place your documents in the knowledge base directory (default: kb/).

Run the app:

bash
AI bot.py
Usage
Open the app in your browser.

Ask questions about your documents in the chat interface.

The assistant will answer only within the scope of your uploaded documents.

Configuration
Update config/credentials.yaml with your Azure OpenAI and embedding model details. You can also change the knowledge base path as needed.


License
MIT License


Acknowledgments
Built with Streamlit, llama_index, Azure OpenAI, and HuggingFace.

