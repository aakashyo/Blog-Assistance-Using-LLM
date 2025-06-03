import os
import sys
import yaml
import logging
import streamlit as st
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
from llama_index.llms import AzureOpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import (
    ServiceContext, 
    StorageContext, 
    load_index_from_storage,
    set_global_service_context
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatConfig:
    """Configuration class for chat application settings."""
    credentials_path: str = "config/credentials.yaml"
    knowledge_base_path: str = "knowledge_base"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    app_title: str = "AI Knowledge Assistant"
    app_description: str = "Interactive Q&A system powered by document knowledge base"

class CredentialManager:
    """Handles loading and managing API credentials."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._credentials = None
    
    def load_credentials(self) -> dict:
        """Load credentials from YAML configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                self._credentials = yaml.safe_load(file)
            logger.info("Credentials loaded successfully")
            return self._credentials
        except FileNotFoundError:
            logger.error(f"Credentials file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing credentials file: {e}")
            raise
    
    @property
    def credentials(self) -> dict:
        """Get loaded credentials."""
        if self._credentials is None:
            self.load_credentials()
        return self._credentials

class IndexManager:
    """Manages vector index operations and query engine."""
    
    def __init__(self, storage_path: str, service_context: ServiceContext):
        self.storage_path = Path(storage_path)
        self.service_context = service_context
        self._query_engine = None
    
    def initialize_query_engine(self) -> Any:
        """Initialize and return the query engine from stored index."""
        if not self.storage_path.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.storage_path}")
        
        logger.info("Initializing query engine from stored index...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_path))
            index = load_index_from_storage(storage_context=storage_context)
            self._query_engine = index.as_query_engine()
            logger.info("Query engine initialized successfully")
            return self._query_engine
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            raise
    
    @property
    def query_engine(self) -> Any:
        """Get the query engine, initializing if necessary."""
        if self._query_engine is None:
            self.initialize_query_engine()
        return self._query_engine

class ChatApplication:
    """Main chat application class."""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.credential_manager = CredentialManager(config.credentials_path)
        self.service_context = self._setup_service_context()
        self.index_manager = IndexManager(config.knowledge_base_path, self.service_context)
    
    def _setup_service_context(self) -> ServiceContext:
        """Configure and return the service context with LLM and embeddings."""
        credentials = self.credential_manager.credentials
        
        # Initialize Azure OpenAI LLM
        azure_llm = AzureOpenAI(
            deployment_name=credentials.get('AZURE_DEPLOYMENT_NAME'),
            model=credentials.get('AZURE_ENGINE'),
            api_key=credentials.get('AZURE_OPENAI_KEY'),
            api_version=credentials.get('AZURE_OPENAI_VERSION'),
            azure_endpoint=credentials.get('AZURE_OPENAI_BASE')
        )
        
        llm_predictor = LLMPredictor(llm=azure_llm)
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbedding(model_name=self.config.embedding_model)
        
        # Create service context
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=embedding_model
        )
        
        set_global_service_context(service_context)
        logger.info("Service context configured successfully")
        return service_context
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "query_engine" not in st.session_state:
            st.session_state.query_engine = self.index_manager.query_engine
        
        if "chat_history" not in st.session_state:
            system_message = {
                "role": "system",
                "content": (
                    "You are a knowledgeable AI assistant designed to answer questions "
                    "based solely on the provided document knowledge base. "
                    "Provide accurate, helpful responses within the scope of available information. "
                    "If a question falls outside your knowledge scope, politely redirect the user. "
                    "When uncertain, respond with 'I don't have enough information to answer that question.'"
                )
            }
            st.session_state.chat_history = [system_message]
    
    def _render_chat_interface(self) -> None:
        """Render the main chat interface."""
        st.title(self.config.app_title)
        st.markdown(f"*{self.config.app_description}*")
        st.divider()
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] in ["user", "assistant"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    def _process_user_input(self, user_input: str) -> None:
        """Process user input and generate response."""
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.query_engine.query(user_input)
                    response_text = str(response)
                    
                    st.markdown(response_text)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_text
                    })
                    
                except Exception as e:
                    error_message = "I encountered an error while processing your question. Please try again."
                    st.error(error_message)
                    logger.error(f"Query processing error: {e}")
    
    def run(self) -> None:
        """Run the main application."""
        try:
            # Configure Streamlit page
            st.set_page_config(
                page_title=self.config.app_title,
                page_icon="🤖",
                layout="centered"
            )
            
            self._initialize_session_state()
            self._render_chat_interface()
            
            # Handle user input
            if user_message := st.chat_input("Ask me anything about the documents..."):
                self._process_user_input(user_message)
                
        except Exception as e:
            st.error("Failed to initialize the application. Please check your configuration.")
            logger.error(f"Application startup error: {e}")

def create_app() -> ChatApplication:
    """Factory function to create and configure the chat application."""
    config = ChatConfig(
        credentials_path=os.getenv('CREDENTIALS_PATH', 'config/credentials.yaml'),
        knowledge_base_path=os.getenv('KB_PATH', 'kb'),
        app_title="Document Q&A Assistant",
        app_description="Ask questions about your document collection and get AI-powered answers"
    )
    return ChatApplication(config)

def main() -> None:
    """Application entry point."""
    try:
        app = create_app()
        app.run()
    except Exception as e:
        st.error("Application failed to start. Please check your setup and try again.")
        logger.critical(f"Critical application error: {e}")

if __name__ == "__main__":
    main()