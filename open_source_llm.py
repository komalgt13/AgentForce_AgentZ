"""
Gemini LLM Integration for Agentic Persona Generator
Uses Google's Gemini 2.5-flash model via LangChain
"""

import os
import logging
from typing import Optional, Dict, Any, List

# LangChain imports for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class LangChainCompatibleGeminiWrapper(BaseLanguageModel):
    """
    LangChain-compatible wrapper around DirectGeminiWrapper
    This allows us to use the working DirectGeminiWrapper with LangChain agents
    """
    
    def __init__(self, direct_wrapper):
        super().__init__()
        self.direct_wrapper = direct_wrapper
        
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, 
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        """Generate responses for given prompts"""
        generations = []
        for prompt in prompts:
            try:
                response = self.direct_wrapper.invoke(prompt)
                generation = Generation(text=response.content)
                generations.append([generation])
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                generation = Generation(text=f"Error: {str(e)}")
                generations.append([generation])
        
        return LLMResult(generations=generations)
    
    def _llm_type(self) -> str:
        return "gemini_direct_wrapper"
    
    def invoke(self, prompt, **kwargs):
        """Direct invocation method for compatibility"""
        return self.direct_wrapper.invoke(prompt)

class DirectGeminiWrapper:
    """
    Simple Direct Gemini API wrapper for basic LLM operations
    Used as fallback when LangChain integration has issues
    """
    
    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        import google.generativeai as genai
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        genai.configure(api_key=api_key)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
    
    def invoke(self, prompt):
        """Invoke the model with a prompt"""
        try:
            if isinstance(prompt, list):
                # Handle list of messages (LangChain format)
                content = prompt[0].content if hasattr(prompt[0], 'content') else str(prompt[0])
            else:
                content = prompt if isinstance(prompt, str) else str(prompt)
            
            response = self.model.generate_content(content)
            
            # Create a response object that mimics LangChain
            class Response:
                def __init__(self, content):
                    self.content = content
                    
                def __str__(self):
                    return self.content
            
            return Response(response.text)
            
        except Exception as e:
            logger.error(f"Direct Gemini API call failed: {str(e)}")
            raise
    
    def __call__(self, prompt):
        """Support old-style calling"""
        return self.invoke(prompt)

class GeminiLLMManager:
    """
    Gemini LLM Manager for Google Generative AI
    Handles Gemini 2.5-flash model integration
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini LLM Manager"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Set the API key in environment
        os.environ["GOOGLE_API_KEY"] = self.api_key
        
        self.available_models = {
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-2.5-flash": "gemini-1.5-flash"  # Map to available model
        }
        
        logger.info(f"Initialized Gemini LLM Manager with API key")
    
    def get_llm(self, 
                model_name: str = "gemini-1.5-flash",
                temperature: float = 0.3,
                max_tokens: int = 2048,
                use_langchain: bool = True,
                **kwargs):
        """
        Get Gemini LLM instance
        
        Args:
            model_name: Model name (default: gemini-1.5-flash)
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            use_langchain: Whether to use LangChain integration (default: True)
            **kwargs: Additional model parameters
        
        Returns:
            ChatGoogleGenerativeAI instance or DirectGeminiWrapper
        """
        try:
            # Use stable models that are available
            stable_models = {
                "gemini-2.5-flash": "gemini-1.5-flash",  # Map to available model
                "gemini-1.5-pro": "gemini-1.5-pro",
                "gemini-1.5-flash": "gemini-1.5-flash"
            }
            
            # Map model name if needed
            full_model_name = stable_models.get(model_name, "gemini-1.5-flash")
            
            if use_langchain:
                try:
                    # Try LangChain ChatGoogleGenerativeAI first
                    llm = ChatGoogleGenerativeAI(
                        model=full_model_name,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        google_api_key=self.api_key
                    )
                    logger.info(f"Created LangChain Gemini LLM: {full_model_name}")
                    return llm
                except Exception as langchain_error:
                    logger.warning(f"LangChain integration failed: {str(langchain_error)}")
                    logger.info("Creating LangChain-compatible wrapper around DirectGeminiWrapper")
                    
                    # Create DirectGeminiWrapper and wrap it for LangChain compatibility
                    direct_wrapper = DirectGeminiWrapper(self.api_key, full_model_name, temperature, max_tokens)
                    return LangChainCompatibleGeminiWrapper(direct_wrapper)
            
            # Use direct API wrapper for non-LangChain usage
            logger.info(f"Using Direct Gemini API for model: {full_model_name}")
            return DirectGeminiWrapper(self.api_key, full_model_name, temperature, max_tokens)
            
        except Exception as e:
            logger.error(f"Failed to create Gemini LLM: {str(e)}")
            raise
    
    def get_embeddings(self, 
                      model_name: str = "models/embedding-001",
                      **kwargs) -> GoogleGenerativeAIEmbeddings:
        """
        Get Gemini embeddings instance
        
        Args:
            model_name: Embedding model name
            **kwargs: Additional parameters
        
        Returns:
            GoogleGenerativeAIEmbeddings instance
        """
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key,
                **kwargs
            )
            
            logger.info(f"Created Gemini embeddings: {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create Gemini embeddings: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        try:
            # Test with a simple request
            llm = self.get_llm()
            test_response = llm.invoke("Hello")
            return True
        except Exception as e:
            logger.warning(f"Gemini API not available: {str(e)}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List available Gemini models"""
        return list(self.available_models.keys())
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Gemini API using direct API"""
        try:
            import google.generativeai as genai
            
            # Configure direct API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content('Hello')
            
            return {
                "status": "success",
                "model": "gemini-1.5-flash",
                "response_length": len(response.text),
                "response": response.text[:50] + "..." if len(response.text) > 50 else response.text,
                "test_successful": True
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "test_successful": False
            }

# Backward compatibility functions
def get_gemini_llm(model_name: str = "gemini-1.5-flash", **kwargs):
    """Get Gemini LLM instance - backward compatibility function"""
    manager = GeminiLLMManager()
    return manager.get_llm(model_name, **kwargs)

def get_gemini_embeddings(model_name: str = "models/embedding-001", **kwargs):
    """Get Gemini embeddings - backward compatibility function"""
    manager = GeminiLLMManager()
    return manager.get_embeddings(model_name, **kwargs)

def list_available_models() -> List[str]:
    """List available models"""
    try:
        manager = GeminiLLMManager()
        return manager.list_available_models()
    except Exception:
        return []

# Legacy compatibility
OpenSourceLLMManager = GeminiLLMManager
get_open_source_llm = get_gemini_llm  
get_open_source_embeddings = get_gemini_embeddings
