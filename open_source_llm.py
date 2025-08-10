"""
Open Source LLM Integration for Agentic Persona Generator
Supports multiple open source LLM providers: Ollama, HuggingFace, LocalAI
"""

import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# LangChain imports for open source models
from langchain.llms.base import LLM
from langchain_community.llms import Ollama, HuggingFacePipeline
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage
from langchain.schema.embeddings import Embeddings

# HuggingFace and Transformers
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        pipeline, BitsAndBytesConfig
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class OpenSourceLLMProvider(ABC):
    """Abstract base class for open source LLM providers"""
    
    @abstractmethod
    def get_llm(self, model_name: str, **kwargs) -> LLM:
        """Get LLM instance"""
        pass
    
    @abstractmethod
    def get_embeddings(self, model_name: str, **kwargs) -> Embeddings:
        """Get embeddings instance"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

class OllamaProvider(OpenSourceLLMProvider):
    """Ollama LLM Provider - runs models locally"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def get_llm(self, model_name: str, **kwargs) -> LLM:
        """Get Ollama LLM instance"""
        return Ollama(
            base_url=self.base_url,
            model=model_name,
            temperature=kwargs.get('temperature', 0.3),
            top_p=kwargs.get('top_p', 0.9),
            num_predict=kwargs.get('max_tokens', 2048),
            verbose=kwargs.get('verbose', False)
        )
    
    def get_embeddings(self, model_name: str = "nomic-embed-text", **kwargs) -> Embeddings:
        """Get Ollama embeddings instance"""
        return OllamaEmbeddings(
            base_url=self.base_url,
            model=model_name
        )
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

class HuggingFaceProvider(OpenSourceLLMProvider):
    """HuggingFace LLM Provider - supports both local and API models"""
    
    def __init__(self, use_api: bool = False, api_token: Optional[str] = None):
        self.use_api = use_api
        self.api_token = api_token or os.getenv('HUGGINGFACE_API_TOKEN')
        self.device = "cuda" if torch.cuda.is_available() and os.getenv('USE_GPU', 'true').lower() == 'true' else "cpu"
    
    def get_llm(self, model_name: str, **kwargs) -> LLM:
        """Get HuggingFace LLM instance optimized for Mistral 7B"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers torch")
        
        # Special handling for Mistral models
        if "mistral" in model_name.lower():
            logger.info(f"Optimizing for Mistral model: {model_name}")
            
        # Quantization config for efficient memory usage
        quantization_config = None
        if self.device == "cuda" and os.getenv('MODEL_QUANTIZATION', '4bit') == '4bit':
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,  # Better for Mistral
                )
                logger.info("Using 4-bit quantization for efficient memory usage")
            except Exception as e:
                logger.warning(f"4-bit quantization not available: {e}")
        
        # Load model and tokenizer
        try:
            logger.info(f"Loading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Ensure proper padding token (important for Mistral)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            logger.info(f"Loading model {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,  # Important for large models like Mistral 7B
                attn_implementation="flash_attention_2" if self.device == "cuda" else None,  # Optimize attention
            )
            
            # Pipeline parameters optimized for Mistral
            pipeline_kwargs = {
                'max_new_tokens': kwargs.get('max_tokens', 2048),
                'temperature': kwargs.get('temperature', 0.3),
                'top_p': kwargs.get('top_p', 0.9),
                'do_sample': kwargs.get('do_sample', True),
                'repetition_penalty': 1.1,  # Reduce repetitions
                'return_full_text': False,  # Only return generated text
                'pad_token_id': tokenizer.eos_token_id,
            }
            
            # Create pipeline
            logger.info("Creating text generation pipeline...")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **pipeline_kwargs
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
            logger.info("Attempting fallback model...")
            # Fallback to a smaller model
            return self._get_fallback_model(**kwargs)
    
    def _get_fallback_model(self, **kwargs) -> LLM:
        """Get a smaller fallback model"""
        try:
            fallback_model = "microsoft/DialoGPT-small"
            logger.info(f"Loading fallback model: {fallback_model}")
            
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=kwargs.get('max_tokens', 512),  # Smaller for fallback
                temperature=kwargs.get('temperature', 0.7),
                do_sample=True
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            raise
    
    def get_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs) -> Embeddings:
        """Get HuggingFace embeddings instance"""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def is_available(self) -> bool:
        """Check if HuggingFace is available"""
        return TRANSFORMERS_AVAILABLE

class OpenSourceLLMManager:
    """Manager class for open source LLM providers"""
    
    def __init__(self):
        self.providers = {
            'ollama': OllamaProvider(base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')),
            'huggingface': HuggingFaceProvider(),
            'hf': HuggingFaceProvider()  # Alias
        }
        
        # Default configurations for different models
        self.model_configs = {
            # Ollama models
            'llama3.2:3b': {'temperature': 0.3, 'max_tokens': 2048, 'provider': 'ollama'},
            'llama3.2:7b': {'temperature': 0.3, 'max_tokens': 2048, 'provider': 'ollama'},
            'mistral:7b': {'temperature': 0.3, 'max_tokens': 2048, 'provider': 'ollama'},
            'codellama:7b': {'temperature': 0.2, 'max_tokens': 2048, 'provider': 'ollama'},
            
            # HuggingFace models
            'microsoft/DialoGPT-medium': {'temperature': 0.7, 'max_tokens': 1024, 'provider': 'huggingface'},
            'microsoft/DialoGPT-small': {'temperature': 0.7, 'max_tokens': 512, 'provider': 'huggingface'},
            'distilgpt2': {'temperature': 0.5, 'max_tokens': 512, 'provider': 'huggingface'},
            'gpt2': {'temperature': 0.5, 'max_tokens': 1024, 'provider': 'huggingface'},
        }
    
    def get_llm(self, model_name: str = None, provider: str = None, **kwargs) -> LLM:
        """Get LLM instance with automatic provider detection"""
        
        # Use environment defaults if not specified
        if not model_name:
            model_name = os.getenv('LLM_MODEL', 'llama3.2:3b')
        
        if not provider:
            provider = os.getenv('LLM_PROVIDER', 'ollama')
        
        # Get model config if available
        config = self.model_configs.get(model_name, {})
        if 'provider' in config and not provider:
            provider = config['provider']
        
        # Merge config with kwargs
        final_kwargs = {**config, **kwargs}
        final_kwargs.pop('provider', None)  # Remove provider from kwargs
        
        # Get provider instance
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(self.providers.keys())}")
        
        provider_instance = self.providers[provider]
        
        # Check availability
        if not provider_instance.is_available():
            logger.warning(f"Provider {provider} not available, trying fallbacks...")
            return self._get_fallback_llm(model_name, **final_kwargs)
        
        try:
            return provider_instance.get_llm(model_name, **final_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize {provider} with model {model_name}: {e}")
            return self._get_fallback_llm(model_name, **final_kwargs)
    
    def get_embeddings(self, model_name: str = None, provider: str = None, **kwargs) -> Embeddings:
        """Get embeddings instance"""
        
        if not model_name:
            model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        if not provider:
            provider = os.getenv('EMBEDDING_PROVIDER', 'huggingface')
        
        provider_instance = self.providers.get(provider)
        if not provider_instance:
            # Fallback to HuggingFace for embeddings
            provider_instance = self.providers['huggingface']
        
        return provider_instance.get_embeddings(model_name, **kwargs)
    
    def _get_fallback_llm(self, original_model: str, **kwargs) -> LLM:
        """Get fallback LLM when primary fails"""
        fallback_options = [
            ('ollama', 'llama3.2:3b'),
            ('huggingface', 'microsoft/DialoGPT-small'),
            ('huggingface', 'distilgpt2')
        ]
        
        for provider, model in fallback_options:
            try:
                provider_instance = self.providers[provider]
                if provider_instance.is_available():
                    logger.info(f"Using fallback: {provider} with {model}")
                    return provider_instance.get_llm(model, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback {provider}/{model} failed: {e}")
                continue
        
        raise RuntimeError("No working LLM provider found")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List available models for each provider"""
        available = {}
        
        for provider_name, provider in self.providers.items():
            if provider.is_available():
                if provider_name == 'ollama':
                    # Get Ollama models
                    try:
                        import requests
                        response = requests.get(f"{provider.base_url}/api/tags")
                        if response.status_code == 200:
                            models = [model['name'] for model in response.json().get('models', [])]
                            available[provider_name] = models
                    except:
                        available[provider_name] = ['llama3.2:3b', 'mistral:7b']  # Common defaults
                
                elif provider_name == 'huggingface':
                    # Common HF models
                    available[provider_name] = [
                        'microsoft/DialoGPT-medium',
                        'microsoft/DialoGPT-small', 
                        'distilgpt2',
                        'gpt2'
                    ]
        
        return available
    
    def test_model(self, model_name: str, provider: str = None) -> bool:
        """Test if a model works"""
        try:
            llm = self.get_llm(model_name, provider)
            # Test with a simple prompt
            response = llm("Hello, world!")
            return len(response.strip()) > 0
        except Exception as e:
            logger.error(f"Model test failed for {model_name}: {e}")
            return False

# Convenience functions
def get_open_source_llm(model_name: str = None, provider: str = None, **kwargs) -> LLM:
    """Get an open source LLM instance"""
    manager = OpenSourceLLMManager()
    return manager.get_llm(model_name, provider, **kwargs)

def get_open_source_embeddings(model_name: str = None, provider: str = None, **kwargs) -> Embeddings:
    """Get an open source embeddings instance"""
    manager = OpenSourceLLMManager()
    return manager.get_embeddings(model_name, provider, **kwargs)

def list_available_models() -> Dict[str, List[str]]:
    """List all available open source models"""
    manager = OpenSourceLLMManager()
    return manager.list_available_models()

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Open Source LLM Integration Test")
    print("=" * 50)
    
    manager = OpenSourceLLMManager()
    
    # List available models
    print("\n📋 Available Models:")
    available = manager.list_available_models()
    for provider, models in available.items():
        print(f"  {provider}: {models}")
    
    # Test model loading
    print(f"\n🔧 Testing Model Loading...")
    
    test_models = [
        ('ollama', 'llama3.2:3b'),
        ('huggingface', 'microsoft/DialoGPT-small')
    ]
    
    for provider, model in test_models:
        print(f"\n  Testing {provider}/{model}...")
        try:
            llm = manager.get_llm(model, provider, temperature=0.5, max_tokens=100)
            print(f"  ✅ {provider}/{model} loaded successfully")
            
            # Test inference
            response = llm("What is artificial intelligence?")
            print(f"  📝 Response preview: {response[:100]}...")
            
        except Exception as e:
            print(f"  ❌ {provider}/{model} failed: {str(e)}")
    
    print(f"\n✨ Open Source LLM integration ready!")
