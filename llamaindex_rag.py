#!/usr/bin/env python3
"""
LlamaIndex RAG Plugin for Dify
Advanced RAG (Retrieval-Augmented Generation) tool for local document querying
Author: Claude AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib
import re

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Dify plugin imports
from dify_plugin import Tool, ToolProvider
from dify_plugin.entities.tool import ToolInvokeMessage, ToolParameter
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class LlamaIndexRAGProvider(ToolProvider):
    """
    LlamaIndex RAG Tool Provider for Dify
    """
    
    def _validate_credentials(self, credentials: dict) -> None:
        """
        Validate credentials for LlamaIndex RAG tool
        Since this is a local tool, just validate basic requirements
        """
        try:
            # Basic validation - check if required dependencies are available
            import llama_index
            # Tool doesn't require external credentials, so validation passes
            pass
        except ImportError as e:
            raise ToolProviderCredentialValidationError(f"LlamaIndex dependencies not available: {str(e)}")


class LlamaIndexRAGTool(Tool):
    """
    LlamaIndex RAG Tool for Dify
    
    Provides advanced RAG capabilities using LlamaIndex with local document support,
    Ollama integration, and semantic search functionality.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.indexes_cache = {}  # Cache for loaded indexes
        self.sales_keywords = [
            "upgrade", "premium", "advanced", "course", "training", 
            "lesson", "membership", "subscription", "buy", "purchase"
        ]
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_inputs(self, tool_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input parameters"""
        query = tool_parameters.get('query', '').strip()
        document_path = tool_parameters.get('document_path', '').strip()
        model_name = tool_parameters.get('model_name', 'llama3.2')
        top_k = int(tool_parameters.get('top_k', 3))
        response_mode = tool_parameters.get('response_mode', 'synthesized')
        enable_sales_detection = tool_parameters.get('enable_sales_detection', False)
        debug_mode = tool_parameters.get('debug_mode', False)
        
        # Validation
        if not query:
            raise ValueError("Query cannot be empty")
        
        if not document_path:
            raise ValueError("Document path cannot be empty")
        
        if not os.path.exists(document_path):
            raise ValueError(f"Document path does not exist: {document_path}")
        
        if top_k < 1 or top_k > 10:
            raise ValueError("top_k must be between 1 and 10")
        
        if response_mode not in ['context_only', 'synthesized']:
            raise ValueError("response_mode must be 'context_only' or 'synthesized'")
        
        return {
            'query': query,
            'document_path': document_path,
            'model_name': model_name,
            'top_k': top_k,
            'response_mode': response_mode,
            'enable_sales_detection': enable_sales_detection,
            'debug_mode': debug_mode
        }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to handle variations and synonyms"""
        # Handle common fencing term variations
        variations = {
            'sabre': 'saber',
            'saber': 'sabre',
            'épée': 'epee',
            'epee': 'épée',
            'foil': 'foil',
            'fencing': 'fencing',
            'sword': 'blade',
            'blade': 'sword',
            'mask': 'helmet',
            'helmet': 'mask',
            'jacket': 'uniform',
            'uniform': 'jacket'
        }
        
        processed_query = query.lower()
        for original, replacement in variations.items():
            if original in processed_query:
                processed_query = processed_query.replace(original, f"{original} {replacement}")
        
        return processed_query
    
    def _detect_sales_opportunities(self, query: str, context: str) -> Dict[str, Any]:
        """Detect potential sales opportunities in query and context"""
        sales_flags = []
        
        query_lower = query.lower()
        context_lower = context.lower()
        
        for keyword in self.sales_keywords:
            if keyword in query_lower:
                sales_flags.append({
                    'keyword': keyword,
                    'location': 'query',
                    'confidence': 0.8
                })
            elif keyword in context_lower:
                sales_flags.append({
                    'keyword': keyword,
                    'location': 'context',
                    'confidence': 0.6
                })
        
        # Detect specific opportunity patterns
        if any(phrase in query_lower for phrase in ['how to improve', 'get better', 'next level']):
            sales_flags.append({
                'keyword': 'improvement_opportunity',
                'location': 'query',
                'confidence': 0.9
            })
        
        return {
            'has_opportunities': len(sales_flags) > 0,
            'flags': sales_flags,
            'recommendation': 'Consider suggesting relevant courses or training programs' if sales_flags else None
        }
    
    def _setup_llama_index(self, model_name: str, debug_mode: bool) -> None:
        """Configure LlamaIndex with Ollama integration"""
        try:
            # Set up Ollama LLM
            llm = Ollama(model=model_name, base_url="http://localhost:11434")
            
            # Set up Ollama embedding
            embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",  # Use a smaller embedding model
                base_url="http://localhost:11434"
            )
            
            # Configure LlamaIndex settings
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            if debug_mode:
                self.logger.info(f"LlamaIndex configured with model: {model_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup LlamaIndex: {str(e)}")
            # Fallback to default settings
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
    
    def _get_cache_key(self, document_path: str) -> str:
        """Generate cache key for document path"""
        return hashlib.md5(document_path.encode()).hexdigest()
    
    def _load_or_create_index(self, document_path: str, debug_mode: bool) -> VectorStoreIndex:
        """Load existing index or create new one"""
        cache_key = self._get_cache_key(document_path)
        
        # Check memory cache first
        if cache_key in self.indexes_cache:
            if debug_mode:
                self.logger.info(f"Using cached index for {document_path}")
            return self.indexes_cache[cache_key]
        
        # Create storage directory
        storage_dir = Path(f"/tmp/llamaindex_storage/{cache_key}")
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try to load existing index
            if (storage_dir / "index_store.json").exists():
                if debug_mode:
                    self.logger.info(f"Loading existing index from {storage_dir}")
                
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(storage_dir)
                )
                index = VectorStoreIndex.from_documents([], storage_context=storage_context)
                
            else:
                if debug_mode:
                    self.logger.info(f"Creating new index for {document_path}")
                
                # Load documents
                documents = SimpleDirectoryReader(document_path).load_data()
                
                if not documents:
                    raise ValueError(f"No documents found in {document_path}")
                
                # Create new index
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(storage_dir)
                )
                index = VectorStoreIndex.from_documents(
                    documents, 
                    storage_context=storage_context
                )
                
                # Persist the index
                index.storage_context.persist(persist_dir=str(storage_dir))
            
            # Cache the index
            self.indexes_cache[cache_key] = index
            
            if debug_mode:
                self.logger.info(f"Index loaded/created successfully for {document_path}")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to load/create index: {str(e)}")
            raise
    
    def _query_index(self, index: VectorStoreIndex, query: str, top_k: int, 
                    response_mode: str, debug_mode: bool) -> Dict[str, Any]:
        """Query the index and return results"""
        try:
            if response_mode == 'context_only':
                # Return only retrieved context
                retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
                nodes = retriever.retrieve(QueryBundle(query))
                
                context_chunks = []
                for node in nodes:
                    context_chunks.append({
                        'content': node.node.text,
                        'score': node.score,
                        'metadata': node.node.metadata
                    })
                
                return {
                    'response': f"Found {len(context_chunks)} relevant context chunks",
                    'context_chunks': context_chunks,
                    'mode': 'context_only'
                }
            
            else:
                # Use LLM for synthesized response
                query_engine = index.as_query_engine(
                    similarity_top_k=top_k,
                    response_synthesizer=get_response_synthesizer(response_mode="compact")
                )
                
                response = query_engine.query(query)
                
                # Extract source nodes if available
                source_nodes = []
                if hasattr(response, 'source_nodes'):
                    for node in response.source_nodes:
                        source_nodes.append({
                            'content': node.node.text,
                            'score': node.score,
                            'metadata': node.node.metadata
                        })
                
                return {
                    'response': str(response),
                    'source_nodes': source_nodes,
                    'mode': 'synthesized'
                }
                
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise
    
    def _invoke(self, user_id: str, tool_parameters: Dict[str, Any]) -> Union[ToolInvokeMessage, List[ToolInvokeMessage]]:
        """
        Main invoke method for the LlamaIndex RAG tool
        
        Args:
            user_id: The user ID from Dify
            tool_parameters: Parameters from the tool form
            
        Returns:
            ToolInvokeMessage with the RAG results
        """
        try:
            # Validate inputs
            validated_params = self._validate_inputs(tool_parameters)
            
            if validated_params['debug_mode']:
                self.logger.info(f"Starting RAG query for user: {user_id}")
                self.logger.info(f"Parameters: {validated_params}")
            
            # Setup LlamaIndex
            self._setup_llama_index(
                validated_params['model_name'], 
                validated_params['debug_mode']
            )
            
            # Preprocess query
            processed_query = self._preprocess_query(validated_params['query'])
            
            # Load or create index
            index = self._load_or_create_index(
                validated_params['document_path'],
                validated_params['debug_mode']
            )
            
            # Query the index
            results = self._query_index(
                index,
                processed_query,
                validated_params['top_k'],
                validated_params['response_mode'],
                validated_params['debug_mode']
            )
            
            # Detect sales opportunities if enabled
            sales_info = None
            if validated_params['enable_sales_detection']:
                context_text = ""
                if 'context_chunks' in results:
                    context_text = " ".join([chunk['content'] for chunk in results['context_chunks']])
                elif 'source_nodes' in results:
                    context_text = " ".join([node['content'] for node in results['source_nodes']])
                
                sales_info = self._detect_sales_opportunities(
                    validated_params['query'],
                    context_text
                )
            
            # Prepare response
            response_data = {
                'query': validated_params['query'],
                'processed_query': processed_query,
                'response': results['response'],
                'mode': results['mode'],
                'model_used': validated_params['model_name'],
                'document_path': validated_params['document_path'],
                'top_k': validated_params['top_k']
            }
            
            # Add context/source information
            if 'context_chunks' in results:
                response_data['context_chunks'] = results['context_chunks']
            if 'source_nodes' in results:
                response_data['source_nodes'] = results['source_nodes']
            
            # Add sales information if available
            if sales_info:
                response_data['sales_opportunities'] = sales_info
            
            # Debug information
            if validated_params['debug_mode']:
                response_data['debug_info'] = {
                    'cache_key': self._get_cache_key(validated_params['document_path']),
                    'index_cached': self._get_cache_key(validated_params['document_path']) in self.indexes_cache,
                    'query_preprocessing': {
                        'original': validated_params['query'],
                        'processed': processed_query
                    }
                }
                self.logger.info(f"RAG query completed successfully for user: {user_id}")
            
            return ToolInvokeMessage(
                type=ToolInvokeMessage.MessageType.JSON,
                message=json.dumps(response_data, indent=2, ensure_ascii=False)
            )
            
        except Exception as e:
            error_msg = f"LlamaIndex RAG Error: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            return ToolInvokeMessage(
                type=ToolInvokeMessage.MessageType.TEXT,
                message=f"❌ {error_msg}\n\nPlease check:\n- Document path exists and contains files\n- Ollama is running on localhost:11434\n- Selected model is available in Ollama"
            )
    
    def get_runtime_parameters(self) -> List[ToolParameter]:
        """
        Get runtime parameters for the tool
        This method is called by Dify to get the tool's parameter definitions
        """
        return [
            ToolParameter(
                name="query",
                label="Query",
                type=ToolParameter.ToolParameterType.STRING,
                required=True,
                description="The question or query to search for in the documents"
            ),
            ToolParameter(
                name="document_path",
                label="Document Path",
                type=ToolParameter.ToolParameterType.STRING,
                required=True,
                description="Local folder path containing documents to search"
            ),
            ToolParameter(
                name="model_name",
                label="Ollama Model",
                type=ToolParameter.ToolParameterType.SELECT,
                required=False,
                description="Select the Ollama model for LLM responses",
                options=[
                    {"value": "llama3.2", "label": "Llama 3.2 (Balanced)"},
                    {"value": "mistral:7b", "label": "Mistral 7B (Advanced)"},
                    {"value": "phi3:mini", "label": "Phi3 Mini (Fast)"},
                    {"value": "codellama:7b", "label": "CodeLlama 7B (Code)"}
                ]
            ),
            ToolParameter(
                name="top_k",
                label="Top K Results",
                type=ToolParameter.ToolParameterType.NUMBER,
                required=False,
                description="Number of most relevant chunks to retrieve (1-10)"
            ),
            ToolParameter(
                name="response_mode",
                label="Response Mode",
                type=ToolParameter.ToolParameterType.SELECT,
                required=False,
                description="Choose between context-only or LLM-synthesized response",
                options=[
                    {"value": "context_only", "label": "Context Only"},
                    {"value": "synthesized", "label": "LLM Synthesized"}
                ]
            ),
            ToolParameter(
                name="enable_sales_detection",
                label="Enable Sales Detection",
                type=ToolParameter.ToolParameterType.BOOLEAN,
                required=False,
                description="Detect potential sales opportunities in queries"
            ),
            ToolParameter(
                name="debug_mode",
                label="Debug Mode",
                type=ToolParameter.ToolParameterType.BOOLEAN,
                required=False,
                description="Enable detailed logging for troubleshooting"
            )
        ]


# Tool class export for Dify
# The plugin system will instantiate this class as needed