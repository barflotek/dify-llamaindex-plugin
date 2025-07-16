from dify_plugin.interfaces.tool import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

class LlamaIndexRAGProvider(ToolProvider):
    """
    Provider for LlamaIndex RAG Tool
    """
    
    def _validate_credentials(self, credentials: dict) -> None:
        """
        Validate credentials for the provider
        """
        # No special credentials needed for this tool
        pass