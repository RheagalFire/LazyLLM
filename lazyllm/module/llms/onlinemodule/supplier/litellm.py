import os
import requests
from typing import Optional
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase


class LiteLLMChat(OnlineChatModuleBase):
    """LiteLLM AI gateway provider — route to 100+ LLM providers via a unified interface.

    Connects to a LiteLLM proxy server that exposes an OpenAI-compatible API.
    Start the proxy with: ``litellm --model <model> --port 4000``

    Model strings use the ``provider/model`` format, e.g.
    ``anthropic/claude-sonnet-4-20250514``, ``azure/gpt-4o``, ``bedrock/anthropic.claude-3-haiku``,
    ``openai/gpt-4o``, ``ollama/llama3``, etc.
    See https://docs.litellm.ai/docs/providers for the full list.
    """

    NO_PROXY = True

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        base_url = base_url or os.environ.get('LITELLM_API_BASE', 'http://localhost:4000/v1/')
        model = model or os.environ.get('LITELLM_MODEL', 'openai/gpt-4o')
        api_key = api_key or os.environ.get('LITELLM_API_KEY', '') or self._default_api_key_safe()
        super().__init__(api_key=api_key, base_url=base_url, model_name=model,
                         stream=stream, return_trace=return_trace, **kwargs)

    @classmethod
    def _default_api_key_safe(cls):
        try:
            return cls._default_api_key()
        except Exception:
            return 'sk-litellm'

    def _get_system_prompt(self):
        return 'You are a helpful assistant.'

    def _validate_api_key(self):
        try:
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class LiteLLMEmbed(LazyLLMOnlineEmbedModuleBase):
    """LiteLLM embedding provider — route to 100+ embedding providers via proxy."""

    NO_PROXY = True

    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, batch_size: int = 16, **kw):
        embed_url = embed_url or os.environ.get('LITELLM_API_BASE', 'http://localhost:4000/v1/')
        embed_model_name = embed_model_name or 'openai/text-embedding-3-small'
        api_key = api_key or os.environ.get('LITELLM_API_KEY', 'sk-litellm')
        super().__init__(embed_url, api_key, embed_model_name, batch_size=batch_size, **kw)

    def _set_embed_url(self):
        self._embed_url = urljoin(self._embed_url, 'embeddings')
