import json
import os
from typing import Any, Dict, List, Optional, Union
from ..base import OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase
from ..base.utils import resolve_online_params

import lazyllm
from lazyllm import globals


class LiteLLMChat(OnlineChatModuleBase):
    """LiteLLM AI gateway provider — route to 100+ LLM providers via a unified interface.

    Model strings use the ``provider/model`` format, e.g.
    ``anthropic/claude-sonnet-4-20250514``, ``azure/gpt-4o``, ``bedrock/anthropic.claude-3-haiku``,
    ``openai/gpt-4o``, ``ollama/llama3``, etc.
    See https://docs.litellm.ai/docs/providers for the full list.
    """

    NO_PROXY = True

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 skip_auth: bool = True, **kwargs):
        base_url = base_url or 'https://api.openai.com/v1/'
        model = model or 'openai/gpt-4o'
        super().__init__(api_key=api_key or os.environ.get('LITELLM_API_KEY', '') or 'unused',
                         base_url=base_url, model_name=model, stream=stream,
                         return_trace=return_trace, skip_auth=skip_auth, **kwargs)

    def _get_system_prompt(self):
        return 'You are a helpful assistant.'

    def _validate_api_key(self):
        return True

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = None, stream: bool = None,
                lazyllm_files=None, url: str = None, model: str = None, **kw):
        try:
            import litellm
        except ImportError:
            raise ImportError(
                'litellm is required for LiteLLMChat. Install it with: pip install litellm'
            )

        stream_output = stream_output if stream_output is not None else stream
        stream_output = stream_output if stream_output is not None else self._stream
        __input, files = self._get_files(__input, lazyllm_files)
        model, _, url, kw = resolve_online_params(model, None, url, kw,
                                                  model_aliases='model_name', url_aliases='base_url')
        runtime_model = model or self._model_name

        params = {'input': __input, 'history': llm_chat_history, 'format': self._message_format}
        if tools:
            params['tools'] = tools
        data = self._prompt.generate_prompt(**params)
        data.update(self._static_params, **dict(model=runtime_model, stream=bool(stream_output)))

        if len(kw) > 0:
            data.update(kw)
        if len(self._model_optional_params) > 0:
            data.update(self._model_optional_params)

        if self.type == 'VLM' and (files or self._vlm_force_format_input_with_files):
            data['messages'][-1]['content'] = self._format_input_with_files(
                data['messages'][-1]['content'], files)
            if llm_chat_history and len(data['messages']) > 1:
                for msg in data['messages'][:-1]:
                    if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                        msg['content'] = self._format_vl_chat_query(msg['content'])

        messages = data.pop('messages', [])
        model_name = data.pop('model', runtime_model)
        do_stream = data.pop('stream', bool(stream_output))

        response = litellm.completion(
            model=model_name,
            messages=messages,
            stream=do_stream,
            drop_params=True,
            **data,
        )

        if do_stream:
            return self._handle_stream_response(response, stream_output)
        else:
            return self._handle_response(response)

    def _handle_response(self, response) -> str:
        resp_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        usage = {'prompt_tokens': -1, 'completion_tokens': -1}
        if 'usage' in resp_dict and isinstance(resp_dict['usage'], dict):
            usage['prompt_tokens'] = resp_dict['usage'].get('prompt_tokens', -1)
            usage['completion_tokens'] = resp_dict['usage'].get('completion_tokens', -1)
        self._record_usage(usage)
        extractor = self._extract_specified_key_fields(resp_dict)
        return self._formatter(extractor) if extractor else ''

    def _handle_stream_response(self, response, stream_output) -> str:
        msg_json_list = []
        for chunk in response:
            chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else dict(chunk)
            if not chunk_dict.get('choices'):
                continue
            color = stream_output.get('color') if isinstance(stream_output, dict) else None
            with self.stream_output(stream_output):
                for item in chunk_dict.get('choices', []):
                    delta = item.get('delta', {})
                    if (reasoning := delta.get('reasoning_content', '')):
                        self._stream_output(reasoning, color, cls='think')
                    elif (content := delta.get('content', '')) and not delta.get('tool_calls'):
                        self._stream_output(content, color)
            msg_json_list.append(chunk_dict)

        usage = {'prompt_tokens': -1, 'completion_tokens': -1}
        if msg_json_list and 'usage' in msg_json_list[-1] and isinstance(msg_json_list[-1]['usage'], dict):
            for k in usage:
                usage[k] = msg_json_list[-1]['usage'].get(k, usage[k])
        self._record_usage(usage)
        merged = self._merge_stream_result(msg_json_list)
        extractor = self._extract_specified_key_fields(merged)
        return self._formatter(extractor) if extractor else ''


class LiteLLMEmbed(LazyLLMOnlineEmbedModuleBase):
    """LiteLLM embedding provider — route to 100+ embedding providers."""

    NO_PROXY = True

    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, batch_size: int = 16, **kw):
        embed_url = embed_url or 'https://api.openai.com/v1/'
        embed_model_name = embed_model_name or 'openai/text-embedding-3-small'
        super().__init__(embed_url, api_key or os.environ.get('LITELLM_API_KEY', '') or 'unused',
                         embed_model_name, batch_size=batch_size, **kw)

    def _set_embed_url(self):
        pass

    def _encapsulated_data(self, text: Union[str, List[str]], **kwargs) -> Dict[str, str]:
        return {'model': self._embed_model_name, 'input': text}

    def _call_embedding(self, text: Union[str, List[str]], **kwargs) -> List[List[float]]:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                'litellm is required for LiteLLMEmbed. Install it with: pip install litellm'
            )
        response = litellm.embedding(
            model=self._embed_model_name,
            input=text if isinstance(text, list) else [text],
            drop_params=True,
        )
        data = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        return [item['embedding'] for item in data['data']]
