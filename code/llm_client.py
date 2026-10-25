# -*- coding: utf-8 -*-
"""LLM 客户端门面：按 config 选择后端并缓存实例。"""
from typing import List, Optional

from llm_backends import BaseBackend


def _build_backend(model_cfg: dict) -> BaseBackend:
    backend = model_cfg.get("backend", "transformers")
    if backend == "mock":
        from llm_backends import MockBackend
        return MockBackend(model_cfg)
    if backend == "transformers":
        from llm_backends import TransformersBackend
        return TransformersBackend(model_cfg)
    if backend == "vllm":
        from llm_backends import VllmBackend
        return VllmBackend(model_cfg)
    if backend == "openai":
        from llm_backends import OpenAIBackend
        return OpenAIBackend(model_cfg)
    raise ValueError(f"未知后端: {backend}")


class LLMClient:
    """统一模型客户端。"""

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self._backend = None

    @property
    def backend(self) -> BaseBackend:
        if self._backend is None:
            self._backend = _build_backend(self.model_cfg)
        return self._backend

    @property
    def backend_type(self) -> str:
        return self.backend.backend_type

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        mock_target: Optional[str] = None,
        mock_round: int = 0,
        mock_gold: Optional[str] = None,
    ) -> List[str]:
        """统一生成入口。mock 相关参数仅 mock 后端使用, 其余后端自动忽略。"""
        s = self.model_cfg.get("_sampling", {})
        temperature = s.get("temperature", 0.7) if temperature is None else temperature
        top_p = s.get("top_p", 0.95) if top_p is None else top_p
        max_tokens = s.get("max_new_tokens", 2048) if max_tokens is None else max_tokens

        if self.backend.supports("mock_hooks"):
            return self.backend.generate(
                prompt, n=n, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens,
                mock_target=mock_target, mock_round=mock_round, mock_gold=mock_gold,
            )
        return self.backend.generate(
            prompt, n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
