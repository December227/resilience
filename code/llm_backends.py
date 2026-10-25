# -*- coding: utf-8 -*-
"""推理后端的统一封装。

四个后端共用 generate(prompt, n, temperature, top_p, max_tokens) 这一个入口，
用 config.model.backend 切换：

  transformers  本地权重加载，机制实验需要直接访问模型与分词器，只能用这个
  vllm          连本地 vLLM 的 OpenAI 兼容服务，批量采样吞吐更高
  openai        连远程 API（DeepSeek、DashScope、SiliconFlow 等），无需显卡
  mock          不加载模型，按规则造假数据，用于跑通全流程和验证判定逻辑

放在一个文件里而不是做成包，是为了让仓库保持扁平，clone 下来不用管包结构。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


# ---------------------------------------------------------------- 抽象基类

class BaseBackend(ABC):
    """所有后端的公共接口。子类只需实现 generate，supports 用来声明可选特性。"""

    backend_type = "base"

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg

    @abstractmethod
    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> List[str]:
        raise NotImplementedError

    def supports(self, feature: str) -> bool:
        """是否支持某个可选特性，目前只有 mock 后端对 mock_hooks 返回 True。"""
        return False


# ---------------------------------------------------------------- 假数据后端，不加载模型

import hashlib
from typing import List, Optional


class MockBackend(BaseBackend):
    """假数据后端，用来在不加载模型的情况下跑通整条流程。

    生成 base CoT 时套固定模板，答案取 mock_gold，保证判定为答对。
    做重采样续写时按目标句的 hash 决定它能撑几轮，前几轮重复原文模拟重新表述，
    之后换成无关句子模拟内容被清除，这样韧性分数才会散开而不是全一样。
    """
    backend_type = "mock"

    BASE_TEMPLATE = (
        "<think>\n"
        "Let me parse this problem carefully.\n"
        "The key step is to set up the equation from the given numbers.\n"
        "I will compute step by step: add then subtract.\n"
        "After simplifying, the result is {gold}.\n"
        "</think>\n"
        "#### {gold}"
    )

    OTHER_SENTENCES = [
        "Let me double check by re-reading the problem.",
        "This step looks straightforward, no need to overthink.",
        "I will verify the arithmetic one more time.",
        "The units here are consistent, so the answer follows directly.",
    ]

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)

    def supports(self, feature: str) -> bool:
        return feature == "mock_hooks"

    @staticmethod
    def _cap_for(target: str, max_k: int) -> int:
        """由目标句文本的 hash 决定其韧性上限（1..max_k），以产生分布差异。"""
        h = int(hashlib.md5(target.encode("utf-8")).hexdigest(), 16)
        return 1 + h % max_k

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> List[str]:
        mock_target = kwargs.get("mock_target")
        mock_round = kwargs.get("mock_round", 0)
        mock_gold = kwargs.get("mock_gold")
        max_k = kwargs.get("mock_max_k", 5)

        if mock_target is not None:
            # 重采样续写：模拟"重新表述/错误纠正"
            cap = self._cap_for(mock_target, max_k)
            if mock_round < cap:
                # 内容在下游重新出现
                text = mock_target + "\nThis confirms my earlier reasoning."
            else:
                # 内容被清除，改为无关表述
                text = self.OTHER_SENTENCES[mock_round % len(self.OTHER_SENTENCES)]
            return [text] * n

        # 生成 base CoT
        gold = mock_gold if mock_gold is not None else "42"
        template = self.BASE_TEMPLATE.format(gold=gold)
        return [template] * n


# ---------------------------------------------------------------- 本地权重推理，机制实验用这个

import os
from typing import List, Optional


class TransformersBackend(BaseBackend):
    """本地权重推理，机制实验只能用它，因为要直接拿到模型和分词器做逐层干预。

    权重按 float16 加载到单卡，V100 不支持 bf16 所以没走 bfloat16。
    注意力默认用 sdpa，eager 的 FP16 注意力在 V100 上会算出 NaN。
    prompt_mode 为 raw 时按论文的做法纯续写，把前缀补成推理模型的固定格式；
    n 大于 1 时一次采样多条，靠 num_return_sequences 提高吞吐。
    """
    backend_type = "transformers"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self._model = None
        self._tokenizer = None
        self._device = model_cfg.get("device", "cuda:0")

    @property
    def model(self):
        """机制实验直接访问模型(触发加载)。"""
        self._load()
        return self._model

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self.model_cfg["name"]
        dtype = torch.float16 if self.model_cfg.get("dtype", "float16") == "float16" else torch.bfloat16
        quant = self.model_cfg.get("quantize")
        # V100 上用 sdpa（eager 的 FP16 注意力在 V100 上会 NaN; sdpa 内部 softmax 用 FP32 更稳）
        attn_impl = self.model_cfg.get("attn_implementation", "sdpa")
        # 机制实验需手动逐层访问, 强制单卡(device_map 分散到多卡会导致跨设备问题)
        device = self.model_cfg.get("device", "cuda:0")
        load_kwargs = dict(
            torch_dtype=dtype,
            device_map={"": device},
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        if quant == "int8":
            load_kwargs["load_in_8bit"] = True
            load_kwargs.pop("torch_dtype")
        elif quant in ("awq", "gptq"):
            load_kwargs["quantization_config"] = None  # 需用户自行安装相应库并配置
            raise ValueError(
                f"量化模式 {quant} 需要先安装相应依赖(awq/gptq)，请在 config 中改用 int8 或 null。"
            )
        print(f"[transformers] 加载模型 {name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
        self._model.eval()
        if self._model.generation_config.pad_token_id is None:
            self._model.generation_config.pad_token_id = self._tokenizer.eos_token_id
        print("[transformers] 模型加载完成。")

    def _build_prompt(self, prompt: str) -> str:
        if self.model_cfg.get("prompt_mode", "raw") == "chat":
            try:
                msgs = [{"role": "user", "content": prompt}]
                return self._tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return prompt
        # R1 蒸馏模型需要特殊 token 触发 <think> 推理格式
        return f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>"

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> List[str]:
        self._load()
        import torch

        prompt_text = self._build_prompt(prompt)
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs = dict(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        # 去掉输入前缀, 只保留新生成部分
        input_len = inputs["input_ids"].shape[1]
        new_ids = outputs[:, input_len:]
        texts = self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        return texts


# ---------------------------------------------------------------- 连本地 vLLM 服务

import requests
from typing import List


class VllmBackend(BaseBackend):
    """连本地 vLLM 的 OpenAI 兼容服务，批量采样的吞吐比 transformers 高。

    先在机器上把服务起起来，例如

        vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --dtype float16 --max-model-len 8192 --port 8000

    再把 config.yaml 里的 backend 改成 vllm，api_base 填 http://localhost:8000/v1。
    这里用 /v1/completions 而不是 chat 接口，因为论文本身就是纯续写，不做对话封装。
    """
    backend_type = "vllm"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.api_base = model_cfg.get("api_base", "http://localhost:8000/v1").rstrip("/")
        self.model_name = model_cfg.get("openai_model") or model_cfg.get("name")
        self.prompt_mode = model_cfg.get("prompt_mode", "raw")

    def _complete(self, prompt: str, n: int, temperature: float, top_p: float, max_tokens: int) -> List[str]:
        url = f"{self.api_base}/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": None,
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return [choice["text"] for choice in data["choices"]]

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> List[str]:
        # vLLM 的 /v1/completions 直接接收原始文本前缀, 与论文的续写方式一致
        return self._complete(prompt, n, temperature, top_p, max_tokens)


# ---------------------------------------------------------------- 连远程 OpenAI 兼容接口

import os
import requests
from typing import List


class OpenAIBackend(BaseBackend):
    """OpenAI 兼容 API 后端（DeepSeek / DashScope / SiliconFlow 等）。

    - 使用 /v1/chat/completions, 便于对接各家推理模型（如 deepseek-reasoner）。
    - 注意: 部分推理模型（如 deepseek-reasoner）不允许/忽略 temperature 参数,
      若服务端报错可把 temperature/top_p 置为 null。
    """
    backend_type = "openai"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.api_base = (model_cfg.get("api_base") or "https://api.deepseek.com/v1").rstrip("/")
        key_env = model_cfg.get("api_key_env") or "OPENAI_API_KEY"
        self.api_key = os.environ.get(key_env, "")
        self.model_name = model_cfg.get("openai_model") or model_cfg.get("name")

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> List[str]:
        if not self.api_key:
            raise RuntimeError(
                f"未找到 API Key: 请设置环境变量 {self.model_cfg.get('api_key_env') or 'OPENAI_API_KEY'}"
            )
        url = f"{self.api_base}/chat/completions"
        # 推理模型常见限制: temperature/top_p 需为 null
        body_temperature = temperature if temperature is not None else None
        body_top_p = top_p if top_p is not None else None
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": body_temperature,
            "top_p": body_top_p,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.post(url, json=payload, headers=headers, timeout=600)
        if r.status_code in (400, 422):
            # 服务端可能不支持 temperature/top_p, 去掉重试
            payload["temperature"] = None
            payload["top_p"] = None
            r = requests.post(url, json=payload, headers=headers, timeout=600)
        r.raise_for_status()
        data = r.json()
        return [choice["message"]["content"] or "" for choice in data["choices"]]
