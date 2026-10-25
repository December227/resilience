# -*- coding: utf-8 -*-
"""工具函数：环境变量、配置加载、随机种子。"""
import os


def ensure_hf_env():
    """HuggingFace 官方站不可达时使用 hf-mirror 镜像。"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_all(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
