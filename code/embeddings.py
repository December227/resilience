# -*- coding: utf-8 -*-
"""句子嵌入模块。

论文使用 bert-large-nli-stsb-mean-tokens（即 sentence-transformers/stsb-bert-large）
计算句子余弦相似度，用于判断"被删除的推理句是否在下游重新出现"。
"""
from typing import List, Optional

import numpy as np

from utils import ensure_hf_env


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        ensure_hf_env()
        from sentence_transformers import SentenceTransformer
        print(f"[embedding] 加载句子嵌入模型 {model_name} ...")
        self.model = SentenceTransformer(model_name, device=device)
        self._cache = {}
        print("[embedding] 嵌入模型加载完成。")

    def embed(self, sentences: List[str]) -> np.ndarray:
        out = []
        missing = []
        for s in sentences:
            key = s
            if key in self._cache:
                out.append(self._cache[key])
            else:
                missing.append(s)
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for s, v in zip(missing, vecs):
                self._cache[s] = v
            out.extend(self._cache[s] for s in missing)
        return np.array(out, dtype=np.float32)

    def similarity(self, target: str, sentences: List[str]) -> np.ndarray:
        """目标句与若干句子的余弦相似度（已归一化）。"""
        t = self.embed([target])[0]
        sents = self.embed(sentences)
        return sents @ t

    def max_similarity(self, target: str, sentences: List[str]) -> float:
        if not sentences:
            return -1.0
        return float(self.similarity(target, sentences).max())
