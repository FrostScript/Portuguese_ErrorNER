"""DeepSeek 客户端：唯一负责与 API 通信的模块。

约束（见 README §八）：
    - key 只从环境变量或 `D:\\pt_ner_work\\.secrets\\deepseek.key` 读取，**绝不打印明文**；
    - 每次调用都落盘原始响应，保证可追溯、可离线重放；
    - 429 / 5xx 做指数退避重试，其余 4xx 直接失败（不刷额度）。

用法：
    from src.generate.client import DeepSeekClient
    c = DeepSeekClient()
    r = c.complete(prompt)          # 返回 Completion(text, usage, raw)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from src.common import env  # noqa: F401

SECRETS_FILE = Path("D:/pt_ner_work/.secrets/deepseek.key")
RAW_DIR = Path("D:/pt_ner_work/raw_responses")

DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
PLACEHOLDER = "sk-REPLACE_ME"


def load_api_key() -> str:
    """环境变量优先，其次密钥文件；两者都没有则抛错。不打印明文。"""
    v = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if v:
        return v
    if SECRETS_FILE.exists():
        raw = SECRETS_FILE.read_text(encoding="utf-8", errors="replace").strip()
        if "=" in raw and not raw.startswith("sk-"):
            raw = raw.split("=", 1)[1]
        raw = raw.strip().strip('"').strip("'")
        if raw and raw != PLACEHOLDER:
            return raw
    raise RuntimeError(
        f"未找到可用 API key：请把 key 写入 {SECRETS_FILE}，或设置环境变量 DEEPSEEK_API_KEY"
    )


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str
    raw_path: str
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """finish_reason=length 且正文为空 —— 推理把输出配额吃光了。"""
        return self.finish_reason == "length"


class TruncatedError(RuntimeError):
    """输出被 max_tokens 截断且正文为空（多为推理模型 thinking 占满配额）。"""


class DeepSeekClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE,
        temperature: float = 0.8,
        max_tokens: int = 8192,
        retries: int = 3,
        raw_dir: Path = RAW_DIR,
        api_key: str | None = None,
        reasoning_effort: str | None = "low",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.raw_dir = raw_dir
        # deepseek-flash 是推理模型：默认 effort=high 时 thinking 会吃掉 6.8k–7.4k 输出 token，
        # 正文只剩几百 token 甚至为空。实测 low 把 thinking 降到约 3.4k，11 条可一次出完。
        self.reasoning_effort = reasoning_effort
        self._key = api_key or load_api_key()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ---------------------------------------------------------------- 内部
    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _save_raw(self, tag: str, payload: dict, response: dict | None, error: str | None) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.raw_dir / f"{stamp}_{tag}.json"
        body = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request": {k: v for k, v in payload.items() if k != "messages"},
            "prompt_preview": payload.get("messages", [{}])[-1].get("content", "")[:400],
            "response": response,
            "error": error,
        }
        path.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
        return str(path)

    # ---------------------------------------------------------------- 公开
    def complete(self, prompt: str, tag: str = "gen", system: str | None = None,
                 extra_body: dict | None = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)
        if self.reasoning_effort:
            payload.setdefault("reasoning_effort", self.reasoning_effort)

        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as e:
                detail = e.read(300).decode("utf-8", errors="replace")
                last_err = f"HTTP {e.code}: {detail}"
                # 4xx（除 429）不重试，避免无效消耗
                if e.code != 429 and 400 <= e.code < 500:
                    self._save_raw(tag, payload, None, last_err)
                    raise RuntimeError(f"API 拒绝请求（{last_err}）") from e
                time.sleep(2 ** attempt)
                continue
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
                continue

            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "") or ""
            usage = data.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            rt = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            raw_path = self._save_raw(tag, payload, data, None)
            finish = choice.get("finish_reason", "")

            self.calls += 1
            self.prompt_tokens += pt
            self.completion_tokens += ct
            if finish == "length" and not text.strip():
                raise TruncatedError(
                    f"输出被截断：finish_reason=length，正文为空，reasoning_tokens={rt}/{ct}。"
                    f"推理模型把 max_tokens({self.max_tokens}) 用光了，需调大或减少每批条数。"
                    f"原始响应：{raw_path}"
                )
            return Completion(
                text=text,
                prompt_tokens=pt,
                completion_tokens=ct,
                model=data.get("model", self.model),
                finish_reason=finish,
                raw_path=raw_path,
                reasoning_tokens=rt,
            )

        self._save_raw(tag, payload, None, last_err)
        raise RuntimeError(f"重试 {self.retries} 次仍失败：{last_err}")
