"""Minimal model clients.

  gemini     Gemini API generateContent (Google AI Studio key). The paper's verifier, Gemini 3.8 Flash, uses this.
  chat       OpenAI-compatible /chat/completions (example inference, or another verifier)
  responses  OpenAI /responses (another verifier)
Rate limits, server errors and connection errors are retried with backoff and never count as attempts.
"""
import base64
import json
import os
import random
import time
from dataclasses import dataclass

import requests


@dataclass
class Reply:
    text: str
    finish: str  # "stop" | "length" | "refused" | other provider-specific reason


class GeminiLLM:
    def __init__(self, model="gemini-3.8-flash", api_key=None, max_tokens=65536, thinking_level="high", timeout=1200):
        if not api_key:
            raise ValueError("Gemini needs an API key (set GEMINI_API_KEY)")
        self.headers = {"x-goog-api-key": api_key}
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.model, self.timeout = model, timeout
        self.generation_config = {"maxOutputTokens": max_tokens}
        if thinking_level:
            self.generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    def complete(self, system, user, images=()):
        """user: text; images: (caption, png_bytes) pairs placed after the text."""
        parts = [{"text": user}]
        for caption, png in images:
            parts += [{"text": caption}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}}]
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": parts}],
                "generationConfig": self.generation_config}
        d = _post_with_retries(lambda: requests.post(self.url, headers=self.headers, json=body, timeout=(30, self.timeout)))
        candidate = (d.get("candidates") or [{}])[0]
        text = "\n".join(p["text"] for p in candidate.get("content", {}).get("parts", []) if "text" in p and not p.get("thought"))
        reason = candidate.get("finishReason")
        return Reply(text, {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "refused"}.get(reason, str(reason)))


class OpenAILLM:
    def __init__(self, model, api="chat", base_url=None, api_key=None, max_tokens=64000, reasoning_effort=None,
                 extra_body=None, timeout=3600):
        from openai import OpenAI
        self.model, self.api, self.max_tokens = model, api, max_tokens
        self.reasoning_effort, self.extra_body = reasoning_effort, extra_body or {}
        # base_url/api_key default to OPENAI_BASE_URL / OPENAI_API_KEY
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=20)

    def complete(self, system, user, images=()):
        if self.api == "responses":
            return self._responses(system, user, images)
        return self._chat(system, user, images)

    def _chat(self, system, user, images):
        content = user  # plain text unless there are images
        if images:
            content = [{"type": "text", "text": user}]
            for caption, png in images:
                content += [{"type": "text", "text": caption}, {"type": "image_url", "image_url": {"url": _data_url(png)}}]
        kwargs = dict(model=self.model, max_completion_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        choice = self.client.chat.completions.create(**kwargs).choices[0]
        finish = {"stop": "stop", "length": "length", "content_filter": "refused"}.get(choice.finish_reason, str(choice.finish_reason))
        return Reply(choice.message.content or "", finish)

    def _responses(self, system, user, images):
        content = [{"type": "input_text", "text": user}]
        for caption, png in images:
            content += [{"type": "input_text", "text": caption}, {"type": "input_image", "image_url": _data_url(png)}]
        kwargs = dict(model=self.model, max_output_tokens=self.max_tokens, store=False,
                      input=[{"role": "system", "content": system}, {"role": "user", "content": content}])
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        r = self.client.responses.create(**kwargs)
        if r.status == "completed":
            finish = "stop"
        else:
            reason = getattr(r.incomplete_details, "reason", None)
            finish = {"max_output_tokens": "length", "content_filter": "refused"}.get(reason, f"incomplete:{reason or r.status}")
        return Reply(r.output_text or "", finish)


def _post_with_retries(send, max_wait=60):
    delay = 5
    while True:
        try:
            r = send()
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (408, 429) and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(delay * (1 + random.random()))
        delay = min(2 * delay, max_wait)


def _data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def add_verifier_args(parser):
    g = parser.add_argument_group("verifier (defaults: the paper's verifier, Gemini 3.8 Flash with thinking level high)")
    g.add_argument("--model", default="gemini-3.8-flash")
    g.add_argument("--api", choices=("gemini", "chat", "responses"), default="gemini")
    g.add_argument("--thinking-level", default="high", help="gemini only")
    g.add_argument("--base-url", help="chat/responses: defaults to $OPENAI_BASE_URL or api.openai.com")
    g.add_argument("--api-key-env", help="environment variable holding the API key "
                                         "(default GEMINI_API_KEY for gemini, OPENAI_API_KEY otherwise)")
    g.add_argument("--max-tokens", type=int, help="output limit (default 65536 for gemini, 16384 otherwise)")
    g.add_argument("--reasoning-effort", help="chat/responses: e.g. low / medium / high; omitted if unset")
    g.add_argument("--extra-body", type=json.loads, help="chat/responses: provider-specific JSON")


def verifier_from_args(args):
    if args.api == "gemini":
        return GeminiLLM(args.model, api_key=os.environ.get(args.api_key_env or "GEMINI_API_KEY"),
                         max_tokens=args.max_tokens or 65536, thinking_level=args.thinking_level)
    return OpenAILLM(args.model, api=args.api, base_url=args.base_url,
                     api_key=os.environ.get(args.api_key_env or "OPENAI_API_KEY"), max_tokens=args.max_tokens or 16384,
                     reasoning_effort=args.reasoning_effort, extra_body=args.extra_body, timeout=1200)
