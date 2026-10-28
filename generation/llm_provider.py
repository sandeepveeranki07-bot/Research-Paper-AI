"""Configurable LLM provider interface supporting OpenAI, Anthropic, Gemini, Groq, and Ollama.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract interface for LLM completion providers."""

    @abstractmethod
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response text from LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs (vLLM, LocalAI, Groq)."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.llm_api_key or ""
        self.model = settings.llm_model or "gpt-4o-mini"
        self.base_url = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = settings.llm_timeout_seconds
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude API."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.llm_api_key or ""
        self.model = settings.llm_model or "claude-3-5-sonnet-20241022"
        self.timeout = settings.llm_timeout_seconds
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        system_content = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content += m["content"] + "\n"
            else:
                user_messages.append(m)

        payload = {
            "model": self.model,
            "system": system_content.strip(),
            "messages": user_messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini API."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.llm_api_key or ""
        self.model = settings.llm_model or "gemini-1.5-flash"
        self.timeout = settings.llm_timeout_seconds

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        contents = []
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        generation_config: Dict[str, Any] = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json={"contents": contents, "generationConfig": generation_config})
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaProvider(BaseLLMProvider):
    """Provider for local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = (settings.llm_base_url or "http://localhost:11434").rstrip("/")
        self.model = settings.llm_model or "llama3"
        self.timeout = settings.llm_timeout_seconds

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json" if json_mode else ""
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]


class ExtractiveLocalSynthesizer(BaseLLMProvider):
    """Deterministic, grounded extractive synthesis engine.

    Used when no external API key is provided, or as an offline fallback.
    Extracts direct claims from retrieved evidence chunks, attaches verifiable page citations,
    and returns compliant JSON without making calls to remote servers.
    """

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].upper()}:\n{m['content']}\n\n"

        # Check if context states NO RELEVANT EVIDENCE FOUND
        if "NO RELEVANT EVIDENCE FOUND" in prompt or "--- EVIDENCE CHUNK" not in prompt:
            return json.dumps({
                "answer": "The available papers do not establish an answer to this question because no relevant evidence was retrieved from the indexed collection.",
                "is_evidence_sufficient": False,
                "claims": [],
                "interpretations_and_inferences": [],
                "overall_confidence": "insufficient_evidence"
            }, indent=2)

        # Parse evidence chunks from the prompt
        chunk_blocks = re.findall(
            r"--- EVIDENCE CHUNK \[\d+\] ---\s+Chunk ID: (.*?)\s+Document: (.*?) \((.*?)\)\s+Page\(s\): p(?:p)?\. (.*?)(?: \| Section: (.*?))?\s+Similarity Score: (.*?)\s+Text:\s+(.*?)(?=\n--- EVIDENCE CHUNK|\nUSER:|\Z)",
            prompt,
            re.DOTALL
        )

        if not chunk_blocks:
            return json.dumps({
                "answer": "The available papers do not establish an answer to this question.",
                "is_evidence_sufficient": False,
                "claims": [],
                "interpretations_and_inferences": [],
                "overall_confidence": "insufficient_evidence"
            }, indent=2)

        # Extract factual sentences with citations
        claims = []
        answer_sentences = []

        for b in chunk_blocks[:4]:
            cid = b[0].strip()
            doc_title = b[1].strip()
            filename = b[2].strip()
            page_str = b[3].strip()
            score_str = b[5].strip()
            text = b[6].strip()

            # Parse page number
            page_match = re.search(r"\d+", page_str)
            page_num = int(page_match.group(0)) if page_match else 1

            # Extract substantive sentences from text
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]
            if sentences:
                selected_sent = sentences[0]
                # Clean sentence
                selected_sent = re.sub(r"\s+", " ", selected_sent)

                claim_text = selected_sent
                formatted_cite = f"[{doc_title}, p. {page_num}]"

                claims.append({
                    "claim_text": claim_text,
                    "source_document": doc_title,
                    "source_page": page_num,
                    "source_chunk_id": cid,
                    "supporting_evidence": selected_sent,
                    "evidence_relevance": "direct",
                    "confidence": "high"
                })
                answer_sentences.append(f"{claim_text} {formatted_cite}")

        if not claims:
            return json.dumps({
                "answer": "The available papers do not establish an answer to this question based on the retrieved evidence.",
                "is_evidence_sufficient": False,
                "claims": [],
                "interpretations_and_inferences": [],
                "overall_confidence": "insufficient_evidence"
            }, indent=2)

        synthesis_answer = " ".join(answer_sentences)

        return json.dumps({
            "answer": synthesis_answer,
            "is_evidence_sufficient": True,
            "claims": claims,
            "interpretations_and_inferences": [
                "Findings are grounded directly in the cited source passages above."
            ],
            "overall_confidence": "high"
        }, indent=2)


def get_llm_provider(settings: Optional[Settings] = None) -> BaseLLMProvider:
    """Factory function to instantiate the configured LLM provider."""
    s = settings or get_settings()
    provider_name = s.llm_provider.lower().strip()

    # If an API key is provided, instantiate the respective cloud provider
    if s.llm_api_key and s.llm_api_key.strip():
        if provider_name in ("openai", "custom"):
            return OpenAIProvider(s)
        elif provider_name == "anthropic":
            return AnthropicProvider(s)
        elif provider_name == "gemini":
            return GeminiProvider(s)
        elif provider_name == "groq":
            return OpenAIProvider(Settings(
                llm_base_url="https://api.groq.com/openai/v1",
                llm_model=s.llm_model or "llama-3.3-70b-versatile",
                llm_api_key=s.llm_api_key
            ))

    if provider_name == "ollama":
        return OllamaProvider(s)

    logger.info("No LLM_API_KEY detected. Using deterministic extractive local synthesizer.")
    return ExtractiveLocalSynthesizer()
