from __future__ import annotations

from openai import OpenAI
from ragas.llms import llm_factory

from app.llm import LLMService


def create_ragas_llm(llm_service: LLMService):
    client = OpenAI(
        base_url=llm_service.base_url,
        api_key=llm_service.api_key,
        timeout=llm_service.request_timeout,
    )

    return llm_factory(
        llm_service.model,
        provider="openai",
        client=client,
        temperature=0.0,
        top_p=0.95,
        max_tokens=20_000,
    )
