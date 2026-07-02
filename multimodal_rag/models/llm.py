
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from multimodal_rag.config import AppConfig, get_settings
from multimodal_rag.utils.helpers import load_image_base64_data_url

load_dotenv()


def get_chat_model(settings: AppConfig | None = None):
	from langchain_groq import ChatGroq

	settings = settings or get_settings()
	if not settings.llm_api_key:
		raise ValueError("GROQ_API_KEY is required for Groq chat models")
	return ChatGroq(
		model=settings.llm_model,
		temperature=settings.llm_temperature,
		api_key=settings.llm_api_key,
	)


def get_embeddings(settings: AppConfig | None = None) -> HuggingFaceInferenceAPIEmbeddings:
	settings = settings or get_settings()
	if not settings.hf_api_key:
		raise ValueError("HF_API_TOKEN is required for Hugging Face embeddings")
	return HuggingFaceInferenceAPIEmbeddings(
		api_key=settings.hf_api_key,
		model_name=settings.hf_embedding_model,
		api_url=settings.hf_api_url,
	)


def summarize_text(text: str, settings: AppConfig | None = None) -> str:
	model = get_chat_model(settings)
	response = model.invoke(
		[
			HumanMessage(
				content=(
					"Summarize the following PDF text for retrieval. Keep it compact, "
					"faithful, and information dense.\n\n"
					f"{text}"
				)
			)
		]
	)
	return str(response.content).strip()


def summarize_table(table_html: str, settings: AppConfig | None = None) -> str:
	model = get_chat_model(settings)
	response = model.invoke(
		[
			HumanMessage(
				content=(
					"Summarize this HTML table for retrieval. Focus on the key columns, "
					"trends, totals, comparisons, and notable values.\n\n"
					f"{table_html}"
				)
			)
		]
	)
	return str(response.content).strip()


def summarize_image(image_path: str | Path, settings: AppConfig | None = None) -> str:
	model = get_chat_model(settings)
	image_data_url = load_image_base64_data_url(image_path)
	response = model.invoke(
		[
			HumanMessage(
				content=[
					{
						"type": "text",
						"text": (
							"Summarize this PDF image, chart, or figure for retrieval. "
							"Capture the subject, visible labels, trends, and any values "
							"that matter for answering questions."
						),
					},
					{
						"type": "image_url",
						"image_url": {"url": image_data_url},
					},
				]
			)
		]
	)
	return str(response.content).strip()


def answer_question(question: str, documents: list[Document], settings: AppConfig | None = None) -> str:
	model = get_chat_model(settings)
	context_blocks: list[dict[str, object]] = [
		{
			"type": "text",
			"text": (
				"Use the retrieved PDF context to answer the question. "
				"If an image is included, inspect it directly. Cite the evidence "
				"from the provided context only.\n\nQuestion: "
				f"{question}"
			),
		}
	]

	for document in documents:
		kind = str(document.metadata.get("kind", "text")).lower()
		if kind == "image":
			image_path = document.metadata.get("image_path")
			if image_path:
				context_blocks.append(
					{
						"type": "image_url",
						"image_url": {"url": load_image_base64_data_url(image_path)},
					}
				)
			if document.page_content:
				context_blocks.append({"type": "text", "text": document.page_content})
			continue

		content = document.page_content.strip()
		if content:
			context_blocks.append({"type": "text", "text": content})

	response = model.invoke([HumanMessage(content=context_blocks)])
	return str(response.content).strip()


@dataclass(slots=True)
class LLMService:
	settings: AppConfig

	@property
	def model(self):
		return get_chat_model(self.settings)

	@property
	def embeddings(self) -> HuggingFaceInferenceAPIEmbeddings:
		return get_embeddings(self.settings)
