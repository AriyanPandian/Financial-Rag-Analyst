from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config import Settings
from app.service import RAGService


class FakeOpenAI:
    def __init__(self, answer: str):
        self.answer = answer
        self.embeddings = SimpleNamespace(create=self.create_embeddings)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    @staticmethod
    def create_embeddings(*, model, input):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input]
        )

    def create_completion(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.answer))
            ]
        )


class RAGServiceTests(unittest.TestCase):
    def make_service(self, directory: str, answer: str) -> RAGService:
        settings = Settings(
            openai_api_key="test-key",
            chat_model="test-chat",
            embedding_model="test-embedding",
            data_dir=Path(directory),
            max_upload_bytes=100_000,
            api_access_token="",
        )
        with patch("app.service.OpenAI", return_value=FakeOpenAI(answer)):
            return RAGService(settings)

    def test_answers_include_only_retrieved_citations_and_exact_excerpt(self):
        with TemporaryDirectory() as directory:
            service = self.make_service(
                directory, "Revenue was $42 million [S1]. Growth was 8% [S99]."
            )
            service.ingest(
                "annual-report.html",
                b"<html><body><h2>Results</h2><p>Revenue was $42 million.</p></body></html>",
            )

            result = service.answer("What was revenue?", [])

            self.assertEqual(
                result["answer"],
                "Revenue was $42 million [S1]. Growth was 8% [unverified citation].",
            )
            self.assertEqual(result["citations"][0]["id"], "S1")
            self.assertEqual(result["citations"][0]["section"], "Results")
            self.assertEqual(result["citations"][0]["excerpt"], "Revenue was $42 million.")

    def test_uncited_model_answer_fails_closed(self):
        with TemporaryDirectory() as directory:
            service = self.make_service(directory, "Revenue was $42 million.")
            service.ingest("report.txt", b"Revenue was $42 million.")

            result = service.answer("What was revenue?", [])

            self.assertEqual(result["citations"], [])
            self.assertIn("could not produce a verifiable", result["answer"])

    def test_document_can_be_removed_from_index(self):
        with TemporaryDirectory() as directory:
            service = self.make_service(directory, "The filing states [S1].")
            uploaded = service.ingest("report.txt", b"The filing states.")

            service.delete_document(uploaded["id"])

            self.assertEqual(service.documents(), [])
            self.assertEqual(
                service.answer("What does it state?", [])["citations"], []
            )


if __name__ == "__main__":
    unittest.main()
