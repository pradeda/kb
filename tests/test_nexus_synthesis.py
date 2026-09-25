from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from kb_search_api import (
    NexusRelevanceRequest,
    RerankerUnavailable,
    kb_synthesize_nexus_relevance,
    _call_synthesis_model,
)
from kb_v2 import Candidate


def request() -> NexusRelevanceRequest:
    return NexusRelevanceRequest(
        query="Nexus Fable 5 GPT 5.6 Sol OpenCode OpenRouter",
        video_title="Fable 5 vs GPT 5.6 Sol",
        video_summary="Comparison of model capability, cost and workflows.",
        initial_assessment="No direct Nexus relevance.",
        tools_models=["Fable 5", "GPT 5.6 Sol"],
    )


def candidate(entry_id: int, score: float) -> Candidate:
    """One ranked homelab candidate, as kb_v2's retrieval lane returns it."""
    return Candidate(
        corpus="homelab",
        entry_id=entry_id,
        title=f"KB entry {entry_id}",
        content="x" * 3000,
        summary=None,
        tags=None,
        source=None,
        date=None,
        distance=0.1,
        relevance=0.95,
        final_score=score,
    )


def retriever(*items: Candidate):
    """Patch target for the v2 retrieval lane the synthesis endpoint depends on."""
    return patch("kb_search_api._synthesis_retriever", lambda query: list(items))


class NexusSynthesisTests(unittest.TestCase):
    def test_synthesis_prompt_requires_concrete_nexus_target(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), patch(
            "kb_search_api.SYNTHESIS_MODEL", "google/gemini-2.5-flash-lite"
        ), patch("kb_search_api.httpx.post") as post:
            post.return_value.json.return_value = {
                "choices": [{"message": {"content": json.dumps({"ok": True})}}]
            }
            self.assertEqual(_call_synthesis_model({"item": {}, "kb_entries": []}), {"ok": True})
        prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("safe first check", prompt)
        self.assertIn("Generic AI interest", prompt)
        self.assertIn("not proof that the item is useful", prompt)

    def test_confirmed_response_keeps_only_retrieved_provenance(self) -> None:
        model_output = {
            "answer": "Directly relevant to the existing Nexus model workflow.",
            "kb_match_confirmed": True,
            "operational_relevance": "direct",
            "supporting_evidence": [
                {"entry_id": 400, "match_reason": "Same model family and API tiers."},
                {"entry_id": 401, "match_reason": "Same benchmark comparison."},
            ],
        }
        with retriever(
            candidate(400, 0.91), candidate(401, 0.82), candidate(999, 0.59)
        ), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                request(), x_kb_synthesis_token="test"
            )

        self.assertEqual(response.status, "operationally_relevant")
        self.assertTrue(response.kb_match_confirmed)
        self.assertEqual(response.operational_relevance, "direct")
        self.assertEqual(
            [item.entry_id for item in response.supporting_entries], [400, 401]
        )
        self.assertEqual(
            response.supporting_entries[0].match_reason,
            "Same model family and API tiers.",
        )
        self.assertEqual(response.provenance.model_call_count, 1)
        context = provider.call_args.args[0]
        self.assertEqual(context["item"]["source_type"], "video")
        self.assertEqual([item["entry_id"] for item in context["kb_entries"]], [400, 401])
        self.assertTrue(all(len(item["excerpt"]) == 2000 for item in context["kb_entries"]))

    def test_article_source_type_reaches_synthesis_context(self) -> None:
        article_request = request().model_copy(update={"source_type": "article"})
        model_output = {
            "answer": "The article matches existing KB knowledge.",
            "kb_match_confirmed": True,
            "operational_relevance": "not_confirmed",
            "supporting_evidence": [{
                "entry_id": 400,
                "match_reason": "Both cover the same model release.",
            }],
        }
        with retriever(candidate(400, 0.91)), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            kb_synthesize_nexus_relevance(
                article_request, x_kb_synthesis_token="test"
            )
        self.assertEqual(provider.call_args.args[0]["item"]["source_type"], "article")

    def test_article_uses_lower_related_knowledge_candidate_threshold(self) -> None:
        article_request = request().model_copy(update={"source_type": "article"})
        model_output = {
            "answer": "The article has related KB knowledge but no operational impact.",
            "kb_match_confirmed": True,
            "operational_relevance": "not_confirmed",
            "supporting_evidence": [{
                "entry_id": 400,
                "match_reason": "Both discuss adoption of open-weight models.",
            }],
        }
        with retriever(candidate(400, 0.51)), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                article_request, x_kb_synthesis_token="test"
            )
        self.assertEqual(response.status, "kb_match_only")
        provider.assert_called_once()

    def test_model_cannot_cite_an_entry_not_returned_by_retrieval(self) -> None:
        model_output = {
            "answer": "Unsupported citation.",
            "kb_match_confirmed": True,
            "operational_relevance": "direct",
            "supporting_evidence": [
                {"entry_id": 777, "match_reason": "Unavailable source."},
            ],
        }
        with retriever(candidate(400, 0.91)), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ), patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(
                    request(), x_kb_synthesis_token="test"
                )
        self.assertEqual(raised.exception.status_code, 502)

    def test_no_strong_match_skips_model_call(self) -> None:
        with retriever(candidate(400, 0.59)), patch(
            "kb_search_api._call_synthesis_model"
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                request(), x_kb_synthesis_token="test"
            )
        self.assertEqual(response.status, "not_confirmed")
        self.assertFalse(response.kb_match_confirmed)
        self.assertEqual(response.operational_relevance, "not_confirmed")
        self.assertFalse(response.connection_confirmed)
        self.assertEqual(response.provenance.model_call_count, 0)
        provider.assert_not_called()

    def test_topic_match_does_not_become_operational_relevance(self) -> None:
        model_output = {
            "answer": (
                "KB #400 corroborates the model-family topic. No concrete Nexus service, "
                "workflow, hardware or roadmap impact is confirmed."
            ),
            "kb_match_confirmed": True,
            "operational_relevance": "not_confirmed",
            "supporting_evidence": [{
                "entry_id": 400,
                "match_reason": "Both discuss the GPT-5.6 Sol, Terra and Luna model family.",
            }],
        }
        with retriever(candidate(400, 0.93)), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ), patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                request(), x_kb_synthesis_token="test"
            )
        self.assertEqual(response.status, "kb_match_only")
        self.assertTrue(response.kb_match_confirmed)
        self.assertEqual(response.operational_relevance, "not_confirmed")
        self.assertFalse(response.connection_confirmed)
        self.assertEqual([item.entry_id for item in response.supporting_entries], [400])
        self.assertIn("Sol, Terra and Luna", response.supporting_entries[0].match_reason)

    def test_model_comes_from_the_environment_not_a_code_default(self) -> None:
        """The deployed env file owns the model; the service invents nothing."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), patch(
            "kb_search_api.SYNTHESIS_MODEL", ""
        ), patch("kb_search_api.httpx.post") as post:
            with self.assertRaises(RuntimeError) as raised:
                _call_synthesis_model({"item": {}, "kb_entries": []})
        self.assertIn("KB_SYNTHESIS_MODEL", str(raised.exception))
        post.assert_not_called()

    def test_configured_model_reaches_the_request(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), patch(
            "kb_search_api.SYNTHESIS_MODEL", "vendor/model-from-env"
        ), patch("kb_search_api.httpx.post") as post:
            post.return_value.json.return_value = {
                "choices": [{"message": {"content": json.dumps({"ok": True})}}]
            }
            _call_synthesis_model({"item": {}, "kb_entries": []})
        self.assertEqual(post.call_args.kwargs["json"]["model"], "vendor/model-from-env")

    def test_missing_model_fails_the_endpoint_closed(self) -> None:
        with retriever(candidate(400, 0.91)), patch(
            "kb_search_api.SYNTHESIS_MODEL", ""
        ), patch("kb_search_api.httpx.post") as post, patch.dict(
            "os.environ", {"KB_SYNTHESIS_TOKEN": "test", "OPENROUTER_API_KEY": "test"}
        ):
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(request(), x_kb_synthesis_token="test")
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("KB_SYNTHESIS_MODEL", str(raised.exception.detail))
        post.assert_not_called()

    def test_unwired_retrieval_lane_fails_closed(self) -> None:
        """No retriever means 503, never an empty "no related knowledge" answer."""
        with patch("kb_search_api._synthesis_retriever", None), patch(
            "kb_search_api._call_synthesis_model"
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(request(), x_kb_synthesis_token="test")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail, {"reason": "retrieval_unavailable"})
        provider.assert_not_called()

    def test_missing_reranker_keeps_the_documented_reason(self) -> None:
        """The v2 lane reports reranker_unavailable; the endpoint keeps that body."""
        def broken(query: str):
            raise RuntimeError("reranker_unavailable")

        with patch("kb_search_api._synthesis_retriever", broken), patch(
            "kb_search_api._call_synthesis_model"
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            with self.assertRaises(RerankerUnavailable):
                kb_synthesize_nexus_relevance(request(), x_kb_synthesis_token="test")
        provider.assert_not_called()

    def test_other_retrieval_failures_are_named_in_the_503(self) -> None:
        def broken(query: str):
            raise RuntimeError("router_config_unavailable")

        with patch("kb_search_api._synthesis_retriever", broken), patch.dict(
            "os.environ", {"KB_SYNTHESIS_TOKEN": "test"}
        ):
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(request(), x_kb_synthesis_token="test")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail, {"reason": "router_config_unavailable"})

    def test_root_app_wires_the_synthesis_lane_to_the_v2_retriever(self) -> None:
        """The seam is the v2 app's own retriever, not a second implementation."""
        import kb_search_api

        with patch("kb_v2.Fts5Index.build", return_value=None):
            kb_search_api.create_root_app()
        try:
            self.assertIsNotNone(kb_search_api._synthesis_retriever)
            self.assertEqual(
                kb_search_api._synthesis_retriever.__name__, "retrieve_homelab"
            )
        finally:
            kb_search_api._synthesis_retriever = None

    def test_route_is_disabled_without_a_dedicated_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "kb_search_api._synthesis_retriever"
        ) as retrieval:
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(
                    request(), x_kb_synthesis_token=None
                )
        self.assertEqual(raised.exception.status_code, 503)
        retrieval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
