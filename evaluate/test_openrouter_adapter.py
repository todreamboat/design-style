"""Offline regression checks; no real credentials, uploads or API charges."""
import base64
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import workflow_benchmark as w


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = w.load(Path(__file__).with_name("workflow_config.json"))

    def test_reference_order_and_payload(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-fixture"}):
            client = w.Client(self.cfg, ".")
            with patch.object(client, "request", return_value={}) as request, patch.object(w, "image_uri", side_effect=lambda p: "data:" + p):
                client.image("task", ["input", "ref1", "ref2"])
                kind, payload = request.call_args.args
                self.assertEqual(kind, "image")
                self.assertEqual([r["image_url"]["url"] for r in payload["input_references"]], ["data:input", "data:ref1", "data:ref2"])
                self.assertNotIn("input", payload)
                self.assertEqual(payload["n"], 1)

    def test_decode_and_reject_bad_outputs(self):
        value = base64.b64encode(b"fixture").decode()
        self.assertEqual(w.output_bytes({"data": [{"b64_json": value}]}, self.cfg), b"fixture")
        for response in ({"data": []}, {"data": [{"b64_json": "!"}]}, {"data": [{"b64_json": value}] * 2}):
            with self.assertRaises(ValueError):
                    w.output_bytes(response, self.cfg)

    def test_paired_seed_is_sent_without_changing_reference_order(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-fixture"}):
            client = w.Client(self.cfg, ".")
            with patch.object(client, "request", return_value={}) as request, patch.object(w, "image_uri", side_effect=lambda p: "data:" + p):
                client.image("task", ["ref1", "ref2"], seed=20261002)
                payload = request.call_args.args[1]
                self.assertEqual(payload["seed"], 20261002)
                self.assertEqual([x["image_url"]["url"] for x in payload["input_references"]], ["data:ref1", "data:ref2"])

    def test_key_cannot_be_routed_to_another_host(self):
        for url in ("https://example.com/api", "http://openrouter.ai/api", "https://openrouter.ai/api?key=x"):
            with patch.dict(os.environ, {"FASHION_IMAGE_ENDPOINT": url}):
                with self.assertRaises(ValueError):
                    w.endpoint(self.cfg, "image")

    def test_local_key_loading_preserves_process_environment(self):
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value="OPENROUTER_API_KEY=local-fixture\nUNRELATED=value\n"):
            with patch.dict(os.environ, {}, clear=True):
                w.load_local_credentials()
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "local-fixture")
                self.assertNotIn("UNRELATED", os.environ)
                os.environ["OPENROUTER_API_KEY"] = "process-fixture"
                w.load_local_credentials()
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "process-fixture")

    def test_provider_routing_and_response_metadata(self):
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {"id": "fixture-id", "model": "fixture-model", "provider": "fixture-provider"}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-fixture", "FASHION_VLM_ENDPOINT": ""}), patch("requests.post", return_value=response) as post, patch.object(w, "save") as save:
            w.Client(self.cfg, ".").request("vlm", {"model": self.cfg["vlm_model"], "messages": []})
            self.assertFalse(post.call_args.kwargs["json"]["provider"]["allow_fallbacks"])
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            self.assertEqual(save.call_args.args[1]["provider"], "fixture-provider")
            self.assertNotIn("offline-fixture", str(save.call_args))

    def test_read_timeout_is_distinguished_and_secret_is_not_logged(self):
        import requests
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-secret"}), patch("requests.post", side_effect=requests.ReadTimeout()), patch.object(w, "save") as save:
            with self.assertRaises(RuntimeError):
                w.Client(self.cfg, ".").request("image", {"model": self.cfg["image_model"]})
            error = save.call_args.args[1]
            self.assertEqual(error["exception_type"], "ReadTimeout")
            self.assertEqual(error["read_timeout_seconds"], self.cfg["timeout_seconds"])
            self.assertNotIn("offline-secret", str(save.call_args))

    def test_style_unknowns_and_duplicate_dimensions(self):
        rows = [{"dimension": "silhouette", "status": "partial", "reference_evidence": "straight", "generated_evidence": "slightly flared"},
                {"dimension": "structure_cut", "status": "unknown", "reference_evidence": "obscured", "generated_evidence": "obscured"}]
        result = w.style_checks({"style_checks": rows}, ["silhouette", "structure_cut"])
        self.assertEqual(result["style_retention_proxy"], 0.25)
        self.assertEqual(result["known_only_style_proxy"], 0.5)
        with self.assertRaises(ValueError):
            w.style_checks({"style_checks": [rows[0], rows[0]]}, ["silhouette", "structure_cut"])
        rows[0]["status"] = "unknown"
        self.assertIsNone(w.style_checks({"style_checks": rows}, ["silhouette", "structure_cut"])["known_only_style_proxy"])


if __name__ == "__main__":
    unittest.main()
