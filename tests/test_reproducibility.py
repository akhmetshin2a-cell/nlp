import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import project_setup

ROOT = Path(__file__).resolve().parents[1]


class ModelBundleTests(unittest.TestCase):
    def make_bundle(self, root):
        bundle = root / "model_bundle"
        bundle.mkdir()
        content = b"model-weights-for-integrity-test"
        files = {"config.json": b"{}", "tokenizer_config.json": b"{}",
                 "tokenizer.json": b"{}", "vocab.txt": b"test\n"}
        for name, data in files.items():
            (bundle / name).write_bytes(data)
        part = bundle / "model.safetensors.part000"
        part.write_bytes(content)
        manifest = {"files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
                    "weights": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                "parts": [{"name": part.name, "size": len(content),
                                           "sha256": hashlib.sha256(content).hexdigest()}]}}
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return content, part

    def test_restore_and_reuse_without_network(self):
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            root = Path(directory)
            content, _ = self.make_bundle(root)
            target = project_setup.ensure_fine_tuned_model(root)
            self.assertEqual((target / "model.safetensors").read_bytes(), content)
            self.assertEqual(project_setup.ensure_fine_tuned_model(root), target)

    def test_corrupted_part_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, part = self.make_bundle(root)
            part.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                project_setup.ensure_fine_tuned_model(root)
            self.assertFalse((root / "fine_tuned_model/model.safetensors").exists())

    def test_missing_part_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, part = self.make_bundle(root)
            part.unlink()
            with self.assertRaises(FileNotFoundError):
                project_setup.ensure_fine_tuned_model(root)

    def test_no_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "model_bundle"):
                project_setup.ensure_fine_tuned_model(directory)

    def test_app_import_does_not_load_weights(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project_setup, "ROOT", Path(directory)):
            spec = importlib.util.spec_from_file_location("app_without_assets", ROOT / "app.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.load_model.cache_info().currsize, 0)
            self.assertIn("Введите", module.predict_sentiment(" "))
            self.assertEqual(module.load_model.cache_info().currsize, 0)


if __name__ == "__main__":
    unittest.main()
