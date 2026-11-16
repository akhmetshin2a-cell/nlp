"""Собрать полный каталог для загрузки на GitHub: python export_github.py."""
from pathlib import Path
import hashlib
import json
import shutil

from project_setup import ensure_fine_tuned_model

ROOT = Path(__file__).resolve().parent
FILES = [
    "00_transformers_overall.ipynb", "01_transformers_day01.ipynb",
    "app.py", "project_setup.py", "project_sources.json", "export_github.py",
    "baseline_features.py", "embedding_utils.py", "tokenization_utils.py",
    "README.md", "requirements.txt", ".gitignore", "baseline_model.pkl", "vectorizer.pkl",
    "baseline_results.txt", "fine_tuned_results.txt", "comparison_results.txt",
    "error_analysis.txt", "confusion_matrix_baseline.png", "confusion_matrix_finetuned.png",
]


def build_model_bundle():
    model = ensure_fine_tuned_model(ROOT)
    bundle = ROOT / "model_bundle"
    bundle.mkdir(exist_ok=True)
    manifest = {"format": 1, "description": "Own fine-tuned DistilBERT, exact original weights", "files": {}}
    for path in sorted(model.iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt"}:
            shutil.copy2(path, bundle / path.name)
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    parts = []
    digest = hashlib.sha256()
    with (model / "model.safetensors").open("rb") as stream:
        while content := stream.read(5 * 1024 * 1024):
            name = f"model.safetensors.part{len(parts):03d}"
            (bundle / name).write_bytes(content)
            parts.append({"name": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            digest.update(content)
    manifest["weights"] = {"filename": "model.safetensors", "size": sum(p["size"] for p in parts),
                           "sha256": digest.hexdigest(), "parts": parts}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    keep = set(manifest["files"]) | {p["name"] for p in parts} | {"manifest.json"}
    for path in bundle.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
    return bundle


def export_project():
    bundle = build_model_bundle()
    destination = ROOT / "github_ready"
    destination.mkdir(exist_ok=True)
    for name in FILES:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"Для поставки не хватает {name}")
        shutil.copy2(source, destination / name)
    shutil.copytree(bundle, destination / "model_bundle", dirs_exist_ok=True)
    shutil.copytree(ROOT / "tests", destination / "tests", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for path in destination.rglob("*"):
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Слишком большой файл поставки: {path}")
    print(f"Готово: {destination}\nЗагрузите всё содержимое, включая model_bundle и tests.")
    return destination


if __name__ == "__main__":
    export_project()
