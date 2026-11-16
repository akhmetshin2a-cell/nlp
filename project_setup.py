"""Восстановление проекта из исходных источников и собственной сохранённой модели.

Запуск: python project_setup.py
Интернет нужен для базового DistilBERT и SST-2. Fine-tuned веса восстанавливаются
побайтно из model_bundle, без подмены checkpoint другой sentiment-моделью.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def _sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _sources():
    return json.loads((ROOT / "project_sources.json").read_text(encoding="utf-8"))


def _download(url, target, expected_sha256):
    target = Path(target)
    if target.exists():
        if _sha256(target) != expected_sha256:
            raise ValueError(f"Файл {target} изменён или повреждён. Проверьте его перед заменой.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка {target.name}", flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=180) as response:
                shutil.copyfileobj(response, output)
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"Неверная SHA-256 для {target.name}")
        temporary.replace(target)
    except OSError as exc:
        raise RuntimeError(f"Не удалось загрузить {url}. Проверьте интернет и повторите запуск.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_base_model(root=None):
    root = Path(root).resolve() if root else ROOT
    spec = _sources()["model"]
    url = f"https://huggingface.co/{spec['model_name']}/resolve/{spec['revision']}"
    tokenizer_dir = root / "artifacts/day01/tokenizer"
    model_dir = root / "artifacts/day02/model"
    for name, digest in spec["files_sha256"].items():
        # save_pretrained может менять tokenizer_config.json, поэтому валидный
        # существующий токенизатор сохраняем; свежие загрузки всегда сверяем.
        target = model_dir / name if name == "model.safetensors" else tokenizer_dir / name
        if not target.exists():
            _download(f"{url}/{name}", target, digest)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "config.json").exists():
        shutil.copy2(tokenizer_dir / "config.json", model_dir / "config.json")
    return tokenizer_dir, model_dir


def _prepare_frame(raw):
    frame = raw.rename(columns={"sentence": "text"})[["text", "label"]].copy()
    frame = frame.dropna(subset=["text", "label"])
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("SST-2 содержит неожиданные метки")
    frame["text"] = frame["text"].str.strip()
    frame = frame.loc[frame["text"].str.len() > 0].copy()
    frame["label"] = frame["label"].astype("int64")
    frame["_key"] = frame["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    counts = frame.groupby("_key")["label"].nunique()
    frame = frame.loc[~frame["_key"].isin(counts[counts > 1].index)]
    return frame.drop_duplicates("_key").reset_index(drop=True)


def ensure_dataset(root=None):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    root = Path(root).resolve() if root else ROOT
    spec = _sources()["dataset"]
    destination = root / "data/processed"
    expected = spec["csv_sha256"]
    if all((destination / f"{name}.csv").exists() for name in expected):
        for name, digest in expected.items():
            if _sha256(destination / f"{name}.csv") != digest:
                raise ValueError(f"Выборка {name}.csv отличается от исходного эксперимента")
        return destination / "train.csv"
    raw_dir = root / "data/raw"
    for split, digest in spec["source_sha256"].items():
        url = (f"https://huggingface.co/datasets/{spec['id']}/resolve/"
               f"{spec['revision']}/data/{split}-00000-of-00001.parquet")
        _download(url, raw_dir / f"{split}.parquet", digest)
    train_pool = _prepare_frame(pd.read_parquet(raw_dir / "train.parquet"))
    test = _prepare_frame(pd.read_parquet(raw_dir / "validation.parquet"))
    train_pool = train_pool.loc[~train_pool["_key"].isin(test["_key"])].reset_index(drop=True)
    study, _ = train_test_split(train_pool, train_size=6000,
                                stratify=train_pool["label"], random_state=spec["seed"])
    train, validation = train_test_split(study, train_size=5000, test_size=1000,
                                        stratify=study["label"], random_state=spec["seed"])
    destination.mkdir(parents=True, exist_ok=True)
    frames = {"train": train, "validation": validation, "test": test}
    # Проверяем результат до замены любого существующего CSV.
    for name, frame in frames.items():
        content = frame[["text", "label"]].to_csv(index=False).encode("utf-8")
        if hashlib.sha256(content).hexdigest() != expected[name]:
            raise ValueError(f"Не воспроизвелось разбиение {name}; используйте проверенные версии из requirements.txt")
    for name, frame in frames.items():
        frame[["text", "label"]].to_csv(destination / f"{name}.csv", index=False, encoding="utf-8")
    train_pool[["text", "label"]].to_csv(destination / "train_pool.csv", index=False, encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return destination / "train.csv"


def ensure_fine_tuned_model(root=None):
    root = Path(root).resolve() if root else ROOT
    model_dir = root / "fine_tuned_model"
    needed = ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
    if all((model_dir / name).is_file() for name in needed):
        return model_dir
    bundle = root / "model_bundle"
    if not (bundle / "manifest.json").exists():
        raise FileNotFoundError(
            "Не найдены fine_tuned_model и model_bundle/manifest.json. "
            "Загрузите весь репозиторий вместе с model_bundle; "
            "либо обучите модель в День 5. Исходные веса не заменяются другой моделью."
        )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    weights = manifest["weights"]
    for name, digest in manifest["files"].items():
        if Path(name).name != name or _sha256(bundle / name) != digest:
            raise ValueError(f"Неверный файл model_bundle/{name}")
    for part in weights["parts"]:
        name = part["name"]
        if Path(name).name != name or not (bundle / name).is_file():
            raise FileNotFoundError(f"Отсутствует часть model_bundle/{name}; загрузите все части")
        if (bundle / name).stat().st_size != part["size"] or _sha256(bundle / name) != part["sha256"]:
            raise ValueError(f"Повреждена часть model_bundle/{name}")
    print("Восстановление fine-tuned модели из model_bundle", flush=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        staged = Path(temporary) / "model.safetensors"
        with staged.open("wb") as output:
            for part in weights["parts"]:
                with (bundle / part["name"]).open("rb") as source:
                    shutil.copyfileobj(source, output)
        if staged.stat().st_size != weights["size"] or _sha256(staged) != weights["sha256"]:
            raise ValueError("Собранная модель не прошла проверку SHA-256")
        # Записываем веса последними: прерванное восстановление можно повторить.
        for name in manifest["files"]:
            shutil.copy2(bundle / name, model_dir / name)
        staged.replace(model_dir / "model.safetensors")
    return model_dir


def prepare_project(root=None):
    root = Path(root).resolve() if root else ROOT
    ensure_base_model(root)
    ensure_dataset(root)
    ensure_fine_tuned_model(root)
    print("Данные, токенизатор и обе модели готовы.", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare_project(args.root)
