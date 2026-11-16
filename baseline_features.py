"""Получение тех же CLS-признаков, что использованы для baseline Дня 4."""

from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from project_setup import ensure_base_model
from embedding_utils import get_embeddings


class CLSVectorizer:
    """Адаптер transform(texts) для замороженного DistilBERT, без TF-IDF."""

    def __init__(self, model_path="artifacts/day02/model", tokenizer_path="artifacts/day01/tokenizer"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def transform(self, texts):
        root = Path(__file__).resolve().parent
        ensure_base_model(root)
        model_path = root / self.model_path
        tokenizer_path = root / self.tokenizer_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_path, local_files_only=True, output_attentions=False
        )
        model.eval()
        return get_embeddings(list(texts), tokenizer, model, batch_size=32, max_length=128)
