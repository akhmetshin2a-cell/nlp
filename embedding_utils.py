"""День 2: CLS-векторы и косинусное сходство с замороженным трансформером."""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from tokenization_utils import tokenize_texts


def get_embeddings(texts, tokenizer, model, batch_size=32, max_length=128):
    """Вернуть массив [число текстов, hidden_size] в исходном порядке.

    Модель переводится в eval; градиенты не вычисляются.
    Входы отправляются на устройство модели, результат возвращается на CPU.
    Пустой список даёт массив формы (0, hidden_size).
    """
    if not isinstance(texts, list):
        raise TypeError("texts должен быть списком строк")
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Каждый элемент texts должен быть строкой")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size должен быть положительным целым числом")
    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    model.eval()
    device = next(model.parameters()).device
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        tokens = tokenize_texts(batch_texts, max_length=max_length, tokenizer=tokenizer)
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def similarity(text1, text2, tokenizer, model):
    """Косинусное сходство CLS-векторов двух текстов, а не вероятность класса."""
    embeddings = get_embeddings([text1, text2], tokenizer, model)
    score = cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0]
    return float(score)
