"""Функции Дня 1: подготовка входов для трансформера, без загрузки весов."""

import pandas as pd
from transformers import PreTrainedTokenizerBase


def tokenize_texts(
    texts: list[str],
    max_length: int = 128,
    *,
    tokenizer: PreTrainedTokenizerBase,
):
    """Вернуть CPU-тензоры input_ids и attention_mask для непустого списка текстов.

    padding=True дополняет до самой длинной строки текущего батча.
    max_length ограничивает всю последовательность, включая спецтокены.
    Токенизатор передаётся явно, чтобы использовать те же правила в День 2.
    """
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts должен быть непустым списком строк")
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Каждый элемент texts должен быть строкой")
    minimum = tokenizer.num_special_tokens_to_add(pair=False)
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length должен быть целым числом")
    if not minimum <= max_length <= tokenizer.model_max_length:
        raise ValueError(
            f"max_length должен быть от {minimum} до {tokenizer.model_max_length}"
        )
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text: str, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    """Показать subword-токены и вернуть таблицу полного входа со спецтокенами."""
    pieces = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(pieces)
    print(f"Исходный текст: {text}")
    print(f"Токены без специальных: {pieces}")
    print(f"IDs без специальных: {ids}")
    print(f"Количество без специальных: {len(pieces)}")

    encoded = tokenizer(text, return_special_tokens_mask=True)
    print(f"Количество со специальными: {len(encoded['input_ids'])}")
    return pd.DataFrame({
        "position": range(len(encoded["input_ids"])),
        "token": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
        "id": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "special": encoded["special_tokens_mask"],
    })
