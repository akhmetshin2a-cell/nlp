from functools import lru_cache

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from project_setup import ensure_fine_tuned_model


@lru_cache(maxsize=1)
def load_model():
    """Загрузить модель только при первом предсказании, восстановив bundle."""
    model_dir = ensure_fine_tuned_model()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return tokenizer, model


def predict_sentiment(text):
    """Предсказать один из двух классов и вывести вероятности."""
    if not text or not text.strip():
        return "Введите текст отзыва на английском языке."

    tokenizer, model_ft = load_model()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = model_ft(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    label_map = {0: "Negative", 1: "Positive"}

    result = f"Prediction: {label_map[pred]}\n\nProbabilities:\n"
    for i, prob in enumerate(probs):
        result += f"{label_map[i]}: {prob.item() * 100:.2f}%\n"
    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3, label="Отзыв",
        placeholder="This movie was absolutely amazing!",
    ),
    outputs=gr.Textbox(label="Результат"),
    title="Анализ тональности отзывов",
    description="Введите отзыв о фильме на английском языке: Positive или Negative.",
    analytics_enabled=False,
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
