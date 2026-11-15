# Анализ тональности с DistilBERT

Учебный проект на семь дней: от токенизации до дообучения, сравнения моделей, анализа ошибок и Gradio-демо. Данные — английский SST-2. Два класса: **0 — Negative**, **1 — Positive**.

## Структура

- `00_transformers_overall.ipynb` — подготовка данных.
- `01_transformers_day01.ipynb` — все задания Дней 1–7 с короткими пояснениями.
- `fine_tuned_model/` — дообученная модель и токенизатор.
- `app.py` — Gradio-приложение.
- `baseline_model.pkl`, `vectorizer.pkl`, `baseline_features.py` — Logistic Regression и получение CLS-признаков базового DistilBERT.
- `data/processed/` — CSV с колонками `text,label`.
- `baseline_results.txt`, `fine_tuned_results.txt`, `comparison_results.txt` — метрики.
- `confusion_matrix_finetuned.png`, `confusion_matrix_baseline.png` — матрицы ошибок.
- `error_analysis.txt` — FP, FN, примеры и наблюдения.

## Запуск демо

На этом Mac окружение `.venv` уже подготовлено. В терминале VS Code из папки проекта:

```bash
.venv/bin/python app.py
```

Откройте [http://127.0.0.1:7860](http://127.0.0.1:7860). Введите отзыв о фильме на английском языке. Приложение выведет Negative/Positive и вероятности обоих классов. Остановка сервера — `Ctrl+C` в терминале.

Для установки на другой машине используйте Python 3.12, на Apple Silicon — arm64:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Gradio ограничен веткой 5.x для совместимости с Transformers 4.x. FastAPI как отдельная альтернатива здесь не реализован: для демо выбран Gradio. Модель загружается из локальной папки, повторное скачивание не требуется.

## Запуск ноутбука в VS Code

Откройте `01_transformers_day01.ipynb`, выберите ядро **Python (transformers-week)** или `.venv/bin/python`. Выполняйте ячейки по порядку. День 7 использует предсказания Дня 6. Повторный запуск обучения Дня 5 занимает время и сохраняет новую версию модели.

Если возникает `No module named torch`, проверьте ядро: установленный здесь Python 3.13 из Anaconda имеет архитектуру x86_64; проект работает в `.venv` с Python 3.12 arm64.

## Результаты

Обучение: 4 000 примеров. Проверка: 1 000 примеров из `data/processed/train.csv`, стратифицированное разбиение 80/20 с `random_state=42`. Оба подхода оценены на одинаковых строках. Fine-tuning — три эпохи.

| Модель | Macro F1 | Accuracy |
|---|---:|---:|
| Baseline: замороженный DistilBERT CLS + Logistic Regression | 0.8546 | 0.8570 |
| Fine-tuned DistilBERT | 0.9075 | 0.9090 |

Относительное улучшение macro F1 — **6.19%**. Эта проверочная тысяча использовалась как validation в День 5; она не является независимым финальным тестом. Отдельные `data/processed/validation.csv` и `test.csv` в сравнении Дней 4–7 не использованы.

Fine-tuned модель ошиблась в **91** из 1 000 случаев: **53 FP**, **38 FN**. Средняя длина ошибочных текстов — 44.6 символа, всей выборки — 54.2. В примерах есть обрывки фраз, названия и смешанная тональность; подробности в `error_analysis.txt`.

## Использование модели в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "./fine_tuned_model", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model", local_files_only=True)
model.eval()
inputs = tokenizer(
    "This movie was great!", return_tensors="pt", truncation=True, max_length=128
)
with torch.no_grad():
    outputs = model(**inputs)
probabilities = torch.softmax(outputs.logits, dim=1)
prediction = probabilities.argmax(dim=1).item()
print({0: "Negative", 1: "Positive"}[prediction])
```

## Источники

- [SST-2 от Stanford NLP](https://huggingface.co/datasets/stanfordnlp/sst2).
- [Базовый DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased).

Ревизии исходников и параметры подготовки сохранены в `data/processed/manifest.json` и `artifacts/day01/run_config.json`.
