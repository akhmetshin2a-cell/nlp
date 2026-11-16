# Анализ тональности с DistilBERT

Учебный проект на семь дней: от токенизации до дообучения, сравнения моделей, анализа ошибок и Gradio-демо. Данные — английский SST-2. Два класса: **0 — Negative**, **1 — Positive**.

## Структура

- `00_transformers_overall.ipynb` — подготовка данных.
- `01_transformers_day01.ipynb` — все задания Дней 1–7 с короткими пояснениями.
- `model_bundle/` — поставляемые части собственной дообученной модели и SHA-256.
- `fine_tuned_model/` — автоматически восстановленная модель и токенизатор.
- `project_setup.py`, `project_sources.json` — воспроизводимая загрузка исходников.
- `export_github.py` — сборка полной папки для GitHub.
- `app.py` — Gradio-приложение.
- `baseline_model.pkl`, `vectorizer.pkl`, `baseline_features.py` — Logistic Regression и получение CLS-признаков базового DistilBERT.
- `data/processed/` — CSV с колонками `text,label`.
- `baseline_results.txt`, `fine_tuned_results.txt`, `comparison_results.txt` — метрики.
- `confusion_matrix_finetuned.png`, `confusion_matrix_baseline.png` — матрицы ошибок.
- `error_analysis.txt` — FP, FN, примеры и наблюдения.

## Запуск из чистого репозитория

Используйте Python 3.12. На Apple Silicon нужен Python arm64.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python project_setup.py
python app.py
```

`project_setup.py` автоматически:

1. Скачивает базовый `distilbert/distilbert-base-uncased` по ревизии `12040accade4e8a0f71eabdb258fecc2e7e948be`.
2. Скачивает SST-2 по ревизии `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`, выполняет прежнюю очистку и воспроизводит CSV. SHA-256 исходников и готовых CSV сверяются с `project_sources.json`.
3. Собирает **нашу модель Дня 5** из `model_bundle` в `fine_tuned_model/`, проверяя SHA-256 каждой части и всего файла весов. Другой checkpoint вместо неё не используется.

Для полного ноутбука первая подготовка требует интернет и скачивает около 271 МБ. Повторно готовые файлы не скачиваются. В начале `01_transformers_day01.ipynb` есть такая же ячейка подготовки; отдельный запуск `00_transformers_overall.ipynb` не обязателен.

Демо откройте по адресу [http://127.0.0.1:7860](http://127.0.0.1:7860). Введите английский отзыв. Модель выдаёт Negative/Positive и вероятности обоих классов. Остановка — `Ctrl+C`.

**Только демо:** после установки зависимостей достаточно `python app.py`. Данные и базовая модель ему не нужны. При первом предсказании fine-tuned веса автоматически восстанавливаются из `model_bundle` без интернета. Импорт `app.py` сам по себе модель не загружает.

## Что загрузить на GitHub

```bash
python export_github.py
```

Загрузите **всё содержимое `github_ready/`**, включая `model_bundle/`, `project_setup.py`, `project_sources.json`, вспомогательные `.py` и `tests/`. Готовая папка содержит исходники, ноутбуки, результаты, baseline и собственную fine-tuned модель. Ни один файл поставки не превышает 5 MiB; Git LFS для такого формата не требуется.

`model_bundle/` содержит 52 бинарные части весов, конфигурацию, токенизатор и `manifest.json`. Части нельзя пропускать, переименовывать или копировать их содержимое как текст. Это полноценные веса модели, а не ссылки на файлы на компьютере автора.

Каталоги `.venv/`, `.cache/`, `artifacts/`, `data/raw/`, `data/processed/`, `fine_tuned_model/` и `github_ready/` исключены через `.gitignore`: они восстанавливаются или содержат локальное окружение. **`model_bundle/` не исключён** и обязательно входит в репозиторий. После нового обучения повторите `python export_github.py`, чтобы обновить bundle.

GitHub блокирует обычные Git-файлы больше 100 MiB ([документация](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)). Здесь веса разделены на части по 5 MiB, в том числе для загрузки через веб-интерфейс.

## Запуск ноутбука в VS Code

Откройте папку клонированного проекта и выберите ядро `.venv/bin/python`. Выполняйте ячейки по порядку, начиная с подготовки. Дни 1–4 используют скачанные исходники; День 4 сохраняет baseline, День 5 дообучает модель, Дни 6–7 загружают результаты.

Переобучение Дня 5 обновляет локальную модель и метрики. Чтобы просто проверить сохранённое демо, достаточно `python app.py`.

## Проверка

```bash
python -m unittest discover -s tests -v
```

Тесты проверяют восстановление bundle, отклонение повреждённых/отсутствующих частей и импорт приложения без локальной модели. Весь проект дополнительно проверен в чистой копии без локальных `artifacts/`, CSV и `fine_tuned_model/`.

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
