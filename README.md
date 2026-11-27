# Классификация степени зрелости фруктов по изображениям

Учебный проект по классификации изображений · соревнование [kmbo-practic](https://www.kaggle.com/competitions/kmbo-practic)

Автор: **Бахтибеков Артур, Макаров Ярослав  КМБО-24-24** · 

## О проекте

Классификация стадии зрелости авокадо по фотографии — одна из трёх стадий:
**unripe** (незрелое, зелёное), **ripe** (спелое, тёмно-зелёное), **overripe** (перезревшее, почти чёрное).
Подход — transfer learning на MobileNetV2 с 3-классовой головой (softmax).

- **Датасет:** [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (ветка `fruits-360_100x100`, версия 2026.11)
- Классы: `Avocado Green 1` → unripe; `Avocado 1`, `Avocado 2` → ripe; `Avocado Black 1`, `Avocado Black 2` → overripe
- Объём после очистки: 4076 изображений (1 пара sha256-дублей удалена)
- Разбиение: train 2600 / validation 459 / test 1017 (стратифицированно, без утечек)
- Классы несбалансированы (23% / 30% / 47%), поэтому основная метрика — **macro-F1**

## Структура репозитория

```
notebooks/
  02-fruits-ripeness-data.ipynb      # 02: исследование и подготовка данных
  03-fruits-ripeness-baseline.ipynb  # 03: baseline (MobileNetV2, 3 класса)
  04-fruits-ripeness-experiments.ipynb # 04: эксперименты H1–H6 и выбор варианта
```

Ноутбуки содержат результаты выполнения (outputs). Интерактивные версии на Kaggle:
- [02. Исследование данных](https://www.kaggle.com/code/gusidusi/fruits-ripeness-02-data-investigation)
- [03. Baseline](https://www.kaggle.com/code/gusidusi/fruits-ripeness-03-baseline)
- [04. Эксперименты](https://www.kaggle.com/code/gusidusi/fruits-ripeness-04-experiments)

## Результаты baseline (ноутбук 03)

| Метрика | Значение (validation, 459 изображ.) |
|---|---|
| Accuracy | **0.996** (457/459) |
| Macro-F1 | **1.00** (0.996) |
| Ошибок | 2 / 459 (обе — unripe→overripe) |

Модель: MobileNetV2 (ImageNet) + аугментации (flip/rotation/zoom) + Dropout(0.2) + Dense(3, softmax);
sparse categorical crossentropy, обучение головы 8 эпох (EarlyStopping, лучшая val_accuracy 0.9956), вход 224×224, batch 32, Adam(1e-3).

## Эксперименты (ноутбук 04)

Сравнение на единой validation (459 изображений), обучение только по train, правило выбора: максимум macro-F1 → accuracy → более ранний вариант.

| Вариант | Изменение | Accuracy | Macro-F1 | Ошибок |
|---|---|---|---|---|
| baseline (03) | — | 0.998 | 0.998 | 1 |
| H1 | fine-tune верхних 30 слоёв (lr 1e-5) | **1.000** | **1.000** | **0** |
| H2 | RandomContrast(0.2) + 15 эпох | 0.996 | 0.995 | 2 |
| H3 | balanced class weights | 1.000 | 1.000 | 0 |
| H4 | H1+H2 | 1.000 | 1.000 | 0 |
| H5 | H1+H3 | 1.000 | 1.000 | 0 |
| H6 | H1+H2+H3 | 1.000 | 1.000 | 0 |

**Выбран H1** (fine-tune верхних 30 слоёв MobileNetV2, lr 1e-5, 5 эпох): исправил единственную ошибку baseline (unripe→overripe, тёмно-зелёное авокадо) без новых ошибок. Итоговая модель `final_model.keras` + `predict.py` + `experiments.csv` — в outputs ноутбука 04 на Kaggle; предсказания воспроизводятся при повторной загрузке.

## Deliverables

Все итоговые файлы проекта лежат в папке [`deliverables/`](deliverables/):

| Файл | Что это |
|---|---|
| `deliverables/01.docx` | Постановка задачи, аналоги, метрики, функции системы |
| `deliverables/02.ipynb` | Исследование данных (с результатами выполнения) |
| `deliverables/03.ipynb` | Baseline MobileNetV2 (с результатами выполнения) |
| `deliverables/04.ipynb` | Эксперименты H1–H6 и выбор итогового варианта (с результатами) |
| `deliverables/writeup_draft.md` | Отчёт по проекту: данные, модель, эксперименты, результаты |

## Как запустить

Ноутбуки выполняются на Kaggle (данные подключаются как `moltean/fruits`).
Локально: Python 3.10+, tensorflow, pandas, matplotlib, Pillow, scikit-learn.

## План

- [x] 01. Постановка задачи и анализ существующих решений
- [x] 02. Исследование и подготовка данных
- [x] 03. Baseline
- [x] 04. Эксперименты и гипотезы
- [ ] 05. Локальное приложение
- [ ] 06. Финальная проверка на тесте

## Источники

- Fruits-360 dataset: https://www.kaggle.com/datasets/moltean/fruits
- MobileNetV2: Sandler et al., «MobileNetV2: Inverted Residuals and Linear Bottlenecks», 2018
- tf.keras: https://www.tensorflow.org/api_docs
