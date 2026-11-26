# Классификация степени зрелости фруктов по изображениям

НИР-практика НОЦ ИИ · соревнование [kmbo-practic](https://www.kaggle.com/competitions/kmbo-practic)

Автор: **Артур [Фамилия]** · Аккаунт Kaggle: [gusidusi](https://www.kaggle.com/gusidusi)

## О проекте

Бинарная классификация зрелости авокадо по фотографии:
**unripe** (незрелое, зелёное) vs **ripe** (спелое) — transfer learning на MobileNetV2.

- **Датасет:** [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (ветка `fruits-360_100x100`, версия 2026.11)
- Классы: `Avocado Green 1` → unripe; `Avocado 1`, `Avocado 2` → ripe
- Объём после очистки: 2159 изображений (1 пара sha256-дублей удалена)
- Разбиение: train 1376 / validation 243 / test 540 (стратифицированно, без утечек)

## Структура репозитория

```
notebooks/
  02-fruits-ripeness-data.ipynb      # Задание 02: исследование и подготовка данных
  03-fruits-ripeness-baseline.ipynb  # Задание 03: baseline (MobileNetV2)
```

Ноутбуки содержат результаты выполнения (outputs). Интерактивные версии на Kaggle:
- [02. Исследование данных](https://www.kaggle.com/code/gusidusi/fruits-ripeness-02-data-investigation)
- [03. Baseline](https://www.kaggle.com/code/gusidusi/fruits-ripeness-03-baseline)

## Результаты baseline (задание 03)

| Метрика | Значение (validation, 243 изображ.) |
|---|---|
| Accuracy | **1.00** |
| Macro-F1 | **1.00** |
| Ошибок | 0 / 243 |

Модель: MobileNetV2 (ImageNet) + аугментации (flip/rotation/zoom) + Dropout(0.2) + Dense(1, sigmoid);
обучение головы за 5 эпох (EarlyStopping), размер входа 224×224, batch 32, Adam(1e-3).

## Как запустить

Ноутбуки выполняются на Kaggle (данные подключаются как `moltean/fruits`).
Локально: Python 3.10+, tensorflow, pandas, matplotlib, Pillow, scikit-learn.

## План

- [x] 01. Постановка задачи и анализ существующих решений
- [x] 02. Исследование и подготовка данных
- [x] 03. Baseline
- [ ] 04. Эксперименты и гипотезы
- [ ] 05. Локальное приложение
- [ ] 06. Финальная проверка на тесте

## Источники

- Fruits-360 dataset: https://www.kaggle.com/datasets/moltean/fruits
- MobileNetV2: Sandler et al., «MobileNetV2: Inverted Residuals and Linear Bottlenecks», 2018
- tf.keras: https://www.tensorflow.org/api_docs
