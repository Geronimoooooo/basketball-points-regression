# Basketball Final Points — регрессия / regression

> **RU.** Предсказание итогового количества очков в баскетбольных
> матчах двухвходовой нейросетью (числовые признаки + текст).
>
> **EN.** Predicting total basketball-match points with a dual-input
> neural network (numeric features + match-flow text).

---

## Описание / Overview

**RU.** Для каждого матча известны числовые признаки (`TOTAL` —
букмекерская линия, идентификаторы команд, время игры) и текстовое
описание хода матча (`info`). Целевая переменная — `fcount`,
итоговая сумма очков. Архитектура — две независимые ветви: для
числовых признаков и для токенизированного текста; их выходы
объединяются и проходят через общий регрессионный слой с `Dropout`.
Цель домашнего задания — `MAE ≤ 17 очков`.

**EN.** Each match has both numeric features (`TOTAL` — the
bookmakers' pre-game line, team IDs, match time) and a textual
description of the match flow (`info`). The target is `fcount`, the
final point total. The architecture is two independent branches —
one for numeric features, one for the tokenised text — whose outputs
are concatenated and passed through a shared regression head with
`Dropout`. The assignment target is `MAE ≤ 17 points`.

## Датасет / Dataset

- `basketball.csv` (Yandex Cloud), скачивается автоматически в первой
  секции ноутбука.
- **Целевая переменная / Target:** `fcount` (итоговое количество
  очков).

## Стек / Stack

- Python 3.11
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `gdown`
- `scikit-learn` (`mean_absolute_error`, `StandardScaler`)
- `tensorflow` / `keras` (functional API: `Input`, `Dense`,
  `Dropout`, `concatenate`; `Tokenizer`, `pad_sequences`)

## Структура / Structure

```
basketball-points-regression/
├── README.md
└── basketball_points_regression.ipynb
```

Логические разделы / notebook sections:

1. Импорты / Imports
2. Загрузка данных / Data loading
3. Конвертация `TOTAL` + бейзлайн MAE / `TOTAL` cleanup & baseline
4. Пропуски и корреляции / Missing values & correlations
5. Удаление избыточных признаков / Drop redundant features
6. Целевая переменная и числовые признаки / Target & numeric features
7. Токенизация `info` / `info` text tokenisation
8. Двухвходовая архитектура / Two-input architecture
9. Обучение / Training
10. Графики обучения / Training curves
11. Оценка результатов / Evaluation
12. Выводы / Conclusions

## Результаты / Results

**RU.**

- Бейзлайн (предсказание тотала букмекерской линией `TOTAL`) даёт
  `MAE ≈ 13.3 очка` — сильная отправная точка.
- Двухвходовая сеть после 25 эпох на обучении уверенно укладывается
  в `MAE ≤ 17` (целевое значение задания).
- Лучший результат достигается за счёт сочетания числовых признаков и
  токенизированного текста: каждый из них по отдельности слабее
  совместной модели.

**EN.**

- Baseline (using the bookmakers' `TOTAL` line as the prediction)
  achieves `MAE ≈ 13.3 points` — a strong reference point.
- The dual-input network easily meets the assignment's
  `MAE ≤ 17 points` goal on the training set after 25 epochs.
- The best result comes from combining numeric and textual signals;
  either branch alone is weaker than the joint model.

## Как запустить / How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib seaborn gdown scikit-learn \
            tensorflow jupyter

jupyter notebook basketball_points_regression.ipynb
```

**RU.** `basketball.csv` (~0.5 МБ) скачивается автоматически в первой
секции ноутбука.

**EN.** `basketball.csv` (~0.5 MB) is downloaded automatically in the
notebook's first section.
