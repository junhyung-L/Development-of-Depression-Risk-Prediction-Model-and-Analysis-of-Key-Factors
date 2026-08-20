# Architecture

| Path | Responsibility |
|---|---|
| `src/config.py` | Shared paths, seed, split fraction, and selected features. |
| `src/preprocessing.py` | CSV loading and encodings. |
| `src/train.py` | Logistic training, evaluation, p-value summary, and result output. |
| `src/generate_plots.py` | Correlation, countplot, and learning-curve figures. |

The model uses four selected fields and an 80/20 split. No external validation
or clinical deployment path is implemented.
