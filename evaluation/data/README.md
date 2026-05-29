# Dataset Directory

Place the dataset CSV files here as obtained from Zenodo
([doi.org/10.5281/zenodo.20347924](https://doi.org/10.5281/zenodo.20347924)):

```
evaluation/data/
├── train_12h/
│   └── power_readings_train.csv   ← 12-hour training dataset
└── test_48h/
    └── power_readings_trimmed.csv ← 48-hour evaluation dataset
```

## Data Cleaning

If you are using the pre-processed CSVs from Zenodo, no action is needed.
If you are generating the dataset from the raw database dumps, you must run
`infrastructure/database/clean_dataset.py` to apply physical-constraint
imputation before running the evaluation scripts.

Two classes of anomalous rows are imputed via per-node rolling median
(window = 10 samples ≈ 10 s):

| Condition | Rationale |
|-----------|-----------|
| `absolute_current_ma > 250` or `power_mw > 1250` | Physically impossible for the ESP32 hardware; ADC glitch |
| `absolute_current_ma < 40` | A powered-on ESP32 always draws ≥ 40 mA; transient artefact |

Imputed rows typically represent < 0.05 % of the dataset.
