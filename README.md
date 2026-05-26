# PowerTrace-IoT — Code and Infrastructure

This repository contains the source code, infrastructure configuration, and evaluation scripts for the paper:

> **"PowerTrace-IoT: An Empirical Power-Consumption Dataset for Physical-Layer Attack Detection in Distributed IoT Sensor Networks"**

**Note:** The full datasets (raw traces, SQL dumps, and cleaned CSVs) are too large for GitHub and are hosted on Zenodo at [doi.org/10.5281/zenodo.20347924](https://doi.org/10.5281/zenodo.20347924). This repository contains the reproducible environment and scripts needed to regenerate all paper figures from those datasets.

---

## Repository Structure

```
├── firmware/
│   └── pylon_main.ino          # ESP32 firmware (FreeRTOS): 500 Hz ADC, Welford DSP, 7 attack workloads
├── infrastructure/
│   ├── docker-compose.yml      # Full data-collection stack (Mosquitto + TimescaleDB + Node-RED)
│   ├── database/
│   │   ├── init.sql            # TimescaleDB hypertable schema
│   │   └── clean_dataset.py    # Post-collection data cleaning utility
│   ├── mosquitto/              # MQTT broker configuration
│   └── nodered/                # Node-RED flows and UI for coordinated attack injection
├── evaluation/
│   ├── eval_indist_ood.py      # ML evaluation pipeline (binary detection + attribution, 5-fold CV + OOD)
│   ├── generate_paper_figures.py # Reproduces all paper figures from raw CSVs and cached results
│   └── requirements.txt        # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Flashing the Firmware

The same `pylon_main.ino` is flashed to all five ESP32 nodes; only `CURRENT_NODE_ID` needs to change per board.

1. Open `firmware/pylon_main.ino` in the Arduino IDE (or PlatformIO).
2. Set `CURRENT_NODE_ID` to the target node number (1–5).
3. Fill in your `SSID`, `PASSWORD`, and `MQTT_SERVER` address.
4. Required libraries (install via Arduino Library Manager):
   - `ArduinoJson` ≥ 7
   - `PubSubClient`
   - `DHT sensor library`
   - `MPU6500_WE`

---

## Running the Infrastructure

1. Copy the environment template and set your credentials:
   ```bash
   cp .env.example .env
   # Edit .env with secure passwords before starting
   ```
2. Start the collection stack:
   ```bash
   cd infrastructure
   docker compose up -d
   ```
3. Access the services:
   - **Node-RED** (attack control & logging): http://localhost:1880
   - **pgAdmin** (database inspection): http://localhost:8081

---

## Reproducing the Evaluation

### 1 — Set up the Python environment

```bash
cd evaluation
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Download and place the datasets

Download the datasets from [Zenodo](https://doi.org/10.5281/zenodo.20347924) and organise them as follows:

```
evaluation/data/
├── train_12h/
│   └── power_readings_train.csv
└── test_48h/
    └── power_readings_trimmed.csv
```

### 3 — Run the ML evaluation pipeline

```bash
python eval_indist_ood.py
```

This trains the XGBoost binary-detection and physical-attribution models, prints the classification reports to stdout, and writes NumPy cache files to `./cache/` for the figure generator.

### 4 — Generate all paper figures

```bash
python generate_paper_figures.py
```

Output PDFs are written to `./output_figures/`. The `cache/` and `output_figures/` directories are created automatically and are excluded from version control.

---

## Citation

If you use this dataset or code in your work, please cite:

```bibtex
@inproceedings{bourreau2026powertrace,
  title     = {{PowerTrace-IoT}: An Empirical Power-Consumption Dataset for
               Physical-Layer Attack Detection in Distributed {IoT} Sensor Networks},
  author    = {Bourreau, Hugo and others},
  booktitle = {Proceedings of the 22nd International Conference on Information and
               Communications Security (ICICS 2026)},
  year      = {2026},
}
```

---

## License

The code and dataset are released under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
