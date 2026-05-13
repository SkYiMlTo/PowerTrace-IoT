# PowerTrace-IoT Dataset: Code and Infrastructure

This repository contains the source code, infrastructure configuration, and evaluation scripts for the paper **"PowerTrace-IoT: An Empirical Power-Consumption Dataset for Physical-Layer Attack Detection in Distributed IoT Sensor Networks"**.

**Note:** The full datasets (raw traces, SQL dumps, and cleaned CSVs) are too large for GitHub and are hosted separately on [Zenodo / HuggingFace] (Link to be added). This repository contains the reproducible environment and scripts.

## Repository Structure

```
├── firmware/
│   └── pylon_main.ino       # Main ESP32 firmware (FreeRTOS) with attack workloads
├── infrastructure/
│   ├── docker-compose.yml   # Full data collection stack
│   ├── database/            # TimescaleDB init schema and dataset cleaning script
│   ├── mosquitto/           # MQTT broker configuration
│   └── nodered/             # Node-RED flows and UI for attack injection
├── evaluation/
│   ├── eval_indist_ood.py   # Machine learning evaluation (Out-of-Distribution)
│   ├── ai_benchmark*.py     # Additional AI benchmarks
│   ├── generate_paper_figures.py # Script to recreate paper plots
│   └── requirements.txt     # Python dependencies
├── .env.example             # Example environment variables
└── .gitignore
```

## Running the Infrastructure

1. Copy the example environment file and update passwords if necessary:
   ```bash
   cp .env.example .env
   ```
2. Start the Docker stack:
   ```bash
   cd infrastructure
   docker compose up -d
   ```
3. Access the services:
   - Node-RED (Attack Control & Logging): http://localhost:1880
   - pgAdmin (Database Inspection): http://localhost:8081

## Reproducing the Evaluation

To run the evaluation scripts and recreate the figures:

1. Create a virtual environment and install dependencies:
   ```bash
   cd evaluation
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download the datasets from the public repository and place them in the appropriate directory.
3. Run the evaluation script:
   ```bash
   python eval_indist_ood.py
   ```

## License

The code and dataset are released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
