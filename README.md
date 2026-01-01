# Spoken Digit Recognition (Afaan Oromoo)

## Project Overview
This project aims to build a machine learning model to recognize spoken digits (0-9) in Afaan Oromoo.

## Project Structure
The project follows the Cookiecutter Data Science structure:

```
├── data/
│   ├── external/       # Data from third party sources.
│   ├── interim/        # Intermediate data that has been transformed.
│   ├── processed/      # The final, canonical data sets for modeling.
│   └── raw/            # The original, immutable data dump.
├── docs/               # Documentation
├── models/             # Trained and serialized models
├── notebooks/          # Jupyter notebooks
├── references/         # Data dictionaries, manuals, etc.
├── reports/            # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/        # Generated graphics and figures
├── src/                # Source code for use in this project
│   ├── __init__.py
│   ├── data/           # Scripts to download or generate data
│   ├── models/         # Scripts to train models
│   └── visualization/  # Scripts to create visualizations
└── README.md
```

## Getting Started
1.  **Data**: Raw data is located in `data/raw/` (zips). Processed audio files are in `data/processed/`.
2.  **Environment**: Install dependencies (e.g., `requirements.txt`).
3.  **Exploration**: Check `notebooks/` for initial data analysis.
