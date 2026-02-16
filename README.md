# NEXAF-AI: AF Trajectory Analysis and Phenotyping

Analysis pipeline for atrial fibrillation (AF) phenotyping using implantable loop recorder (ILR) data from the NEXAF study. This project implements trajectory-based clustering and outcome association analysis to identify clinically meaningful AF phenotypes.

## Features

- **Data preprocessing**: Load and clean SPSS data files, align episodes to implant date, handle missing data
- **Feature extraction**: Burden features, episode patterns, temporal patterns, RR interval features, trajectory features
- **Clustering**: K-means, hierarchical clustering, HDBSCAN
- **Outcome analysis**: Association with hospitalization, quality of life (AFEQT scores), survival analysis
- **Visualization**: PCA/UMAP embeddings, phenotype profiles, outcome comparisons

## Project Structure

```
nexaf-ai/
├── src/                    # Main package
│   ├── preprocessing/      # Data loading and cleaning
│   ├── features/           # Feature extraction
│   ├── clustering/         # Clustering algorithms
│   ├── trajectory/         # Time series and state modeling
│   ├── association/        # Outcome prediction and analysis
│   └── visualization/      # Plotting utilities
├── notebooks/              # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_phenotyping.ipynb
│   └── 04_associations.ipynb
├── database/               # Data files (not tracked in git)
├── output/                 # Generated outputs (not tracked in git)
├── pyproject.toml          # Project configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Option 1: Using pip (standard)

```bash
# Clone the repository
git clone https://github.com/NTNU/nexaf-ai.git
cd nexaf-ai

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (optional)
pip install -e .
```

### Option 2: Using uv (faster)

```bash
# Clone the repository
git clone https://github.com/NTNU/nexaf-ai.git
cd nexaf-ai

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Or install as editable package with all dependencies
uv pip install -e ".[all]"
```

### Optional dependencies

```bash
# For notebook support
pip install -e ".[notebooks]"

# For development tools (pytest, black, ruff)
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[all]"
```

## Data Setup

Place the SPSS data files in the `database/` directory:

```
database/
├── df_ep5_fil_sep25_EFL.sav              # Episode data
├── Hovedfil_analyser_hovedartikkel021225_oppdatertAFburden.sav  # Outcomes
├── NEXAF_baselinefil_171125_EFL.sav      # Baseline characteristics
└── valideringsdatasett300126.sav         # Validation dataset
```

## Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

Navigate to the `notebooks/` directory and run the notebooks in order:
1. `01_data_exploration.ipynb` - Explore data, extract features
2. `02_feature_engineering.ipynb` - Analyze feature importance, select curated feature set (~14 features)
3. `03_phenotyping.ipynb` - Run clustering analysis, identify phenotypes
4. `04_associations.ipynb` - Analyze baseline characteristics and clinical outcomes by phenotype

## License

License TBD.