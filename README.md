# CO2-ECBM Adsorption Modeling and Machine Learning Codes

This repository contains Python scripts used for adsorption isotherm modeling, exploratory data analysis, machine learning model training, SHAP-based model interpretation, and symbolic regression for CO2-ECBM adsorption prediction.

The dataset is not included in this public repository. Users must place the required input Excel file locally before running the scripts.

## Repository files

```text
CO2_ECBM/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── scripts/
    ├── data_analytics.py
    ├── adsorption_isotherm_modeling.py
    ├── train_ann.py
    ├── train_random_forest.py
    ├── train_xgboost.py
    ├── train_catboost.py
    ├── train_lightgbm.py
    └── symbolic_regression.py
```

## Script descriptions

| Script | Purpose |
|---|---|
| `scripts/data_analytics.py` | Performs exploratory data analysis and saves descriptive statistics, train/test distribution plots, correlation heatmap, and pairwise relationship plots. |
| `scripts/adsorption_isotherm_modeling.py` | Fits adsorption isotherm models and saves model parameters, error metrics, and adsorption plots for pure-gas and gas-mixture cases. |
| `scripts/train_ann.py` | Trains an artificial neural network using a stratified train/test split and randomized hyperparameter search, then saves optimized hyperparameters, error metrics, cross-validation results, actual-vs-predicted values, and the trained model. |
| `scripts/train_random_forest.py` | Trains a Random Forest model using a stratified train/test split and randomized hyperparameter search, then saves optimized hyperparameters, error metrics, cross-validation results, actual-vs-predicted values, trained models, and SHAP outputs when available. |
| `scripts/train_xgboost.py` | Trains an XGBoost model using a stratified train/test split, randomized hyperparameter search, categorical gas handling, and a pressure monotonic constraint, then saves optimized hyperparameters, error metrics, cross-validation results, actual-vs-predicted values, trained models, and SHAP outputs when available. |
| `scripts/train_catboost.py` | Trains a CatBoost model using a stratified train/test split, randomized hyperparameter search, native categorical handling, and a pressure monotonic constraint, then saves optimized hyperparameters, error metrics, cross-validation results, actual-vs-predicted values, trained models, and SHAP outputs when available. |
| `scripts/train_lightgbm.py` | Trains a LightGBM model using manual randomized cross-validation, categorical gas handling, positive-support objectives, and a pressure monotonic constraint, then saves optimized hyperparameters, error metrics, cross-validation results, actual-vs-predicted values, feature importance, trained model, predictions, and residual diagnostics. |
| `scripts/symbolic_regression.py` | Performs symbolic regression using PySR with stratified train/test splitting and inner cross-validation for equation selection, then saves the selected equation, validation results, error metrics, prediction tables, and model bundle. |

## Data availability

The dataset is not uploaded to this repository.

Most scripts expect the following local Excel file:

```text
Dataset.xlsx
```

with the worksheet:

```text
Data
```

Place `Dataset.xlsx` in the repository root directory, not inside the `scripts/` folder:

```text
CO2_ECBM/
├── Dataset.xlsx
├── README.md
├── requirements.txt
└── scripts/
```

The expected target column is:

```text
Gas Adsorption, scf/ton
```

Depending on the script, the input columns may include:

```text
P, psi
T, °C
Ash content, wt.%
Fuel_Ratio
Inertinite, vol. %
Composition CO2%
Gas Type
Coal
```

Some scripts use only a subset of these columns. The exact feature list is defined in the configuration section of each script.

## Installation

Create a Python environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

## How to run

Place `Dataset.xlsx` in the repository root directory. Then run the scripts from the repository root directory using the `scripts/` path.

Examples:

```bash
python scripts/data_analytics.py
python scripts/adsorption_isotherm_modeling.py
python scripts/train_xgboost.py
python scripts/train_catboost.py
python scripts/train_lightgbm.py
python scripts/train_random_forest.py
python scripts/train_ann.py
python scripts/symbolic_regression.py
```

The training scripts may take a long time because they include randomized hyperparameter search, cross-validation, SHAP analysis, or symbolic regression.

## Generated outputs

The scripts may generate Excel result files, model files, plots, and output folders such as:

```text
*.xlsx
*.pkl
*.joblib
*.png
ML_Results/
ANN_layers_search_allinone/
pure_plots/
mixture_plots/
catboost_info/
```

These outputs are excluded from the repository using `.gitignore`.

## Reproducibility notes

- Random seeds are fixed where applicable.
- Train/test splitting is performed inside the scripts.
- The ANN, Random Forest, XGBoost, and CatBoost scripts use `RandomizedSearchCV`.
- The LightGBM script uses manual randomized cross-validation.
- The symbolic regression script uses fixed PySR settings with cross-validation for equation selection.
- The main target variable is `Gas Adsorption, scf/ton`.
- Gas type is treated as categorical where supported by the model.
- Several boosting models apply a non-decreasing monotonic constraint with respect to pressure, consistent with adsorption behavior.

## Citation

If using these scripts, please cite the associated journal article once published.

## License

This repository includes an open-source license for code sharing. Please modify the license if required by the journal, university, or research group.
