
import warnings
warnings.filterwarnings("ignore")

import re, os, random, json
import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import joblib
import matplotlib.pyplot as plt

# ---------------- Config ----------------
EXCEL_IN   = "Dataset.xlsx"
SHEET_NAME = "Data"

MODEL_OUT    = "LGBM.pkl"
PRED_XLSX    = "Predictions.xlsx"   
RESULTS_XLSX = "LGBM.xlsx"            

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
N_ITER       = 120        

rng = random.Random(RANDOM_STATE)

# -------------- Helpers -----------------
def mape_safe(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def rmse_safe(y, yhat): 
    return float(np.sqrt(mean_squared_error(y, yhat)))

def print_metrics(name, y, yhat, units="scf/ton"):
    r2   = r2_score(y, yhat)
    rmse = rmse_safe(y, yhat)
    mae  = mean_absolute_error(y, yhat)
    mape = mape_safe(y, yhat)
    print(f"{name}: R²={r2:.5f} | RMSE={rmse:.5f} {units} | MAE={mae:.5f} {units} | MAPE={mape:.2f}%")
    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape)

# ---------- Feature-name sanitizer (LightGBM JSON-safe) ----------
def _safe(name: str) -> str:
    s = re.sub(r'[^0-9A-Za-z_]', '_', str(name))
    s = re.sub(r'__+', '_', s).strip('_')
    return s or "feature"

# ---------------- Load data ----------------
Total_Data = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
Total_Data.columns = [c.strip() for c in Total_Data.columns]


FEATURES = [
    'P, psi', 'T, °C', 'Ash content, wt.%','Fuel_Ratio', 'Composition CO2%', 'Gas Type'
]
TARGET = 'Gas Adsorption, scf/ton'
                                  
ORIG_FEATURES = FEATURES[:]                                                   
safe_map, used = {}, set()
for c in ORIG_FEATURES:
    base = _safe(c); new = base; k = 1
    while new in used:
        k += 1
        new = f"{base}_{k}"
    safe_map[c] = new; used.add(new)
SAFE_FEATURES = [safe_map[c] for c in ORIG_FEATURES]
inverse_map = {v: k for k, v in safe_map.items()}

                                 
X = Total_Data[ORIG_FEATURES].copy().rename(columns=safe_map)
y = Total_Data[TARGET].astype(float).copy()

                                                                        
gas_safe = safe_map['Gas Type']
if gas_safe in X.columns:
    try:
                                                  
        if pd.api.types.is_numeric_dtype(X[gas_safe]):
            X[gas_safe] = X[gas_safe].astype('category')
        else:
            X[gas_safe] = X[gas_safe].astype('category')
    except Exception:
        pass

has_zero = bool((y <= 0.0).any())                                                  

# ---------------- Monotone constraint (non-decreasing in Pressure) ----------------
                                                               
pressure_safe = safe_map['P, psi']
monotone_constraints = [1 if f == pressure_safe else 0 for f in SAFE_FEATURES]

# --------------- Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ------------- Search space (positive-support objectives only) -----------
objective_candidates = ['tweedie'] + ([] if has_zero else ['gamma'])

space_common = {
    "num_leaves"        : [16, 24, 31, 40, 48, 64, 80, 96, 112, 128],
    "max_depth"         : [-1, 4, 5, 6, 7, 8, 10],
    "learning_rate"     : [0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "min_child_samples" : [10, 20, 30, 50, 75, 100, 150, 200],
    "subsample"         : [0.6, 0.7, 0.8, 0.9, 1.0],                        
    "subsample_freq"    : [1, 3, 5, 10],                                             
    "colsample_bytree"  : [0.6, 0.7, 0.8, 0.9, 1.0],                        
    "reg_alpha"         : [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0],
    "reg_lambda"        : [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0],
}

tweedie_p_grid = [1.1, 1.2, 1.35, 1.5, 1.7, 1.9]         

def sample_params():
    params = {k: rng.choice(v) for k, v in space_common.items()}
    obj = rng.choice(objective_candidates)
    params["objective"] = obj
    if obj == "tweedie":
        params["tweedie_variance_power"] = rng.choice(tweedie_p_grid)
    return params

# ------------- Manual randomized CV (NO early stop) -------------
def cv_score(params, X_, y_, folds=CV_FOLDS):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in kf.split(X_):
        X_tr, X_va = X_.iloc[tr_idx], X_.iloc[va_idx]
        y_tr, y_va = y_.iloc[tr_idx], y_.iloc[va_idx]

        model = LGBMRegressor(
            boosting_type="gbdt",
            n_estimators=3500,                                          
            random_state=RANDOM_STATE,
            n_jobs=-1,
            enable_categorical=True,                                                
            monotone_constraints=monotone_constraints,                                   
            **params
        )
        model.fit(X_tr, y_tr)
        yhat = model.predict(X_va)
                                                                          
        if (yhat < 0).any():
            yhat = np.maximum(yhat, 0.0)
        scores.append(r2_score(y_va, yhat))
    return float(np.mean(scores))

best_params, best_score, cv_log = None, -1e9, []

for i in range(1, N_ITER+1):
    params = sample_params()
    score = cv_score(params, X_train, y_train, folds=CV_FOLDS)
    cv_log.append({**{"iter": i, "mean_test_score": score}, **params})
    if score > best_score:
        best_score = score
        best_params = params
    if i % 10 == 0 or i == 1:
        print(f"[{i}/{N_ITER}] current best CV R² = {best_score:.5f}  | params: {best_params}")

print("\nBest CV params:", best_params)
print("Best CV R²:", f"{best_score:.5f}")

# ------------- Final fit using best params (NO early stop) -------------
best_model = LGBMRegressor(
    boosting_type="gbdt",
    n_estimators=3500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    enable_categorical=True,
    monotone_constraints=monotone_constraints,
    **best_params
)
best_model.fit(X_train, y_train)

# --------------- Evaluation -------------
def predict_nonneg(model, X_):
    yhat = model.predict(X_)
    return np.maximum(yhat, 0.0)

yhat_tr = predict_nonneg(best_model, X_train)
yhat_te = predict_nonneg(best_model, X_test)

print("\nHoldout results (full n_estimators used):")
m_tr = print_metrics("Train", y_train.values, yhat_tr, units="scf/ton")
m_te = print_metrics("Test ", y_test.values,  yhat_te, units="scf/ton")
if best_params["objective"] == "tweedie":
    print(f"Objective: tweedie, tweedie_variance_power={best_params.get('tweedie_variance_power')}")
else:
    print("Objective: gamma")
print(f"Min pred train: {float(np.min(yhat_tr)):.6g}")
print(f"Min pred test : {float(np.min(yhat_te)):.6g}")

# --------------- Save model/predictions -------
joblib.dump(best_model, MODEL_OUT)
print(f"\nSaved model -> {MODEL_OUT}")

all_pred = predict_nonneg(best_model, X)
df_out = Total_Data.copy()
df_out["Predicted Adsorption (LGBM) [scf/ton]"] = all_pred
df_out["Residual (Actual - Pred) [scf/ton]"] = df_out[TARGET] - df_out["Predicted Adsorption (LGBM) [scf/ton]"]
df_out.to_excel(PRED_XLSX, index=False)
print(f"Saved predictions -> {PRED_XLSX}")

# --------------- Build Excel tables ---------------
n_train, n_test = len(y_train), len(y_test)
num_params = len(SAFE_FEATURES) + 1
mse_train = mean_squared_error(y_train, yhat_tr)
mse_test  = mean_squared_error(y_test,  yhat_te)
rmse_train = float(np.sqrt(mse_train))
rmse_test  = float(np.sqrt(mse_test))
r2_train = r2_score(y_train, yhat_tr)
r2_test  = r2_score(y_test,  yhat_te)
mape_train = mape_safe(y_train, yhat_tr)
mape_test  = mape_safe(y_test,  yhat_te)
eps = 1e-12
aic_train = n_train*log(max(mse_train, eps)) + 2*num_params
aic_test  = n_test *log(max(mse_test,  eps)) + 2*num_params
bic_train = n_train*log(max(mse_train, eps)) + num_params*log(max(n_train, 2))
bic_test  = n_test *log(max(mse_test,  eps)) + num_params*log(max(n_test,  2))

param_ranges = {
    "objective"             : str(objective_candidates),
    "tweedie_variance_power": str(tweedie_p_grid) + " (Tweedie only)",
    "num_leaves"            : "range(16,256)",
    "max_depth"             : "[-1] + range(3,13)",
    "learning_rate"         : "0.005..0.20 (grid)",
    "min_child_samples"     : "range(5,200)",
    "subsample"             : "0.6..1.0",
    "colsample_bytree"      : "0.6..1.0",
    "reg_alpha"             : "1e-4..1e1 (logspace)",
    "reg_lambda"            : "1e-4..1e1 (logspace)",
}
param_display = []
for name, desc in param_ranges.items():
    param_display.append((name, desc, best_params.get(name, "")))
df_hyper_table = pd.DataFrame(param_display, columns=["Hyperparameter", "Values / Range", "Optimized value"])
df_hyper_table.insert(0, "ML Tool", ["LightGBM (Positive-support, No ES, Safe Names, Monotone P)"] + [""]*(len(df_hyper_table)-1))

rows = []
for data_split, vals in [("Training data", (rmse_train, r2_train, aic_train, bic_train)),
                         ("Testing data",  (rmse_test,  r2_test,  aic_test,  bic_test))]:
    rows.append([data_split, "RMSE, scf/ton", vals[0]])
    rows.append(["",         "R², fraction", vals[1]])
    rows.append(["",         "AIC",          vals[2]])
    rows.append(["",         "BIC",          vals[3]])
df_pretty_metrics = pd.DataFrame(rows, columns=["", "", "LightGBM"])

                                         
df_cv = pd.DataFrame(cv_log)
front_cols = ["iter", "mean_test_score"]
other_cols = [c for c in df_cv.columns if c not in front_cols]
df_cv = df_cv[front_cols + other_cols].sort_values("iter")

                     
df_train_pred = pd.DataFrame({"Actual": y_train.values, "Predicted": yhat_tr}, index=X_train.index)
df_test_pred  = pd.DataFrame({"Actual": y_test.values,  "Predicted": yhat_te}, index=X_test.index)

                                                         
fi = getattr(best_model, "feature_importances_", None)
if fi is not None:
    fi_df = pd.DataFrame({
        "Feature (original)": [inverse_map.get(f, f) for f in SAFE_FEATURES],
        "Feature (safe)"    : SAFE_FEATURES,
        "Importance"        : fi
    }).sort_values("Importance", ascending=False)
else:
    fi_df = pd.DataFrame(columns=["Feature (original)", "Feature (safe)", "Importance"])

# -------------------- SAVE RESULTS WORKBOOK --------------------
with pd.ExcelWriter(RESULTS_XLSX) as writer:
    df_hyper_table.to_excel(writer, sheet_name="Hyperparameter_Table", index=False)
    df_pretty_metrics.to_excel(writer, sheet_name="Error_Metrics", index=False)
    df_cv.to_excel(writer, sheet_name="CV_Results", index=False)
    df_train_pred.to_excel(writer, sheet_name="Train_Actual_vs_Pred", index=True)
    df_test_pred.to_excel(writer,  sheet_name="Test_Actual_vs_Pred",  index=True)
    fi_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

print(f"Excel: {RESULTS_XLSX}")


