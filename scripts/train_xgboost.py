
from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    try:
        ip.run_line_magic('clear', '')
        ip.run_line_magic('reset', '-sf')
    except Exception:
        pass

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import joblib
from xgboost import XGBRegressor
from pandas.api.types import CategoricalDtype

def mape(actual, pred, eps=1e-9):
    actual = np.asarray(actual); pred = np.asarray(pred)
    denom = np.where(np.abs(actual) < eps, eps, np.abs(actual))
    return np.mean(np.abs((actual - pred) / denom)) * 100

# ====================================================================================================================
def make_strat_bins(y: pd.Series, max_bins: int = 8, min_count: int = 2) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    idx_valid = y.dropna().index
    for q in range(max_bins, 2-1, -1):
        try:
            bins = pd.qcut(y.loc[idx_valid], q=q, duplicates="drop")
        except Exception:
            continue
        if bins.value_counts().min() >= min_count:
            out = pd.Series(index=y.index, dtype=object)
            out.loc[idx_valid] = bins.astype(str)
            out = out.fillna("all")
            return out
    return pd.Series(["all"] * len(y), index=y.index)

# ====================================================================================================================
EXCEL_IN   = 'Dataset.xlsx'
SHEET_NAME = 'Data'

# ----------- Data -----------
df = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

FEATURES = [
    'P, psi', 'T, °C','Ash content, wt.%','Fuel_Ratio', 'Composition CO2%', 'Gas Type'
]
TARGET = 'Gas Adsorption, scf/ton'

                               
required = set(FEATURES + [TARGET])
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing expected columns: {missing}")

X = df[FEATURES].copy()
y = pd.to_numeric(df[TARGET], errors='coerce').astype(float)

# ---- Guard for Poisson ----
if (y < 0).any():
    raise ValueError("Poisson loss requires non-negative targets. Found negatives in y.")

# --- CATEGORICAL: native categorical for XGBoost (hist) ---
                                                                         
if 'Gas Type' in X.columns:
    X['Gas Type'] = X['Gas Type'].astype(CategoricalDtype(categories=[1, 2]))

# --- MONOTONE: non-decreasing in Pressure (string-form tuple, safest across versions) ---
constraint_vec = [1 if f == 'P, psi' else 0 for f in FEATURES]
monotone_str = "(" + ",".join(str(int(v)) for v in constraint_vec) + ")"

                  
y_bins = make_strat_bins(y, max_bins=8, min_count=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y_bins
)

# ====================================================================================================================
              
base_xgb = XGBRegressor(
    objective='count:poisson',
    tree_method='hist',
    enable_categorical=True,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1,
    monotone_constraints=monotone_str,                 
)

# ====================================================================================================================
                     
HP = {
    'n_estimators'     : [300, 500, 700, 1000, 1500, 2000, 2500, 3500, 4000],
    'max_depth'        : [3, 4, 5, 6, 7, 8],
    'learning_rate'    : [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
    'min_child_weight' : [1, 2, 3, 4, 6, 8, 10],
    'gamma'            : [0, 0.1, 0.3, 0.5, 1, 2],
    'subsample'        : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha'        : [0, 0.1, 0.3, 0.5, 1, 2, 5, 10],
    'reg_lambda'       : [0, 1, 2, 5, 10, 15, 20],
                                                                  
}
param_distributions = HP

# ====================================================================================================================
                    
r2_scorer = make_scorer(r2_score)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_distributions,
    n_iter=500,                                   
    scoring=r2_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
    return_train_score=False,
    error_score='raise'                                                     
)

search.fit(X_train, y_train)

# ====================================================================================================================
                          
df_cv = pd.DataFrame(search.cv_results_)
K_fold_result = df_cv[[c for c in [
    'param_n_estimators','param_learning_rate','param_max_depth',
    'param_min_child_weight','param_subsample','param_colsample_bytree',
    'param_gamma','param_reg_alpha','param_reg_lambda','param_max_delta_step',
    'mean_test_score','rank_test_score'
] if c in df_cv.columns]]

Train_Pred = pd.Series(search.predict(X_train), index=X_train.index, name='Predicted').clip(lower=0)
Test_Pred  = pd.Series(search.predict(X_test),  index=X_test.index,  name='Predicted').clip(lower=0)

         
n_train, n_test = len(y_train), len(y_test)
num_params = X_train.shape[1] + 1

mape_train = mape(y_train, Train_Pred)
mape_test  = mape(y_test,  Test_Pred)

mse_train = mean_squared_error(y_train, Train_Pred)
mse_test  = mean_squared_error(y_test,  Test_Pred)

rmse_train = float(np.sqrt(mse_train))
rmse_test  = float(np.sqrt(mse_test))

r2_train = r2_score(y_train, Train_Pred)
r2_test  = r2_score(y_test,  Test_Pred)

eps = 1e-12
aic_train = n_train*log(max(mse_train, eps)) + 2*num_params
aic_test  = n_test *log(max(mse_test,  eps)) + 2*num_params
bic_train = n_train*log(max(mse_train, eps)) + num_params*log(max(n_train, 2))
bic_test  = n_test *log(max(mse_test,  eps)) + num_params*log(max(n_test,  2))

print("Best params:", search.best_params_)
print("Best CV R2:", search.best_score_)
print("Train R2:", r2_train, " Test R2:", r2_test)

# ====================================================================================================================
                  
def list_to_str(vals): return ', '.join(map(str, vals))

param_display = []
for name, vals in HP.items():
    opt_value = search.best_params_.get(name, '')
    param_display.append((name, list_to_str(vals), opt_value))

df_hyper_table = pd.DataFrame(param_display, columns=['Hyperparameter', 'Values / Range', 'Optimized value'])
df_hyper_table.insert(0, 'ML Tool', ['XGBoost (Poisson, Monotone P, Categorical Gas)'] + ['']*(len(df_hyper_table)-1))

rows = []
for label, rmse, r2, aic, bic in [
    ('Training data', rmse_train, r2_train, aic_train, bic_train),
    ('Testing data',  rmse_test,  r2_test,  aic_test,  bic_test),
]:
    rows.append([label, 'RMSE, scf/ton', rmse])                                     
    rows.append(['',     'R², fraction', r2])
    rows.append(['',     'AIC',          aic])
    rows.append(['',     'BIC',          bic])
df_pretty_metrics = pd.DataFrame(rows, columns=['', '', 'XGBoost'])

df_train_pred = pd.DataFrame({'Actual': y_train}, index=X_train.index)
df_train_pred['Predicted'] = Train_Pred
df_test_pred  = pd.DataFrame({'Actual': y_test},  index=X_test.index)
df_test_pred['Predicted']  = Test_Pred

with pd.ExcelWriter('XGB_RandomizedSearch_Results.xlsx', engine='openpyxl') as writer:
    df_hyper_table.to_excel(writer, sheet_name='Hyperparameter_Table', index=False)
    df_pretty_metrics.to_excel(writer, sheet_name='Error_Metrics', index=False)
    K_fold_result.to_excel(writer, sheet_name='CV_Results', index=False)
    df_train_pred.to_excel(writer, sheet_name='Train_Actual_vs_Pred', index=True)
    df_test_pred.to_excel(writer,  sheet_name='Test_Actual_vs_Pred',  index=True)

# ====================================================================================================================
           
joblib.dump(search.best_estimator_, 'XGB_best_cv_random.pkl')
joblib.dump(search, 'XGB_randomizedsearch_cv.pkl')

                                                       
final_xgb = XGBRegressor(
    objective='count:poisson', tree_method='hist', enable_categorical=True,
    eval_metric='rmse', monotone_constraints=monotone_str,                        
    random_state=42, n_jobs=-1, **search.best_params_
)
final_xgb.fit(X, y)
joblib.dump(final_xgb, 'XGB_final_full_random.pkl')

print("Saved PKLs: XGB_best_cv_random.pkl, XGB_randomizedsearch_cv.pkl, XGB_final_full_random.pkl")
print("Excel: XGB_RandomizedSearch_Results.xlsx")

# ======================= SHAP plots & tables =======================
try:
    import shap
except Exception as e:
    raise RuntimeError("Install SHAP: pip install shap") from e

best_model = search.best_estimator_
explainer = shap.TreeExplainer(best_model)
shap_values_train = explainer.shap_values(X_train)

# ---- SHAP summary table -> Excel (append) ----
mean_abs_shap = np.abs(shap_values_train).mean(axis=0)
df_shap_summary = (
    pd.DataFrame({'Feature': FEATURES, 'Mean |SHAP|': mean_abs_shap})
      .sort_values('Mean |SHAP|', ascending=False)
      .reset_index(drop=True)
)
with pd.ExcelWriter('XGB_RandomizedSearch_Results.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df_shap_summary.to_excel(writer, sheet_name='SHAP_Summary', index=False)

                                       
FEATURES_FULL = [
    'Pressure, psi',
    'Gas Type',
    'Composition CO2%',
    'Ash content, wt.%',
    'Fuel Ratio',
    'Temperature, °C'
]

                                      
FEATURES_FULL = [
    'Pressure, psi',
    'Temperature, °C',
    'Ash content, wt.%',
    'Fuel Ratio',
    'Composition CO2%',
    'Gas Type'
]

# ------------ SHAP beeswarm (full names, no cutting) ------------
plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values_train, X_train,
    feature_names=FEATURES_FULL,                     
    show=False, max_display=30                                         
)
ax = plt.gca()
ax.margins(x=0.02)
fig = plt.gcf()
fig.subplots_adjust(left=0.34, right=0.995, bottom=0.18, top=0.98)
plt.savefig('SHAP_Summary_Beeswarm.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()

# ------------ SHAP bar (full names, no cutting) ------------
plt.figure(figsize=(12, 4))
shap.summary_plot(
    shap_values_train, X_train,
    feature_names=FEATURES_FULL,
    plot_type='bar',
    show=False, max_display=30
)
ax = plt.gca()
ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', labelpad=12)
ax.margins(x=0.02)
fig = plt.gcf()
fig.subplots_adjust(left=0.34, right=0.995, bottom=0.30, top=0.98)
plt.savefig('SHAP_Summary_Bar.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()


# ---- Dependence plots for top-K features ----
topk = 6
top_features = df_shap_summary['Feature'].head(topk).tolist()
for f in top_features:
    plt.figure()
    shap.dependence_plot(f, shap_values_train, X_train, feature_names=FEATURES, show=False)
    plt.tight_layout()
    safe_name = re.sub(r'[^A-Za-z0-9_]+', '_', f)
    plt.savefig(f'SHAP_Dependence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---- Optional: per-row SHAP on a small test slice -> Excel ----
test_slice = min(200, X_test.shape[0])
X_test_slice = X_test.iloc[:test_slice]
sv_test = explainer.shap_values(X_test_slice)
df_sv = pd.DataFrame(sv_test, columns=FEATURES)
df_sv.insert(0, 'Prediction', best_model.predict(X_test_slice))

ev = explainer.expected_value
if isinstance(ev, (list, np.ndarray)):
    ev = ev[0] if len(np.atleast_1d(ev)) > 0 else float(ev)
elif not np.isscalar(ev):
    ev = float(ev)
df_sv.insert(1, 'ExpectedValue', ev)

with pd.ExcelWriter('XGB_RandomizedSearch_Results.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df_sv.to_excel(writer, sheet_name='SHAP_TestSlice', index=False)

print("Saved: SHAP_Summary_Beeswarm.png, SHAP_Summary_Bar.png, SHAP_Dependence_*.png and Excel sheets 'SHAP_Summary', 'SHAP_TestSlice'")
