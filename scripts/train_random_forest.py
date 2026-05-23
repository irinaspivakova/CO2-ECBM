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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import joblib

# ---------------- utils ----------------
def mape(actual, pred, eps=1e-9):
    actual = np.asarray(actual); pred = np.asarray(pred)
    denom = np.where(np.abs(actual) < eps, eps, np.abs(actual))
    return float(np.mean(np.abs((actual - pred) / denom)) * 100)

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

def _numfmt(x):
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf) >= 1e3 or (0 < abs(xf) < 1e-3):
        return f"{xf:.2e}"
    return f"{xf:.6g}"

def describe_distribution(obj):
    """Pretty-print a RandomizedSearch param distribution (lists/arrays shown verbatim up to 20 items)."""
    import numpy as _np
    if isinstance(obj, (list, tuple, _np.ndarray)):
        vals = ", ".join(_numfmt(v) for v in list(obj)[:20])
        if len(obj) > 20: vals += ", …"
        return f"[{vals}]"
              
    try:
        return str(obj)
    except Exception:
        return repr(obj)

# ====================================================================================================================
EXCEL_IN   = 'Dataset.xlsx'
SHEET_NAME = 'Data'

Total_Data = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
Total_Data.columns = [c.strip() for c in Total_Data.columns]

FEATURES = [
    'P, psi', 'T, °C', 'Ash content, wt.%','Fuel_Ratio', 'Composition CO2%', 'Gas Type'
]
TARGET = 'Gas Adsorption, scf/ton'

X = Total_Data[FEATURES].copy()
y = pd.to_numeric(Total_Data[TARGET], errors='coerce').astype(float)

                                                     
if 'Gas Type' in X.columns and not pd.api.types.is_numeric_dtype(X['Gas Type']):
    X['Gas Type'] = X['Gas Type'].astype('category').cat.codes

# ---------- Stratified (by target quantile bins) train/test split ----------
y_bins = make_strat_bins(y, max_bins=8, min_count=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y_bins
)

# ----- Model -----
base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# ----- Random Search Space (ACTUAL; no hardcoding in Excel later) -----
rng = np.random.default_rng(42)
real_max_features = list(np.round(rng.uniform(0.3, 1.0, size=15), 3))

# ----- Random Search Space (static + safe) -----
param_distributions = {
    'n_estimators'             : [300, 500, 800, 1200, 1500],
    'max_depth'                : [None, 6, 8, 10, 14, 20, 30, 40, 60, 80],
    'min_samples_split'        : [2, 5, 10, 15, 20, 30],
    'min_samples_leaf'         : [1, 2, 3, 4, 6, 8, 10],
    'min_weight_fraction_leaf' : [0.0, 0.01, 0.02, 0.04, 0.06, 0.08],
    'max_leaf_nodes'           : [None, 300, 600, 900, 1200, 1500, 2000, 2500, 3000],
                                                      
    'max_features'             : ['sqrt', 'log2', None, 0.2, 0.33, 0.5, 0.67, 0.8, 1.0],
    'bootstrap'                : [True],
    'max_samples'              : [None, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_impurity_decrease'    : [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    'ccp_alpha'                : [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    'criterion'                : ['squared_error', 'absolute_error'],
}

# ----- RandomizedSearchCV -----
r2_scorer = make_scorer(r2_score)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_distributions,
    n_iter=500,
    scoring=r2_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
    return_train_score=False,
    error_score='raise',                                 
)

     
search.fit(X_train, y_train)

            
df = pd.DataFrame(search.cv_results_)
K_fold_result = df[[c for c in [
    'param_n_estimators','param_max_features','param_max_depth',
    'param_min_samples_split','param_min_samples_leaf','param_bootstrap',
    'param_max_samples','param_min_impurity_decrease','param_ccp_alpha',
    'mean_test_score','rank_test_score'
] if c in df.columns]]

                                                                                    
Train_Pred = pd.Series(search.predict(X_train), index=X_train.index).clip(lower=0)
Test_Pred  = pd.Series(search.predict(X_test),  index=X_test.index ).clip(lower=0)

# ----- Metrics -----
n_train, n_test = len(y_train), len(y_test)
num_params = len(X_train.columns) + 1

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
print("Min Train Pred:", float(np.min(Train_Pred)))
print("Min Test  Pred:", float(np.min(Test_Pred)))

# ----- Hyperparameter table (FROM ACTUAL SPACE + BEST) -----
best_params = search.best_params_
param_display = []
for name, dist in param_distributions.items():
    desc = describe_distribution(dist)                
    opt_value = best_params.get(name, '')
    param_display.append((name, desc, opt_value))

df_hyper_table = pd.DataFrame(param_display, columns=['Hyperparameter', 'Values / Range (actual)', 'Optimized value'])
df_hyper_table.insert(0, 'ML Tool', ['Random Forest (Randomized, stratified split)'] + ['']*(len(df_hyper_table)-1))

# ----- Pretty metrics table -----
rows = []
rows += [['Training data', 'RMSE, scf/ton', rmse_train],
         ['',              'R², fraction',  r2_train],
         ['',              'AIC',           aic_train],
         ['',              'BIC',           bic_train]]
rows += [['Testing data',  'RMSE, scf/ton', rmse_test],
         ['',              'R², fraction',  r2_test],
         ['',              'AIC',           aic_test],
         ['',              'BIC',           bic_test]]
df_pretty_metrics = pd.DataFrame(rows, columns=['', '', 'Random Forest'])

# ----- Actual vs Predicted tables -----
df_train_pred = pd.DataFrame({'Actual': y_train, 'Predicted': Train_Pred}, index=X_train.index)
df_test_pred  = pd.DataFrame({'Actual': y_test,  'Predicted': Test_Pred},  index=X_test.index)

# ===================== SAVE: Excel + PKL =====================
with pd.ExcelWriter('RF_RandomizedSearch_Results.xlsx', engine='openpyxl') as writer:
    df_hyper_table.to_excel(writer, sheet_name='Hyperparameter_Table', index=False)
    df_pretty_metrics.to_excel(writer, sheet_name='Error_Metrics', index=False)
    K_fold_result.to_excel(writer, sheet_name='CV_Results', index=False)
    df_train_pred.to_excel(writer, sheet_name='Train_Actual_vs_Pred', index=True)
    df_test_pred.to_excel(writer,  sheet_name='Test_Actual_vs_Pred',  index=True)

                                      
joblib.dump(search.best_estimator_, 'RF_best_cv_random.pkl')

                                              
joblib.dump(search, 'RF_randomizedsearch_cv.pkl')

                                                 
final_rf = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
final_rf.fit(X, y)
joblib.dump(final_rf, 'RF_final_full_random.pkl')

print("Saved PKLs: RF_best_cv_random.pkl, RF_randomizedsearch_cv.pkl, RF_final_full_random.pkl")
print("Excel: RF_RandomizedSearch_Results.xlsx")

# ======================= SHAP plots & tables =======================
try:
    import shap
    shap_explainer = shap.TreeExplainer(search.best_estimator_)
    shap_values_train = shap_explainer.shap_values(X_train)

    mean_abs_shap = np.abs(shap_values_train).mean(axis=0)
    df_shap_summary = (
        pd.DataFrame({'Feature': FEATURES, 'Mean |SHAP|': mean_abs_shap})
        .sort_values('Mean |SHAP|', ascending=False)
        .reset_index(drop=True)
    )
    with pd.ExcelWriter('RF_RandomizedSearch_Results.xlsx',
                        mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df_shap_summary.to_excel(writer, sheet_name='SHAP_Summary', index=False)

                          
    plt.figure()
    shap.summary_plot(shap_values_train, X_train, feature_names=FEATURES, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig('RF_SHAP_Summary_Beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values_train, X_train, feature_names=FEATURES, plot_type='bar', show=False, max_display=30)
    plt.tight_layout()
    plt.savefig('RF_SHAP_Summary_Bar.png', dpi=300, bbox_inches='tight')
    plt.close()

                                       
    topk = 6
    top_features = df_shap_summary['Feature'].head(topk).tolist()
    for f in top_features:
        plt.figure()
        shap.dependence_plot(f, shap_values_train, X_train, feature_names=FEATURES, show=False)
        plt.tight_layout()
        safe_name = re.sub(r'[^A-Za-z0-9_]+', '_', f)
        plt.savefig(f'RF_SHAP_Dependence_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("Saved: RF_SHAP_Summary_Beeswarm.png, RF_SHAP_Summary_Bar.png, RF_SHAP_Dependence_*.png; Excel sheet 'SHAP_Summary'")
except Exception as e:
    print(f"SHAP skipped (install shap to enable): {e}")

# ----- Optional quick importance plot -----
try:
    importances = search.best_estimator_.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(7,5))
    plt.bar(range(len(order)), importances[order])
    plt.xticks(range(len(order)), X.columns[order], rotation=45, ha='right')
    plt.title("Random Forest Feature Importance (Best CV Model)")
    plt.tight_layout()
    plt.savefig("RF_Feature_Importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: RF_Feature_Importance.png")
except Exception:
    pass
