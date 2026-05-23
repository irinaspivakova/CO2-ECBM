
import os, joblib, numpy as np, pandas as pd, tensorflow as tf
from math import log
from scipy.stats import loguniform, randint, uniform

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from scikeras.wrappers import KerasRegressor
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------- Config -------------------------
EXCEL_IN   = 'Dataset.xlsx'
SHEET_NAME = 'Data'
TARGET     = 'Gas Adsorption, scf/ton'

NUM_COLS = [
    'P, psi','T, °C','Ash content, wt.%','Fuel_Ratio',
    'Inertinite, vol. %','Composition CO2%'
]
CAT_COLS = ['Gas Type']                         

N_ITER      = 300                                                    
SEED        = 42
OUT_DIR     = "ANN_layers_search_allinone"
OUT_PREFIX  = "ANN_layers_1to4"

# -------------------- Small utilities --------------------
def make_strat_bins(y: pd.Series, max_bins: int = 8, min_count: int = 2) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    idx_valid = y.dropna().index
    for q in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y.loc[idx_valid], q=q, duplicates="drop")
        except Exception:
            continue
        if bins.value_counts().min() >= min_count:
            out = pd.Series(index=y.index, dtype=object)
            out.loc[idx_valid] = bins.astype(str)
            return out.fillna("all")
    return pd.Series(["all"] * len(y), index=y.index)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _vals_to_str(v):
    try:
        name = v.dist.name if hasattr(v, "dist") else None
        if name == "randint":    return f"[{int(v.kwds.get('low', v.args[0]))}..{int(v.kwds.get('high', v.args[1]))-1}]"
        if name == "uniform":    return f"[{v.kwds.get('loc', v.args[0]):.2f}..{(v.kwds.get('loc', v.args[0])+v.kwds.get('scale', v.args[1])):.2f}]"
        if name == "loguniform": return f"[{v.args[0]:.1e}..{v.args[1]:.1e}]"
    except Exception:
        pass
    return ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)

# --------------------- Load & split data ---------------------
df = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

for c in NUM_COLS + [TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[NUM_COLS + CAT_COLS].copy()
y = df[TARGET].astype(float).copy()

y_bins = make_strat_bins(y, max_bins=8, min_count=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, shuffle=True, stratify=y_bins
)

# ------------------------ Preprocessor ------------------------
def make_ohe():
    """Version-safe OneHotEncoder (sklearn >=1.2 uses sparse_output)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", make_ohe(), CAT_COLS),
    ],
    remainder="drop"
)

                                             
preprocess.fit(X_train)
INPUT_DIM = preprocess.transform(X_train).shape[1]

# ----------------------- Keras model -----------------------
def build_ann(n_hidden=2, units=128, shrink_rate=0.8, activation="relu",
              dropout=0.1, bn=True, lr=1e-3, l2=0.0, input_dim=None, **kwargs):
    """
    **kwargs absorbs any extra params SciKeras might forward (e.g., y_transformer).
    """
    m = Sequential()
    u = int(units)
    for i in range(int(n_hidden)):
        kwargs_dense = dict(
            units=u,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None
        )
        if i == 0 and input_dim is not None:
            m.add(Dense(**kwargs_dense, input_shape=(input_dim,)))
        else:
            m.add(Dense(**kwargs_dense))
        if bn:
            m.add(BatchNormalization())
        if dropout > 0:
            m.add(Dropout(dropout))
        u = max(32, int(u * float(shrink_rate)))

                         
    m.add(Dense(1, activation="softplus"))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m

                                                       
_log1p = FunctionTransformer(np.log1p, inverse_func=np.expm1)
try:
    reg = KerasRegressor(
        model=build_ann,
        epochs=300,
        batch_size=128,
        verbose=0,
        y_transformer=_log1p,                           
    )
except TypeError:
    reg = KerasRegressor(
        model=build_ann,
        epochs=300,
        batch_size=128,
        verbose=0,
        target_transformer=_log1p,                 
    )

pipe = Pipeline([
    ("prep", preprocess),
    ("ann", reg)
])

# ---------------------- Randomized search ----------------------
os.makedirs(OUT_DIR, exist_ok=True)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
r2_scorer = make_scorer(r2_score)
es = EarlyStopping(monitor="val_loss", mode="min", patience=20, restore_best_weights=True)

                                            
param_distributions = {
    "ann__model__n_hidden": [1, 2, 3, 4],
    "ann__model__units": randint(64, 257),                    
    "ann__model__shrink_rate": uniform(0.65, 0.3),               
    "ann__model__activation": ["relu", "elu", "gelu"],
    "ann__model__dropout": uniform(0.0, 0.3),
    "ann__model__bn": [True, False],
    "ann__model__lr": loguniform(1e-4, 1e-2),
    "ann__model__l2": loguniform(1e-7, 1e-4),
    "ann__epochs": randint(200, 401),
    "ann__batch_size": [64, 128],
                                                 
    "ann__model__input_dim": [INPUT_DIM],
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring=r2_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    random_state=SEED,
    return_train_score=False,
)

search.fit(X_train, y_train, ann__callbacks=[es], ann__validation_split=0.2)

# ---------------- Results, metrics, saves ----------------
y_tr = search.predict(X_train)
y_te = search.predict(X_test)

n_train, n_test = len(y_train), len(y_test)
mse_train = mean_squared_error(y_train, y_tr)
mse_test  = mean_squared_error(y_test,  y_te)
r2_train  = r2_score(y_train, y_tr)
r2_test   = r2_score(y_test,  y_te)
rmse_train = rmse(y_train, y_tr)
rmse_test  = rmse(y_test,  y_te)

                                            
k_params  = INPUT_DIM + 1
aic_train = n_train * log(max(mse_train, 1e-12)) + 2 * k_params
aic_test  = n_test  * log(max(mse_test,  1e-12)) + 2 * k_params
bic_train = n_train * log(max(mse_train, 1e-12)) + k_params * log(max(n_train, 2))
bic_test  = n_test  * log(max(mse_test,  1e-12)) + k_params * log(max(n_test,  2))

df_pretty_metrics = pd.DataFrame([
    ['Training data', 'RMSE (scf/ton)', rmse_train],
    ['',              'R² (fraction)',  r2_train],
    ['',              'AIC',            aic_train],
    ['',              'BIC',            bic_train],
    ['Testing data',  'RMSE (scf/ton)', rmse_test],
    ['',              'R² (fraction)',  r2_test],
    ['',              'AIC',            aic_test],
    ['',              'BIC',            bic_test],
], columns=['', '', 'ANN'])

                                                      
param_display = []
for name, vals in param_distributions.items():
    param_display.append((name, _vals_to_str(vals), search.best_params_.get(name, '')))
df_hyper_table = pd.DataFrame(param_display, columns=['Hyperparameter', 'Values / Range', 'Optimized value'])
df_hyper_table.insert(0, 'ML Tool', ['ANN (softplus, y=log1p, n_hidden∈{1..4})'] + ['']*(len(df_hyper_table)-1))

                                                            
cv_df = pd.DataFrame(search.cv_results_)
keep_cols = [c for c in cv_df.columns if c.startswith('param_ann__model__')]
keep_cols += ['param_ann__epochs', 'param_ann__batch_size', 'mean_test_score', 'rank_test_score']
K_fold_result = cv_df[keep_cols].copy()

                
df_train_pred = pd.DataFrame({'Actual': y_train, 'Predicted': y_tr}, index=X_train.index)
df_test_pred  = pd.DataFrame({'Actual': y_test,  'Predicted': y_te}, index=X_test.index)

                 
os.makedirs(OUT_DIR, exist_ok=True)
excel_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_Results.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    df_hyper_table.to_excel(writer, sheet_name='Hyperparameter_Table', index=False)
    df_pretty_metrics.to_excel(writer, sheet_name='Error_Metrics', index=False)
    K_fold_result.to_excel(writer, sheet_name='CV_Results', index=False)
    df_train_pred.to_excel(writer, sheet_name='Train_Actual_vs_Pred', index=True)
    df_test_pred.to_excel(writer, sheet_name='Test_Actual_vs_Pred', index=True)

model_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_best.pkl")
joblib.dump(search.best_estimator_, model_path)

print(f"\nBest CV R²: {search.best_score_:.4f}")
print(f"Train/Test R²: {r2_train:.4f} / {r2_test:.4f}")
print(f"Excel saved to: {excel_path}")
print(f"Best model saved to: {model_path}")
