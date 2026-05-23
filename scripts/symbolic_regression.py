from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    try:
        ip.run_line_magic("clear", "")
        ip.run_line_magic("reset", "-sf")
    except Exception:
        pass

import json, gc, warnings, inspect, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

import joblib
from pysr import PySRRegressor
import sympy as sp


# ============================ Helpers ============================

def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def slugify(name: str) -> str:
    s = str(name)
    s = s.replace("%", "pct").replace(" ", "_").replace("/", "_per_")
    s = s.replace("(", "").replace(")", "").replace(",", "").replace("__", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def qbins(series, q=8):
    s = pd.to_numeric(series, errors="coerce")
    keep = s.notna()
    sv = s[keep]
    for k in range(q, 2, -1):
        try:
            b = pd.qcut(sv, q=k, duplicates="drop").astype(str)
        except Exception:
            continue
        return pd.Series(index=s.index, data=np.where(keep, b, "na"), dtype=object)
    return pd.Series(index=s.index, data=["na"] * len(s), dtype=object)

def pysr_supported_kwargs(**kwargs):
    sig = inspect.signature(PySRRegressor.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}

def _safe_symbol(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = "v_" + s
    return s

def get_sympy_expr_from_model(model: PySRRegressor, eq_index: int):
    for call in (
        lambda: model.sympy(index=eq_index),
        lambda: model.sympy(eq_index),
        lambda: model.get_sympy(index=eq_index),
        lambda: model.get_sympy(eq_index),
    ):
        try:
            out = call()
            if out is not None:
                return out
        except Exception:
            pass

    eqs = model.equations_
    row = eqs.iloc[eq_index]
    for col in ["sympy_format", "sympy", "equation"]:
        if col in row.index:
            s = row[col]
            if isinstance(s, str) and s.strip():
                try:
                    return sp.sympify(s)
                except Exception:
                    pass
    raise RuntimeError("Could not extract a sympy expression from the PySR model.")

def replace_xi(expr, new_syms):
    repl = {sp.Symbol(f"x{i}"): new_syms[i] for i in range(len(new_syms))}
    repl.update({sp.Symbol(f"X{i}"): new_syms[i] for i in range(len(new_syms))})
    return sp.simplify(expr.xreplace(repl))

def build_lambdify(expr, var_syms, mode="protected"):
    """
    Protected mode keeps P=0 safe (log/sqrt protected).
    """
    if mode == "standard":
        modules = {"log": np.log, "sqrt": np.sqrt, "Abs": np.abs, "exp": np.exp, "pow": np.power}
    elif mode == "protected":
        def plog(x):  return np.log(np.abs(x) + 1e-12)
        def psqrt(x): return np.sqrt(np.abs(x))
        modules = {"log": plog, "sqrt": psqrt, "Abs": np.abs, "exp": np.exp, "pow": np.power}
    else:
        raise ValueError("mode must be 'standard' or 'protected'.")
    return sp.lambdify(var_syms, expr, modules=[modules, "numpy"])

def eval_expr(lamb, X):
    cols = [X[:, i] for i in range(X.shape[1])]
    y = lamb(*cols)
    return np.asarray(y, dtype=np.float64)

def diff_stats(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    return {
        "max_abs": float(np.nanmax(np.abs(d))),
        "mean_abs": float(np.nanmean(np.abs(d))),
        "rmse": float(np.sqrt(np.nanmean(d**2))),
        "nan_a": int(np.isnan(a).sum()),
        "nan_b": int(np.isnan(b).sum()),
    }

def clip0(y):
    return np.maximum(0.0, np.asarray(y, dtype=np.float64))

def report_negative_preds(name, y):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    nneg = int(np.sum(y < 0))
    ymin = float(np.min(y)) if n else float("nan")
    if nneg > 0:
        print(f"[NEG PRED] {name}: count={nneg}/{n} ({100*nneg/n:.2f}%), min={ymin:.6g}")
    else:
        print(f"[NEG PRED] {name}: none (min={ymin:.6g})")

def round_sympy_constants_pretty(expr, dec_ge1=4, dec_lt1=5):
    """
    Rounds ONLY numeric constants for reporting.
    """
    repl = {}
    for n in expr.atoms(sp.Number):
        try:
            v = float(n)
        except Exception:
            continue
        if abs(v - round(v)) < 1e-15:
            repl[n] = sp.Integer(int(round(v)))
            continue
        d = dec_ge1 if abs(v) >= 1.0 else dec_lt1
        rv = round(v, d)
        if abs(rv) == 0.0:
            rv = 0.0
        repl[n] = sp.Float(rv)
    return expr.xreplace(repl)

def make_stratified_train_test_indices(y_values, test_size=0.2, random_state=42, q=8):
    labels = qbins(pd.Series(y_values), q=q)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx = np.arange(len(labels))
    (tr_idx, te_idx) = next(sss.split(np.zeros(len(idx)), labels))
    return tr_idx, te_idx

def pick_best_equation_by_val(
    model: PySRRegressor,
    X_val: np.ndarray,
    y_val: np.ndarray,
    var_syms,
    max_complexity=None,
    require_all_vars=False,
):
    """
    Pick best equation by VAL R2 (RAW eq eval), tie VAL RMSE, tie lower complexity.
    Uses lambdified sympy for stable scoring (protected eval).
    """
    eqs = model.equations_
    if eqs is None or len(eqs) == 0:
        raise RuntimeError("model.equations_ is empty.")

    required = set(var_syms)
    best_i, best_r2, best_rm, best_c = None, -np.inf, np.inf, np.inf

    for i in range(len(eqs)):
        c = eqs.iloc[i].get("complexity", np.nan)
        c = float(c) if pd.notna(c) else np.inf
        if max_complexity is not None and c > max_complexity:
            continue

        try:
            expr_raw = get_sympy_expr_from_model(model, i)
            expr_vars = replace_xi(expr_raw, var_syms)

            if require_all_vars:
                used = set(expr_vars.free_symbols)
                if not required.issubset(used):
                    continue

            lamb = build_lambdify(expr_vars, var_syms, mode="protected")
            yhat = eval_expr(lamb, X_val)
            if np.any(~np.isfinite(yhat)):
                continue

            r2v = float(r2_score(y_val, yhat))
            rmv = rmse(y_val, yhat)

        except Exception:
            continue

        if (r2v > best_r2) or (np.isclose(r2v, best_r2) and rmv < best_rm) or \
           (np.isclose(r2v, best_r2) and np.isclose(rmv, best_rm) and c < best_c):
            best_i, best_r2, best_rm, best_c = i, r2v, rmv, c

    if best_i is None:
        raise RuntimeError("No equation found that satisfies constraints.")
    return best_i, eqs.iloc[best_i].to_dict(), best_r2, best_rm, best_c


# ============================ CONFIG ============================

EXCEL_IN   = "Dataset.xlsx"
SHEET_NAME = "Data"

P_COL   = "P, psi"
GAS_COL = "Gas Type"

                                                          
                                                                                              
FEATURES_BASE = [
    "P, psi",
    "T, °C",
    "Ash content, wt.%",
    "Fuel_Ratio",
    "Composition CO2%",
]
FEATURES = FEATURES_BASE + ["g"]                                          

TARGET = "Gas Adsorption, scf/ton"

TEST_SIZE = 0.20
RANDOM_STATE = 42

                      
N_SPLITS_CV = 5
CV_RANDOM_STATE = 42
QBINS_Q = 8

                       
REQUIRE_ALL_VARIABLES = False
MAX_COMPLEXITY_FOR_SELECTION = None

                 
CLIP_PRED_TO_0 = True

                       
VALID_MAX_ABS_PYSR_VS_EQ = 1e-6
VALID_RMSE_ABS_PYSR_VS_EQ = 1e-6

                             
VALID_MAX_ABS_ROUND_VS_EXACT = 5e-4
VALID_RMSE_ABS_ROUND_VS_EXACT = 5e-4

OUTDIR = safe_mkdir(Path("ML_Results"))
ROOT_DIR = safe_mkdir(OUTDIR / "PYSR_ONE_EQUATION__KEEP_P0__CLIP0__VALIDATE_ROUNDED__EXCEL_TEMPLATE")


# ============================ Fixed PySR budget (NO tuning) ============================

FIXED_PYSR_KWARGS = dict(
    populations=12,
    population_size=65,
    ncycles_per_iteration=85,

    niterations=550,
    timeout_in_seconds=60 * 60 * 2,

    maxsize=60,

    binary_operators=["*", "+", "-", "/"],
    unary_operators=["sqrt", "log", "square", "cube"],

    constraints={"/": (-1, 10), "sqrt": 10, "log": 8, "square": 10, "cube": 10},
    nested_constraints={
        "log": {"log": 0, "sqrt": 1, "square": 1, "cube": 1},
        "sqrt": {"log": 1, "sqrt": 0, "square": 1, "cube": 1},
        "square": {"log": 1, "sqrt": 1, "square": 1, "cube": 1},
        "cube": {"log": 1, "sqrt": 1, "square": 1, "cube": 1},
    },

    progress=False,
    verbosity=0,

    batching=True,
    batch_size=10000,

    weight_randomize=0.1,
    precision=32,
    warm_start=False,
    turbo=True,

    early_stop_condition=None,

    parallelism="multithreading",
    procs=6,
)


# ============================ Excel template helpers (CatBoost-like) ============================

def build_hyper_table(run_name: str, used_kwargs: dict) -> pd.DataFrame:
    rows = []
    for k, v in sorted(used_kwargs.items(), key=lambda x: x[0]):
        rows.append([k, "", str(v)])
    df_h = pd.DataFrame(rows, columns=["Hyperparameter", "Values / Range", "Optimized value"])
    df_h.insert(0, "ML Tool", [f"PySR (fixed budget) | {run_name}"] + [""]*(len(df_h)-1))
    return df_h

def build_pretty_metrics_table(train_rmse, train_r2, test_rmse, test_r2) -> pd.DataFrame:
    rows = []
    rows.append(["Training data", "RMSE, scf/ton", float(train_rmse)])
    rows.append(["",              "R², fraction",  float(train_r2)])
    rows.append(["Testing data",  "RMSE, scf/ton", float(test_rmse)])
    rows.append(["",              "R², fraction",  float(test_r2)])
    return pd.DataFrame(rows, columns=["", "", "PySR"])

def build_cv_results_table(df_cv_folds: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["Fold", "Status", "Val_R2_clipped", "Val_RMSE_clipped", "Complexity", "Error", "Best_Expression"]
    keep_cols = [c for c in keep_cols if c in df_cv_folds.columns]
    return df_cv_folds[keep_cols].copy()

def save_run_excel(
    excel_path: Path,
    df_hyper: pd.DataFrame,
    df_err: pd.DataFrame,
    df_cv: pd.DataFrame,
    df_best_row: pd.DataFrame,
    df_train_pred: pd.DataFrame,
    df_test_pred: pd.DataFrame,
    df_val_train: pd.DataFrame,
    df_val_test: pd.DataFrame,
):
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_hyper.to_excel(writer, sheet_name="Hyperparameter_Table", index=False)
        df_err.to_excel(writer, sheet_name="Error_Metrics", index=False)
        df_cv.to_excel(writer, sheet_name="CV_Results", index=False)
        df_best_row.to_excel(writer, sheet_name="Best_Equation_Row", index=False)
        df_train_pred.to_excel(writer, sheet_name="Train_Actual_vs_Pred", index=True)
        df_test_pred.to_excel(writer, sheet_name="Test_Actual_vs_Pred", index=True)
        df_val_train.to_excel(writer, sheet_name="Validate_Train", index=False)
        df_val_test.to_excel(writer, sheet_name="Validate_Test", index=False)


# ============================ Core runner (ONE model) ============================

def run_one_pysr(
    d: pd.DataFrame,
    features: list,
    target: str,
    outdir: Path,
    run_name: str,
    random_state_outer: int,
    test_size: float,
    n_splits_cv: int,
    cv_random_state: int,
    clip_pred_to_0: bool,
    qbins_q: int = 8,
):
    outdir = safe_mkdir(outdir)
    tag = slugify(run_name)
    tdir = safe_mkdir(outdir / tag)

                        
    d = d.dropna(subset=features + [target]).reset_index(drop=True)
    if d.empty:
        raise RuntimeError(f"No rows after dropna for run: {run_name}")

         
    X_all = d[features].values.astype(np.float64, copy=False)
    y_all = pd.to_numeric(d[target], errors="coerce").astype(float).values

    if (y_all < 0).any():
        raise ValueError(f"Negative targets found in y for run: {run_name}")

                 
    tr_idx, te_idx = make_stratified_train_test_indices(
        y_all, test_size=test_size, random_state=random_state_outer, q=qbins_q
    )
    Xtr_full = X_all[tr_idx]
    ytr_full = y_all[tr_idx]
    Xte = X_all[te_idx]
    yte = y_all[te_idx]

    split_json = {
        "run_name": run_name,
        "test_size": test_size,
        "random_state": random_state_outer,
        "train_n": int(len(tr_idx)),
        "test_n": int(len(te_idx)),
        "features": list(features),
        "target": target,
        "keep_pressure_zero": True,
    }
    (tdir / "train_test_split.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")

                            
    clean_feat = [_safe_symbol(f) for f in features]
    var_names = clean_feat
    var_syms = [sp.Symbol(n, real=True) for n in var_names]

                       
    labels_tr = qbins(pd.Series(ytr_full), q=qbins_q)
    skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=cv_random_state)

    cv_rows = []

    print("\n" + "-" * 110)
    print(f"[RUN] {run_name}")
    print(f"[SPLIT] train_rows={len(Xtr_full)} test_rows={len(Xte)} | CV folds={n_splits_cv}")
    print(f"[FIXED BUDGET] populations=12 population_size=65 ncycles_per_iteration=85 | NO tuning")
    print("[NOTE] KEEP P0: rows with P==0 are kept. Protected log/sqrt handle P==0 safely.")
    print("-" * 110)

    for fold_id, (tr_in, va_in) in enumerate(skf.split(np.zeros(len(ytr_full)), labels_tr), start=1):
        X_tr, y_tr = Xtr_full[tr_in], ytr_full[tr_in]
        X_va, y_va = Xtr_full[va_in], ytr_full[va_in]

        used_kwargs = pysr_supported_kwargs(**dict(FIXED_PYSR_KWARGS))
        used_kwargs["random_state"] = int(100000 + fold_id + (hash(run_name) % 10000))

        model = None

        # FIT
        try:
            model = PySRRegressor(**used_kwargs)
            model.fit(X_tr, y_tr)
        except Exception as e:
            cv_rows.append([fold_id, "FIT_FAIL", np.nan, np.nan, np.nan, str(e)[:160], ""])
            if model is not None:
                del model
            gc.collect()
            continue

        # PICK BEST EQ ON VAL
        try:
            best_idx, best_row, _, _, comp = pick_best_equation_by_val(
                model, X_va, y_va, var_syms,
                max_complexity=MAX_COMPLEXITY_FOR_SELECTION,
                require_all_vars=REQUIRE_ALL_VARIABLES
            )
            expr = str(get_sympy_expr_from_model(model, best_idx))
        except Exception as e:
            cv_rows.append([fold_id, "PICK_FAIL", np.nan, np.nan, np.nan, str(e)[:160], ""])
            if model is not None:
                del model
            gc.collect()
            continue

        # PREDICT VAL
        try:
            try:
                yhat_va_raw = model.predict(X_va, index=best_idx)
            except TypeError:
                yhat_va_raw = model.predict(X_va, equation_index=best_idx)
        except Exception as e:
            cv_rows.append([fold_id, "PRED_FAIL", np.nan, np.nan, np.nan, str(e)[:160], expr])
            if model is not None:
                del model
            gc.collect()
            continue

        yhat_va = clip0(yhat_va_raw) if clip_pred_to_0 else np.asarray(yhat_va_raw, dtype=np.float64)
        r2_va = float(r2_score(y_va, yhat_va))
        rm_va = rmse(y_va, yhat_va)

        cv_rows.append([fold_id, "OK", r2_va, rm_va, comp, "", expr])
        print(f"[CV] fold={fold_id} | Val R2={r2_va:.6f} | Val RMSE={rm_va:.6g} | comp={comp}")

        if model is not None:
            del model
        gc.collect()

    df_cv = pd.DataFrame(
        cv_rows,
        columns=["Fold", "Status", "Val_R2_clipped", "Val_RMSE_clipped", "Complexity", "Error", "Best_Expression"]
    )
    cv_ok = df_cv[df_cv["Status"] == "OK"].copy()
    if cv_ok.empty:
        raise RuntimeError(f"ALL CV folds failed for run: {run_name}")

    cv_mean_r2 = float(cv_ok["Val_R2_clipped"].mean())
    cv_std_r2  = float(cv_ok["Val_R2_clipped"].std(ddof=1)) if len(cv_ok) > 1 else 0.0
    cv_mean_rm = float(cv_ok["Val_RMSE_clipped"].mean())
    cv_std_rm  = float(cv_ok["Val_RMSE_clipped"].std(ddof=1)) if len(cv_ok) > 1 else 0.0
    cv_median_comp = float(np.nanmedian(cv_ok["Complexity"].values))

    print(f"[CV SUMMARY] R2 mean±std = {cv_mean_r2:.6f} ± {cv_std_r2:.6f} | "
          f"RMSE mean±std = {cv_mean_rm:.6g} ± {cv_std_rm:.6g} | median comp = {cv_median_comp:.2f}")

                             
    used_kwargs_final = pysr_supported_kwargs(**dict(FIXED_PYSR_KWARGS))
    used_kwargs_final["random_state"] = int(777777 + (hash(run_name) % 10000))

    print("[FIT FINAL] fitting on FULL TRAIN...")
    final_model = PySRRegressor(**used_kwargs_final)
    final_model.fit(Xtr_full, ytr_full)

                                 
    best_idx, best_row, tr_r2_raw_pick, tr_rm_raw_pick, best_comp = pick_best_equation_by_val(
        final_model, Xtr_full, ytr_full, var_syms,
        max_complexity=MAX_COMPLEXITY_FOR_SELECTION,
        require_all_vars=REQUIRE_ALL_VARIABLES
    )

                               
    try:
        yhat_tr_raw = final_model.predict(Xtr_full, index=best_idx)
        yhat_te_raw = final_model.predict(Xte, index=best_idx)
    except TypeError:
        yhat_tr_raw = final_model.predict(Xtr_full, equation_index=best_idx)
        yhat_te_raw = final_model.predict(Xte, equation_index=best_idx)

    report_negative_preds("PySR TRAIN (raw)", yhat_tr_raw)
    report_negative_preds("PySR TEST  (raw)", yhat_te_raw)

    yhat_tr = clip0(yhat_tr_raw) if clip_pred_to_0 else np.asarray(yhat_tr_raw, dtype=np.float64)
    yhat_te = clip0(yhat_te_raw) if clip_pred_to_0 else np.asarray(yhat_te_raw, dtype=np.float64)

    tr_r2 = float(r2_score(ytr_full, yhat_tr))
    tr_rm = rmse(ytr_full, yhat_tr)
    te_r2 = float(r2_score(yte, yhat_te))
    te_rm = rmse(yte, yhat_te)

    print(f"[FINAL] idx={best_idx} | Train R2(clipped)={tr_r2:.6f} RMSE={tr_rm:.6g} | "
          f"Test R2(clipped)={te_r2:.6f} RMSE={te_rm:.6g} | complexity={best_comp}")

                          
    eq_csv = tdir / "FINAL_equations.csv"
    final_model.equations_.to_csv(eq_csv, index=False)

                                                                   
    sym_expr_raw = get_sympy_expr_from_model(final_model, best_idx)
    sym_expr_vars_exact = replace_xi(sym_expr_raw, var_syms)
    sym_expr_vars_round = round_sympy_constants_pretty(sym_expr_vars_exact, dec_ge1=4, dec_lt1=5)

                                                              
    lamb_exact = build_lambdify(sym_expr_vars_exact, var_syms, mode="protected")
    lamb_round = build_lambdify(sym_expr_vars_round, var_syms, mode="protected")

    ytr_eq_exact_raw = eval_expr(lamb_exact, Xtr_full)
    yte_eq_exact_raw = eval_expr(lamb_exact, Xte)

    ytr_eq_round_raw = eval_expr(lamb_round, Xtr_full)
    yte_eq_round_raw = eval_expr(lamb_round, Xte)

    ytr_eq_exact = clip0(ytr_eq_exact_raw) if clip_pred_to_0 else np.asarray(ytr_eq_exact_raw, dtype=np.float64)
    yte_eq_exact = clip0(yte_eq_exact_raw) if clip_pred_to_0 else np.asarray(yte_eq_exact_raw, dtype=np.float64)

    ytr_eq_round = clip0(ytr_eq_round_raw) if clip_pred_to_0 else np.asarray(ytr_eq_round_raw, dtype=np.float64)
    yte_eq_round = clip0(yte_eq_round_raw) if clip_pred_to_0 else np.asarray(yte_eq_round_raw, dtype=np.float64)

                                                   
    diff_tr_pysr_vs_exact = diff_stats(yhat_tr, ytr_eq_exact)
    diff_te_pysr_vs_exact = diff_stats(yhat_te, yte_eq_exact)
    validated_pysr_vs_exact = (
        (diff_te_pysr_vs_exact["max_abs"] <= VALID_MAX_ABS_PYSR_VS_EQ) and
        (diff_te_pysr_vs_exact["rmse"] <= VALID_RMSE_ABS_PYSR_VS_EQ) and
        (diff_te_pysr_vs_exact["nan_b"] == 0)
    )

                                                        
    diff_tr_round_vs_exact = diff_stats(ytr_eq_round, ytr_eq_exact)
    diff_te_round_vs_exact = diff_stats(yte_eq_round, yte_eq_exact)
    validated_round_vs_exact = (
        (diff_te_round_vs_exact["max_abs"] <= VALID_MAX_ABS_ROUND_VS_EXACT) and
        (diff_te_round_vs_exact["rmse"] <= VALID_RMSE_ABS_ROUND_VS_EXACT) and
        (diff_te_round_vs_exact["nan_a"] == 0) and
        (diff_te_round_vs_exact["nan_b"] == 0)
    )

    print(f"[VALIDATION PySR vs EqExact] Test max_abs={diff_te_pysr_vs_exact['max_abs']:.3e} rmse={diff_te_pysr_vs_exact['rmse']:.3e} | "
          f"{'✅' if validated_pysr_vs_exact else '❌'}")
    print(f"[VALIDATION EqRounded vs EqExact] Test max_abs={diff_te_round_vs_exact['max_abs']:.3e} rmse={diff_te_round_vs_exact['rmse']:.3e} | "
          f"{'✅' if validated_round_vs_exact else '❌'}")

                        
    eq_txt = tdir / "FINAL_equation.txt"
    with open(eq_txt, "w", encoding="utf-8") as f:
        f.write(f"RUN_NAME: {run_name}\n")
        f.write(f"TARGET: {target}\n\n")
        f.write("DATA:\n")
        f.write("  KEEP P0: Pressure==0 rows are kept (no pre-filter P<=0).\n\n")
        f.write("GAS ENCODING:\n")
        f.write("  g = 1 if Gas Type == 2 else 0 (binary indicator)\n\n")
        f.write("HYPERPARAM_TUNING:\n  None (fixed PySR hyperparameters; CV used only for equation selection)\n\n")
        f.write("FIXED_PYSR_BUDGET:\n")
        f.write("  populations=12\n  population_size=65\n  ncycles_per_iteration=85\n\n")
        f.write("OUTER_SPLIT:\n")
        f.write(f"  Stratified by y-bins; test_size={test_size}; random_state={random_state_outer}\n\n")
        f.write("INNER_CV (TRAIN ONLY):\n")
        f.write(f"  StratifiedKFold n_splits={n_splits_cv}; random_state={cv_random_state}\n")
        f.write(f"  CV R2 mean±std:   {cv_mean_r2:.10f} ± {cv_std_r2:.10f}\n")
        f.write(f"  CV RMSE mean±std: {cv_mean_rm:.10g} ± {cv_std_rm:.10g}\n")
        f.write(f"  CV median complexity (fold-best): {cv_median_comp:.6g}\n\n")
        f.write("FEATURES (raw):\n")
        for i, feat in enumerate(features):
            f.write(f"  x{i} = {feat}   (symbol: {var_names[i]})\n")
        f.write("\nPOST_PROCESSING (PREDICTIONS):\n")
        f.write("  y_final = max(0, y_raw)\n\n")
        f.write("FINAL_EQUATION_SELECTION (on TRAIN):\n")
        f.write("  maximize Train R2 (tie Train RMSE, tie lower complexity)\n")
        f.write(f"  BEST_INDEX: {best_idx}\n")
        f.write(f"  COMPLEXITY: {best_comp}\n\n")
        f.write("FINAL_METRICS:\n")
        f.write(f"  Train R2 (raw_pick):   {tr_r2_raw_pick:.10f}\n")
        f.write(f"  Train RMSE (raw_pick): {tr_rm_raw_pick:.10g}\n")
        f.write(f"  Train R2 (clipped):    {tr_r2:.10f}\n")
        f.write(f"  Train RMSE (clipped):  {tr_rm:.10g}\n")
        f.write(f"  Test  R2 (clipped):    {te_r2:.10f}\n")
        f.write(f"  Test  RMSE (clipped):  {te_rm:.10g}\n\n")

        f.write("BEST_EQUATION_ROW:\n")
        for k, v in best_row.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nEQUATION (EXACT, in RAW variables):\n")
        f.write(str(sym_expr_vars_exact) + "\n\n")
        f.write("EQUATION (ROUNDED for reporting):\n")
        f.write(str(sym_expr_vars_round) + "\n\n")

        f.write("VALIDATION (CLIPPED):\n")
        f.write("  Mode: protected lambdify\n\n")
        f.write("  PySR vs EqExact:\n")
        f.write(f"    Train: max_abs={diff_tr_pysr_vs_exact['max_abs']:.6g}, rmse={diff_tr_pysr_vs_exact['rmse']:.6g}, nan_eq={diff_tr_pysr_vs_exact['nan_b']}\n")
        f.write(f"    Test : max_abs={diff_te_pysr_vs_exact['max_abs']:.6g}, rmse={diff_te_pysr_vs_exact['rmse']:.6g}, nan_eq={diff_te_pysr_vs_exact['nan_b']}\n")
        f.write(f"    Thresholds: max_abs<={VALID_MAX_ABS_PYSR_VS_EQ} rmse<={VALID_RMSE_ABS_PYSR_VS_EQ}\n")
        f.write(f"    Status: {'VALID' if validated_pysr_vs_exact else 'NOT VALID'}\n\n")

        f.write("  EqRounded vs EqExact:\n")
        f.write(f"    Train: max_abs={diff_tr_round_vs_exact['max_abs']:.6g}, rmse={diff_tr_round_vs_exact['rmse']:.6g}\n")
        f.write(f"    Test : max_abs={diff_te_round_vs_exact['max_abs']:.6g}, rmse={diff_te_round_vs_exact['rmse']:.6g}\n")
        f.write(f"    Thresholds: max_abs<={VALID_MAX_ABS_ROUND_VS_EXACT} rmse<={VALID_RMSE_ABS_ROUND_VS_EXACT}\n")
        f.write(f"    Status: {'VALID' if validated_round_vs_exact else 'NOT VALID'}\n\n")

        f.write("USED_KWARGS (FINAL REFIT):\n")
        for k, v in sorted(used_kwargs_final.items(), key=lambda x: x[0]):
            f.write(f"  {k}: {v}\n")

    # ===================== Excel outputs (CatBoost-like template) =====================

    df_cv_results = build_cv_results_table(df_cv)
    df_hyper = build_hyper_table(run_name, used_kwargs_final)
    df_err = build_pretty_metrics_table(tr_rm, tr_r2, te_rm, te_r2)

                                                
    tr_index = d.index.to_numpy()[tr_idx]
    te_index = d.index.to_numpy()[te_idx]

    df_train_pred = pd.DataFrame({"Actual": ytr_full}, index=tr_index)
    df_train_pred["Predicted"] = yhat_tr
    df_test_pred = pd.DataFrame({"Actual": yte}, index=te_index)
    df_test_pred["Predicted"] = yhat_te

    df_best = pd.DataFrame([best_row])

    df_val_train = pd.DataFrame({
        "Actual": ytr_full,
        "Pred_PySR_Clipped": yhat_tr,
        "Pred_EqExact_Clipped": ytr_eq_exact,
        "Pred_EqRounded_Clipped": ytr_eq_round,
        "AbsDiff_PySR_vs_Exact": np.abs(yhat_tr - ytr_eq_exact),
        "AbsDiff_Round_vs_Exact": np.abs(ytr_eq_round - ytr_eq_exact),
    })
    df_val_test = pd.DataFrame({
        "Actual": yte,
        "Pred_PySR_Clipped": yhat_te,
        "Pred_EqExact_Clipped": yte_eq_exact,
        "Pred_EqRounded_Clipped": yte_eq_round,
        "AbsDiff_PySR_vs_Exact": np.abs(yhat_te - yte_eq_exact),
        "AbsDiff_Round_vs_Exact": np.abs(yte_eq_round - yte_eq_exact),
    })

    out_xlsx = tdir / "PYSR_Results.xlsx"
    save_run_excel(
        excel_path=out_xlsx,
        df_hyper=df_hyper,
        df_err=df_err,
        df_cv=df_cv_results,
        df_best_row=df_best,
        df_train_pred=df_train_pred,
        df_test_pred=df_test_pred,
        df_val_train=df_val_train,
        df_val_test=df_val_test,
    )

                  
    bundle_path = tdir / "FINAL_bundle.joblib"
    joblib.dump(
        {
            "run_name": run_name,
            "target": target,
            "features": features,
            "var_names_sanitized": var_names,
            "outer_split": split_json,
            "cv": {
                "n_splits": n_splits_cv,
                "cv_seed": cv_random_state,
                "folds_table": df_cv,
                "cv_mean_r2": cv_mean_r2,
                "cv_std_r2": cv_std_r2,
                "cv_mean_rmse": cv_mean_rm,
                "cv_std_rmse": cv_std_rm,
                "cv_median_complexity": cv_median_comp,
            },
            "best": {
                "best_index": best_idx,
                "best_equation_row": best_row,
                "complexity": best_comp,
                "sympy_best_raw": str(sym_expr_raw),
                "sympy_best_vars_exact": str(sym_expr_vars_exact),
                "sympy_best_vars_rounded": str(sym_expr_vars_round),
                "metrics": {
                    "train_r2_raw_pick": tr_r2_raw_pick,
                    "train_rmse_raw_pick": tr_rm_raw_pick,
                    "train_r2_clipped": tr_r2,
                    "train_rmse_clipped": tr_rm,
                    "test_r2_clipped": te_r2,
                    "test_rmse_clipped": te_rm,
                },
                "validation_mode": "protected",
                "validation_pysr_vs_exact_train": diff_tr_pysr_vs_exact,
                "validation_pysr_vs_exact_test": diff_te_pysr_vs_exact,
                "validated_pysr_vs_exact": validated_pysr_vs_exact,
                "validation_round_vs_exact_train": diff_tr_round_vs_exact,
                "validation_round_vs_exact_test": diff_te_round_vs_exact,
                "validated_round_vs_exact": validated_round_vs_exact,
                "thresholds": {
                    "pysr_vs_exact": {"max_abs": VALID_MAX_ABS_PYSR_VS_EQ, "rmse": VALID_RMSE_ABS_PYSR_VS_EQ},
                    "round_vs_exact": {"max_abs": VALID_MAX_ABS_ROUND_VS_EXACT, "rmse": VALID_RMSE_ABS_ROUND_VS_EXACT},
                }
            },
            "model": final_model,
            "equations_df": final_model.equations_,
            "config_used": used_kwargs_final,
            "post_processing": "y_final = max(0, y_raw)" if clip_pred_to_0 else "none",
        },
        bundle_path
    )

    print("Saved:")
    print(" ", eq_csv)
    print(" ", eq_txt)
    print(" ", out_xlsx)
    print(" ", bundle_path)

    return {
        "Run": run_name,
        "Rows_Total": int(len(d)),
        "Train_Rows": int(len(Xtr_full)),
        "Test_Rows": int(len(Xte)),
        "CV_Mean_R2": cv_mean_r2,
        "CV_Std_R2": cv_std_r2,
        "CV_Mean_RMSE": cv_mean_rm,
        "CV_Std_RMSE": cv_std_rm,
        "CV_Median_Complexity": cv_median_comp,
        "Final_Train_R2_clipped": tr_r2,
        "Final_Train_RMSE_clipped": tr_rm,
        "Final_Test_R2_clipped": te_r2,
        "Final_Test_RMSE_clipped": te_rm,
        "Final_Complexity": float(best_comp),
        "Validated_PySR_vs_EqExact_CLIPPED": bool(validated_pysr_vs_exact),
        "Validated_EqRounded_vs_EqExact_CLIPPED": bool(validated_round_vs_exact),
        "OutDir": str(tdir),
    }


# ============================ Load dataset (KEEP P==0) ============================

print("Loading:", EXCEL_IN, "sheet:", SHEET_NAME)
df = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

                                                                               
needed_cols = FEATURES_BASE + [GAS_COL] + [TARGET]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in Excel: {missing}")

ensure_numeric(df, FEATURES_BASE + [GAS_COL] + [TARGET])

                                       
n_before = len(df)
df[P_COL] = pd.to_numeric(df[P_COL], errors="coerce")
df = df[df[P_COL].notna()].copy()
n_after = len(df)
print(f"[P0 POLICY] Kept P==0 rows. Dropped only NaN P rows: {n_before - n_after} | Remaining: {n_after}")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
if (df[TARGET] < 0).any():
    raise ValueError("Negative targets found in y. Fix/filter before running.")

df[GAS_COL] = pd.to_numeric(df[GAS_COL], errors="coerce")
print(f"[INFO] Rows total after keep-P0: {len(df)} | GasType counts:\n{df[GAS_COL].value_counts(dropna=False)}")

                                                
df = df[df[GAS_COL].notna()].copy()

                                             
                               
df["g"] = (df[GAS_COL].astype(int) == 2).astype(int)

                                     
bad = df[~df["g"].isin([0, 1])]
if not bad.empty:
    raise ValueError("Unexpected values created in g. Check Gas Type values.")

# ============================ RUN ONE MODEL ============================

summary = run_one_pysr(
    d=df,
    features=FEATURES,
    target=TARGET,
    outdir=ROOT_DIR,
    run_name="ONE_EQUATION__ALL_DATA__with_binary_g_indicator",
    random_state_outer=RANDOM_STATE,
    test_size=TEST_SIZE,
    n_splits_cv=N_SPLITS_CV,
    cv_random_state=CV_RANDOM_STATE,
    clip_pred_to_0=CLIP_PRED_TO_0,
    qbins_q=QBINS_Q,
)

# ============================ Global summary ============================

df_sum = pd.DataFrame([summary])
sum_xlsx = ROOT_DIR / "GLOBAL_SUMMARY.xlsx"
df_sum.to_excel(sum_xlsx, index=False)

print("\n" + "=" * 110)
print("[DONE] Wrote global summary ->", sum_xlsx.resolve())
print("Outputs root ->", ROOT_DIR.resolve())
print("=" * 110)
