import os, re, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
mpl.rcParams["axes.grid"] = False
plt.rcParams["axes.grid"] = False

# ---------------------- CONFIG ----------------------
EXCEL_IN          = "Dataset.xlsx"
SHEET_IN          = 'Data'

PRESSURE_UNIT_IN  = "psi"                                            
PRESSURE_IS_GAUGE = False
ADS_UNIT_OUT      = "scf/ton"

# ---- PURE plots (unchanged) ----
# ---- PURE plots (unchanged visual intent) ----
SELECTIONS_PURE_BASE = [
    ("Coal A", "CO2", 100),
    ("Coal B", "CO2", 100),
    ("Coal C", "CH4",   0),
    ("Coal D", "CH4",   0),
]

# ---- MIXTURE plots (NEW) ----
MIXTURE_CASES = [
    ("Coal F", 40),
    ("Coal A", 50),
]

                                                                      
                                            
SELECTIONS_PURE = list(SELECTIONS_PURE_BASE)
_needed = {str(c).strip() for c, _ in MIXTURE_CASES}
for coal in sorted(_needed):
    SELECTIONS_PURE.append((coal, "CO2", 100))                 
    SELECTIONS_PURE.append((coal, "CH4", 0))                   

                                          
_seen = set()
SELECTIONS_PURE = [x for x in SELECTIONS_PURE if not (x in _seen or _seen.add(x))]

CO2_MATCH_TOL = 1.0       

# ---- MIXTURE plots (NEW) ----
MIXTURE_CASES = [
    ("Coal F", 40),                    
    ("Coal A", 50),         
]
MIXTURE_PLOT_GASES = ["CO2", "CH4"]
MIXTURE_DIR  = "mixture_plots"
MIXTURE_ERR_XLSX = "mixture_el_error_metrics.xlsx"
MIXTURE_CO2_TOL = 1.0       

            
IRINA_XGB_COL_EXACT = "Spivakova et al., 2025"                                   
THIS_XGB_COL_FORCE = None                                       

THIS_STUDY_XGB_CANDS = [
    "XGBoost Prediction (scf/ton)","XGB Pred (scf/ton)","XGBoost Pred (scf/ton)",
    "XGBoost Prediction","XGB Pred","XGBoost Pred",
    "XGBoost predicted (This study)","This study XGBoost",
]

                              
SMOOTH_ML = True
SMOOTH_N  = 320
SMOOTH_METHOD = "pchip"                        

               
PLOT_DIR   = "pure_plots"
PAR_XLSX   = "pure_model_parameters.xlsx"
ERR_XLSX   = "pure_model_error_metrics.xlsx"
PAR_XLSX_TABLE10 = "pure_model_parameters_table10.xlsx"
ERR_XLSX_TABLE9  = "pure_model_error_metrics_table9.xlsx"
PLOT_DPI   = 220

              
MIN_PTS = 3
EPS = 1e-12

# ---------------------- Column detection ----------------------
P_COLS    = ["P, psi","Pressure, psi","Pressure (psi)","P (psi)","P,psi","P",
             "Pressure (bar)","P (bar)","Pressure (MPa)","P (MPa)","Pressure (kPa)","P (kPa)"]
T_COLS    = ["T, °C","Temperature (°C)","Temperature (C)","T, C","T (°C)","T"]
Q_COLS    = ["Gas Adsorption, scf/ton","Adsorption, scf/ton","Gas Adsorption (scf/ton)",
             "q (scf/ton)","Adsorption","A","q","A (mmol/g)","Adsorption (mmol/g)","q (mmol/g)"]
COAL_COLS = ["Coal","Coal ID","Coal_ID","Coal Name","Coal sample"]
GT_COLS   = ["Gas Type","GasType","gas type","Gas_Type","Gas_Type Code","Gas type"]
CO2_COLS  = ["Composition CO2%","CO2 (%)","CO2%","Y_CO2, %","CO₂ concentration mol%","CO2, %"]

def load_table(path, sheet=None):
    p = Path(path); ext = p.suffix.lower()
    if ext in [".xls",".xlsx",".xlsm"]:
        try:    return pd.read_excel(p, sheet_name=sheet)
        except: return pd.read_excel(p, sheet_name=sheet, engine="openpyxl")
    return pd.read_csv(p)

def pick_col(df: pd.DataFrame, cands: List[str], name: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    norm = {re.sub(r"\s+","",c).lower(): c for c in df.columns}
    for c in cands:
        key = re.sub(r"\s+","",c).lower()
        if key in norm:
            return norm[key]
    raise KeyError(f"{name} column not found. Candidates tried: {cands}. Available: {list(df.columns)}")

def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.Series):
        return obj
    best = max(obj.columns, key=lambda c: obj[c].notna().sum())
    return obj[best]

# ---------------------- Units ----------------------
_PSI_PER_BAR = 14.5037738
_PSI_PER_MPA = 145.037738
_PSI_PER_KPA = 0.145037738

def to_psia(series: pd.Series, unit: str, is_gauge: bool) -> pd.Series:
    u = str(unit).lower()
    x = pd.to_numeric(series, errors="coerce")
    if u in ("psia","psi"): out = x
    elif u == "psig":       out = x + 14.7
    elif u == "bar":        out = x * _PSI_PER_BAR
    elif u == "mpa":        out = x * _PSI_PER_MPA
    elif u == "kpa":        out = x * _PSI_PER_KPA
    else:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    if u in ("psia","psi") and is_gauge:
        out = out + 14.7
    return out

# ---------------------- Gas normalization ----------------------
def norm_gas(val) -> str:
    if pd.isna(val): return ""
    s = str(val).strip()
    try:
        code = int(float(s))
        if code == 1: return "CH4"
        if code == 2: return "CO2"
    except Exception:
        pass
    s_up = s.upper().replace("₄","4").replace("₂","2").replace(" ", "")
    if s_up.startswith("CO2"): return "CO2"
    if s_up.startswith("CH4"): return "CH4"
    if s_up == "1": return "CH4"
    if s_up == "2": return "CO2"
    return s_up

# ---------------------- Models (DR/DA REMOVED) ----------------------
def langmuir(P, Q0, PL):
    P = np.asarray(P, float)
    return Q0 * P / (PL + P)

def freundlich(P, KF, m):
    P = np.asarray(P, float)
    m = max(float(m), 1.001)
    return KF * np.power(np.maximum(P, EPS), 1.0/m)

def toth(P, Q0, b, t):
    P = np.asarray(P, float)
    b = max(float(b), 1e-12)
    t = max(float(t), 0.05)
    x = np.power(np.clip(b*P, EPS, None), t)
    return Q0 * (b*P) / np.power(1.0 + x, 1.0/t)

                            
def sips(P, Nsm, KLF, n):
    P = np.asarray(P, float)
    Nsm = max(float(Nsm), EPS)
    KLF = max(float(KLF), 1e-12)
    n   = max(float(n), 1e-6)
    x = np.power(np.clip(KLF*P, EPS, None), n)
    return Nsm * x / (1.0 + x)

# ---------------------- Extended Langmuir (EL) mixture prediction (NEW) ----------------------
def extended_langmuir_binary(
    P_total: np.ndarray,
    y_co2: np.ndarray,
    VL_co2: float, b_co2: float,
    VL_ch4: float, b_ch4: float,
    component: str
) -> np.ndarray:
    """
    Binary EL:
      V_i = (VL_i b_i P_i) / (1 + b_co2 P_co2 + b_ch4 P_ch4)
      P_co2 = y_co2 * P
      P_ch4 = (1-y_co2) * P
    Here b_i = 1/PL_i if using Langmuir form q=Q0*P/(PL+P).
    """
    P_total = np.asarray(P_total, float)
    y_co2   = np.asarray(y_co2, float)
    y_co2   = np.clip(y_co2, 0.0, 1.0)
    P_co2 = y_co2 * P_total
    P_ch4 = (1.0 - y_co2) * P_total

    denom = 1.0 + b_co2 * P_co2 + b_ch4 * P_ch4
    denom = np.maximum(denom, EPS)

    comp = str(component).upper()
    if comp.startswith("CO2"):
        num = VL_co2 * b_co2 * P_co2
    else:
        num = VL_ch4 * b_ch4 * P_ch4
    return num / denom

# ---------------------- Metrics ----------------------
def metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan}
    res = y - yhat
    ssr = float(np.sum(res*res))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ssr/sst if sst > 0 else np.nan
    rmse = float(np.sqrt(ssr / max(len(y), 1)))
    mae = float(np.mean(np.abs(res)))
    mse = float(np.mean(res*res))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MSE": mse}

@dataclass
class FitResult:
    model: str
    params: Dict[str, float]
    m: Dict[str, float]

# ---------------------- Fit with bounds (DR/DA REMOVED) ----------------------
def fit_pure_models(P, q) -> List[FitResult]:
    P = np.asarray(P, float)
    q = np.asarray(q, float)

    ok = np.isfinite(P) & np.isfinite(q)
    ok = ok & (P > 0) & (q >= 0)
    P, q = P[ok], q[ok]
    if len(P) < MIN_PTS:
        return []

    qmax_guess = max(float(np.max(q)), EPS)
    P_med = max(float(np.median(P)), 1.0)
    out: List[FitResult] = []

              
    try:
        p0 = (qmax_guess, P_med)
        lb = (qmax_guess*0.01, 1e-6)
        ub = (qmax_guess*50.0, 1e7)
        popt, _ = curve_fit(langmuir, P, q, p0=p0, bounds=(lb, ub), maxfev=250000)
        yhat = langmuir(P, *popt)
        out.append(FitResult("Langmuir", {"Q0": float(popt[0]), "PL": float(popt[1])},
                             metrics(q, yhat)))
    except:
        pass

                
    try:
        KF0 = qmax_guess / max(P.max(), 1.0)**0.5
        p0 = (max(KF0, 1e-9), 2.0)
        lb = (1e-12, 1.01)
        ub = (qmax_guess*200.0, 20.0)
        popt, _ = curve_fit(freundlich, P, q, p0=p0, bounds=(lb, ub), maxfev=250000)
        yhat = freundlich(P, *popt)
        out.append(FitResult("Freundlich", {"KF": float(popt[0]), "m": float(popt[1])},
                             metrics(q, yhat)))
    except:
        pass

          
    try:
        b0 = 1.0 / max(P.max(), 1.0)
        p0 = (qmax_guess, b0, 1.0)
        lb = (qmax_guess*0.01, 1e-12, 0.05)
        ub = (qmax_guess*100.0, 1e3, 10.0)
        popt, _ = curve_fit(toth, P, q, p0=p0, bounds=(lb, ub), maxfev=300000)
        yhat = toth(P, *popt)
        out.append(FitResult("Toth", {"Q0": float(popt[0]), "b": float(popt[1]), "t": float(popt[2])},
                             metrics(q, yhat)))
    except:
        pass

          
    try:
        Nsm0 = qmax_guess
        K0 = 1.0 / max(P_med, 1.0)
        n0 = 0.7
        p0 = (Nsm0, K0, n0)
        lb = (qmax_guess*0.01, 1e-12, 0.05)
        ub = (qmax_guess*200.0, 1e3, 1.50)
        popt, _ = curve_fit(sips, P, q, p0=p0, bounds=(lb, ub), maxfev=400000)
        yhat = sips(P, *popt)
        out.append(FitResult("Sips", {"Nsm": float(popt[0]), "KLF": float(popt[1]), "n": float(popt[2])},
                             metrics(q, yhat)))
    except:
        pass

    return out

# ---------------------- Smoothing for ML curves ----------------------
def smooth_xy(x, y, n=300, method="pchip"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return x, y
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    xu, inv = np.unique(x, return_inverse=True)
    if len(xu) < 2:
        return x, y
    yu = np.zeros_like(xu, dtype=float)
    cnt = np.zeros_like(xu, dtype=float)
    for i, j in enumerate(inv):
        yu[j] += y[i]
        cnt[j] += 1
    yu = yu / np.maximum(cnt, 1)
    xi = np.linspace(float(xu.min()), float(xu.max()), int(max(n, 50)))
    if method.lower() == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator
            yi = PchipInterpolator(xu, yu)(xi)
        except Exception:
            yi = np.interp(xi, xu, yu)
    else:
        yi = np.interp(xi, xu, yu)
    return xi, yi

# ---------------------- Helpers ----------------------
def co2_match_mask(series, target, tol=CO2_MATCH_TOL):
    s = pd.to_numeric(series, errors="coerce")
    mask = np.abs(s - target) <= tol
    if mask.any():
        return mask
    s_r = np.rint(s)
    return np.abs(s_r - target) <= 0.0

def choose_best_T_group(dfd: pd.DataFrame) -> Optional[pd.DataFrame]:
    if dfd is None or dfd.empty:
        return None
    d = dfd.copy()
    d["_T_round"] = d["T_C"].round(1)
    grp = d.groupby("_T_round", dropna=False)["q_exp"].count().sort_values(ascending=False)
    for Tval, cnt in grp.items():
        if cnt >= MIN_PTS:
            return d[d["_T_round"] == Tval].drop(columns=["_T_round"])
    if len(d) >= MIN_PTS:
        return d.drop(columns=["_T_round"])
    return None

def gas_tex(gas: str) -> str:
    return r"CO$_2$" if gas.strip().upper().startswith("CO2") else r"CH$_4$"

# ---------------------- Plot style ----------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 13,
    "axes.labelsize": 15,
    "legend.fontsize": 12.1,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.handlelength": 3.0,
    "legend.borderpad": 0.4,
    "axes.grid": False,
})

def solid_border(ax, width=2.8):
    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(width)
    ax.tick_params(width=width, length=6)

STYLE = {
    "Experimental": dict(
        marker="s", ms=7.8, mfc="#2f6bff",
        mec="black", mew=1.1, lw=0
    ),

    # ---- PHYSICAL MODELS ----
    "Langmuir": dict(ls="--", dashes=(8, 4), lw=2.4, color="#2c7a2c"),
    "Freundlich": dict(ls=":", dashes=(1, 2), lw=2.6, color="#1f77b4"),
    "Toth": dict(ls="-.", dashes=(6, 3, 1.5, 3), lw=2.4, color="#ff7f0e"),
    "Sips": dict(ls="-", lw=2.5, color="#7a1fa2"),

    # ---- NEW: Extended Langmuir ----
    # ---- Extended Langmuir (distinct from solid red "This study") ----
    "EL": dict(ls="--", dashes=(6, 3), lw=2.8, color="black"),


    # ---- ML curves ----
    "SPIVAKOVA": dict(ls="--", dashes=(10, 4), lw=2.8, color="#c1121f"),
    "THIS_STUDY": dict(ls="-", lw=3.0, color="red"),
}

# ---------------------- MAIN ----------------------
df = load_table(EXCEL_IN, SHEET_IN)
df.columns = df.columns.str.strip()

COL_P    = pick_col(df, P_COLS,    "Pressure")
COL_T    = pick_col(df, T_COLS,    "Temperature")
COL_Q    = pick_col(df, Q_COLS,    "Adsorption")
COL_COAL = pick_col(df, COAL_COLS, "Coal")
COL_CO2  = pick_col(df, CO2_COLS,  "CO2%")
COL_GT   = pick_col(df, GT_COLS,   "Gas Type")

df = df.copy()
df["P_psia"]  = to_psia(get_series(df, COL_P), PRESSURE_UNIT_IN, PRESSURE_IS_GAUGE)
df["T_C"]     = pd.to_numeric(get_series(df, COL_T), errors="coerce")
df["q_exp"]   = pd.to_numeric(get_series(df, COL_Q), errors="coerce")
df["CO2_pct"] = pd.to_numeric(get_series(df, COL_CO2), errors="coerce")
df["Gas_norm"]= get_series(df, COL_GT).map(norm_gas)

                
IRINA_XGB_COL = IRINA_XGB_COL_EXACT if IRINA_XGB_COL_EXACT in df.columns else None

if THIS_XGB_COL_FORCE is not None:
    THIS_XGB_COL = THIS_XGB_COL_FORCE if THIS_XGB_COL_FORCE in df.columns else None
else:
    THIS_XGB_COL = None
    for c in THIS_STUDY_XGB_CANDS:
        if c in df.columns:
            THIS_XGB_COL = c
            break
    if THIS_XGB_COL is None:
        for c in df.columns:
            lc = c.lower()
            if ("xgb" in lc or "xgboost" in lc) and ("pred" in lc or "prediction" in lc):
                if IRINA_XGB_COL is not None and c == IRINA_XGB_COL:
                    continue
                THIS_XGB_COL = c
                break

df = df.dropna(subset=["P_psia","q_exp","CO2_pct",COL_COAL,"Gas_norm"]).copy()
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MIXTURE_DIR, exist_ok=True)

param_rows = []
err_rows = []

                                                           
                                                   
pure_lang: Dict[Tuple[str, str], Tuple[float, float]] = {}

# =========================
# A) PURE FITTING LOOP
# =========================
for coal, gas, co2pct in SELECTIONS_PURE:
    d0 = df[df[COL_COAL].astype(str).str.strip().eq(str(coal).strip())].copy()
    d0 = d0[d0["Gas_norm"].str.upper().eq(str(gas).upper())].copy()
    d0 = d0[co2_match_mask(d0["CO2_pct"], co2pct, tol=CO2_MATCH_TOL)].copy()

    dsel = choose_best_T_group(d0)
    if dsel is None or len(dsel) < MIN_PTS:
        print(f"[WARN] Not enough points for PURE: {coal}, {gas}, {co2pct}%")
        continue

    dsel = dsel.sort_values("P_psia")
    P = dsel["P_psia"].to_numpy(float)
    q = dsel["q_exp"].to_numpy(float)

    fits = fit_pure_models(P, q)
    if not fits:
        print(f"[WARN] Fit failed for PURE: {coal}, {gas}, {co2pct}%")
        continue

                                   
    for fr in fits:
        if fr.model == "Langmuir":
            pure_lang[(str(coal).strip(), str(gas).upper().strip())] = (float(fr.params["Q0"]), float(fr.params["PL"]))

    preds_ml_raw = {}
    if IRINA_XGB_COL is not None and dsel[IRINA_XGB_COL].notna().any():
        preds_ml_raw["Spivakova et al., 2025"] = pd.to_numeric(dsel[IRINA_XGB_COL], errors="coerce").to_numpy(float)
    if THIS_XGB_COL is not None and dsel[THIS_XGB_COL].notna().any():
        preds_ml_raw["XGBoost predicted (This study)"] = pd.to_numeric(dsel[THIS_XGB_COL], errors="coerce").to_numpy(float)

    T_K = int(round((dsel["T_C"].iloc[0] + 273.15))) if dsel["T_C"].notna().any() else None
    gas_label = "CO2" if str(gas).upper().startswith("CO2") else "CH4"

                                
    for fr in fits:
        pr = {
            "Coal sample": coal, "Gas type": gas_label, "Mixture": "pure",
            "Temperature": f"T = {T_K} K" if T_K is not None else "",
            "Model": fr.model
        }
        for k, v in fr.params.items():
            pr[k] = v
        param_rows.append(pr)

        er = {
            "Coal sample": coal, "Gas type": gas_label, "Mixture": "pure",
            "Temperature": f"T = {T_K} K" if T_K is not None else "",
            "Model": fr.model,
            "Model_R2": fr.m["R2"],
            "Model_RMSE": fr.m["RMSE"],
            "Model_MAE": fr.m["MAE"],
            "Model_MSE": fr.m["MSE"],
        }

        if "Spivakova et al., 2025" in preds_ml_raw:
            msp = metrics(q, preds_ml_raw["Spivakova et al., 2025"])
            er["Spivakova_R2"] = msp["R2"]
            er["Spivakova_RMSE"] = msp["RMSE"]
            er["Spivakova_MAE"] = msp["MAE"]
            er["Spivakova_MSE"] = msp["MSE"]

        if "XGBoost predicted (This study)" in preds_ml_raw:
            mth = metrics(q, preds_ml_raw["XGBoost predicted (This study)"])
            er["ThisStudy_R2"] = mth["R2"]
            er["ThisStudy_RMSE"] = mth["RMSE"]
            er["ThisStudy_MAE"] = mth["MAE"]
            er["ThisStudy_MSE"] = mth["MSE"]

        err_rows.append(er)

    # ---------------- PURE Plot ----------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.grid(False)
    solid_border(ax, width=2.8)

    Pmax = float(np.max(P))
    Pplt = np.linspace(0.0, Pmax, 450)

    for fr in fits:
        if fr.model == "Langmuir":
            y = langmuir(Pplt, fr.params["Q0"], fr.params["PL"])
        elif fr.model == "Freundlich":
            y = freundlich(Pplt, fr.params["KF"], fr.params["m"])
        elif fr.model == "Toth":
            y = toth(Pplt, fr.params["Q0"], fr.params["b"], fr.params["t"])
        else:        
            y = sips(Pplt, fr.params["Nsm"], fr.params["KLF"], fr.params["n"])

        y = np.clip(y, 0.0, None)
        ax.plot(Pplt, y, zorder=2, **STYLE[fr.model], label=fr.model)

               
    if "Spivakova et al., 2025" in preds_ml_raw:
        yml = preds_ml_raw["Spivakova et al., 2025"]
        xml = P.copy()
        if SMOOTH_ML and len(xml) >= 2:
            xs, ys = smooth_xy(xml, yml, n=SMOOTH_N, method=SMOOTH_METHOD)
            ax.plot(xs, np.clip(ys, 0.0, None), zorder=3, **STYLE["SPIVAKOVA"], label="Spivakova et al., 2025")
        else:
            ax.plot(xml, yml, zorder=3, **STYLE["SPIVAKOVA"], label="Spivakova et al., 2025")

    if "XGBoost predicted (This study)" in preds_ml_raw:
        yml = preds_ml_raw["XGBoost predicted (This study)"]
        xml = P.copy()
        if SMOOTH_ML and len(xml) >= 2:
            xs, ys = smooth_xy(xml, yml, n=SMOOTH_N, method=SMOOTH_METHOD)
            ax.plot(xs, np.clip(ys, 0.0, None), zorder=3, **STYLE["THIS_STUDY"], label="XGBoost predicted (This study)")
        else:
            ax.plot(xml, yml, zorder=3, **STYLE["THIS_STUDY"], label="XGBoost predicted (This study)")

    ax.plot(P, q, linestyle="None", zorder=6, **STYLE["Experimental"], label="Experimental Data")

    ax.set_xlabel("Pressure (psi)", fontweight="bold")
    ax.set_ylabel(f"{gas_tex(gas_label)} adsorption amount ({ADS_UNIT_OUT})", fontweight="bold")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    desired = [
        "Experimental Data",
        "Langmuir","Freundlich","Toth","Sips",
        "Spivakova et al., 2025",
        "XGBoost predicted (This study)"
    ]
    h, l = ax.get_legend_handles_labels()
    order = []
    for lbl in desired:
        order.extend(i for i, lab in enumerate(l) if lab == lbl)
    H = [h[i] for i in order if i < len(h)]
    L = [l[i] for i in order if i < len(l)]
    ax.legend(H, L, loc="best", frameon=False, borderaxespad=0.7)

    fig.tight_layout()
    fn = f"{str(coal).replace(' ','_')}_{gas_label}_pure.png"
    fig.savefig(Path(PLOT_DIR, fn), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

                          
par_df = pd.DataFrame(param_rows)
err_df = pd.DataFrame(err_rows)

if par_df.empty:
    par_df = pd.DataFrame({"note": ["no pure cases were fitted"]})
if err_df.empty:
    err_df = pd.DataFrame({"note": ["no pure cases were fitted"]})

par_df.to_excel(PAR_XLSX, index=False)
err_df.to_excel(ERR_XLSX, index=False)

print("PURE Done.")
print(f"- Plots: {PLOT_DIR}/")
print(f"- Parameters: {PAR_XLSX}")
print(f"- Error metrics: {ERR_XLSX}")
print(f"Detected ML cols: Spivakova={IRINA_XGB_COL}, ThisStudy={THIS_XGB_COL}")

# =========================
                                                          
# =========================
mix_err_rows = []

def _get_pure_lang(coal_name: str) -> Optional[Tuple[Tuple[float,float], Tuple[float,float]]]:
    """Return (CO2(Q0,PL), CH4(Q0,PL)) for the coal if both exist."""
    k_co2 = (str(coal_name).strip(), "CO2")
    k_ch4 = (str(coal_name).strip(), "CH4")
    if k_co2 not in pure_lang or k_ch4 not in pure_lang:
        return None
    return pure_lang[k_co2], pure_lang[k_ch4]

for coal, co2pct in MIXTURE_CASES:
    pure_pair = _get_pure_lang(coal)
    if pure_pair is None:
        print(f"[WARN] Missing PURE Langmuir for EL mixture: need {coal} CO2(100%) and CH4(0%). Skipping mixtures for this coal.")
        continue

    (VL_co2, PL_co2), (VL_ch4, PL_ch4) = pure_pair
    b_co2 = 1.0 / max(PL_co2, EPS)
    b_ch4 = 1.0 / max(PL_ch4, EPS)

    for gas in MIXTURE_PLOT_GASES:
        d0 = df[df[COL_COAL].astype(str).str.strip().eq(str(coal).strip())].copy()
        d0 = d0[d0["Gas_norm"].str.upper().eq(str(gas).upper())].copy()
        d0 = d0[co2_match_mask(d0["CO2_pct"], co2pct, tol=MIXTURE_CO2_TOL)].copy()

        dsel = choose_best_T_group(d0)
        if dsel is None or len(dsel) < MIN_PTS:
            print(f"[WARN] Not enough points for MIXTURE: {coal}, {gas}, {co2pct}%")
            continue

        dsel = dsel.sort_values("P_psia")
        P = dsel["P_psia"].to_numpy(float)
        q = dsel["q_exp"].to_numpy(float)

        y_co2 = (pd.to_numeric(dsel["CO2_pct"], errors="coerce").to_numpy(float)) / 100.0
        y_co2 = np.clip(y_co2, 0.0, 1.0)

                                                            
        q_el_pts = extended_langmuir_binary(
            P_total=P,
            y_co2=y_co2,
            VL_co2=VL_co2, b_co2=b_co2,
            VL_ch4=VL_ch4, b_ch4=b_ch4,
            component=gas
        )
        mel = metrics(q, q_el_pts)

                       
        preds_ml_raw = {}
        if IRINA_XGB_COL is not None and dsel[IRINA_XGB_COL].notna().any():
            preds_ml_raw["Spivakova et al., 2025"] = pd.to_numeric(dsel[IRINA_XGB_COL], errors="coerce").to_numpy(float)
        if THIS_XGB_COL is not None and dsel[THIS_XGB_COL].notna().any():
            preds_ml_raw["XGBoost predicted (This study)"] = pd.to_numeric(dsel[THIS_XGB_COL], errors="coerce").to_numpy(float)

                                
        gas_label = "CO2" if str(gas).upper().startswith("CO2") else "CH4"
        mix_err_rows.append({
            "Coal sample": coal,
            "Gas type": gas_label,
            "CO2%": float(co2pct),
            "Model": "Extended Langmuir",
            "EL_R2": mel["R2"],
            "EL_RMSE": mel["RMSE"],
            "EL_MAE": mel["MAE"],
            "EL_MSE": mel["MSE"],
        })

        # ---------------- MIXTURE Plot ----------------
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.grid(False)
        solid_border(ax, width=2.8)

        Pmax = float(np.max(P))
        Pplt = np.linspace(0.0, Pmax, 450)

                                                                                         
        y_co2_const = float(np.nanmedian(y_co2)) if np.isfinite(np.nanmedian(y_co2)) else float(co2pct)/100.0
        y_co2_line = np.full_like(Pplt, y_co2_const, dtype=float)

        q_el_line = extended_langmuir_binary(
            P_total=Pplt,
            y_co2=y_co2_line,
            VL_co2=VL_co2, b_co2=b_co2,
            VL_ch4=VL_ch4, b_ch4=b_ch4,
            component=gas
        )
        ax.plot(Pplt, np.clip(q_el_line, 0.0, None), zorder=2, **STYLE["EL"], label="Extended Langmuir")

                                             
        if "Spivakova et al., 2025" in preds_ml_raw:
            yml = preds_ml_raw["Spivakova et al., 2025"]
            xml = P.copy()
            if SMOOTH_ML and len(xml) >= 2:
                xs, ys = smooth_xy(xml, yml, n=SMOOTH_N, method=SMOOTH_METHOD)
                ax.plot(xs, np.clip(ys, 0.0, None), zorder=3, **STYLE["SPIVAKOVA"], label="Spivakova et al., 2025")
            else:
                ax.plot(xml, yml, zorder=3, **STYLE["SPIVAKOVA"], label="Spivakova et al., 2025")

        if "XGBoost predicted (This study)" in preds_ml_raw:
            yml = preds_ml_raw["XGBoost predicted (This study)"]
            xml = P.copy()
            if SMOOTH_ML and len(xml) >= 2:
                xs, ys = smooth_xy(xml, yml, n=SMOOTH_N, method=SMOOTH_METHOD)
                ax.plot(xs, np.clip(ys, 0.0, None), zorder=3, **STYLE["THIS_STUDY"], label="XGBoost predicted (This study)")
            else:
                ax.plot(xml, yml, zorder=3, **STYLE["THIS_STUDY"], label="XGBoost predicted (This study)")

        ax.plot(P, q, linestyle="None", zorder=6, **STYLE["Experimental"], label="Experimental Data")

        ax.set_xlabel("Pressure (psi)", fontweight="bold")
        ax.set_ylabel(f"{gas_tex(gas_label)} adsorption amount ({ADS_UNIT_OUT})", fontweight="bold")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)

        desired = [
            "Experimental Data",
            "Extended Langmuir",
            "Spivakova et al., 2025",
            "XGBoost predicted (This study)",
        ]
        h, l = ax.get_legend_handles_labels()
        order = []
        for lbl in desired:
            order.extend(i for i, lab in enumerate(l) if lab == lbl)
        H = [h[i] for i in order if i < len(h)]
        L = [l[i] for i in order if i < len(l)]
        ax.legend(H, L, loc="best", frameon=False, borderaxespad=0.7)

        fig.tight_layout()
        fn = f"{str(coal).replace(' ','_')}_{gas_label}_mix_CO2{int(round(co2pct))}.png"
        fig.savefig(Path(MIXTURE_DIR, fn), dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

                         
mix_err_df = pd.DataFrame(mix_err_rows)
if mix_err_df.empty:
    mix_err_df = pd.DataFrame({"note": ["no mixture cases were plotted (missing pure Langmuir fits or insufficient points)"]})
mix_err_df.to_excel(MIXTURE_ERR_XLSX, index=False)

print("MIXTURE Done.")
print(f"- Mixture plots: {MIXTURE_DIR}/")
print(f"- Mixture EL metrics: {MIXTURE_ERR_XLSX}")

# ---------------------- WRITE "TABLE 10" STYLE PARAMETERS (MERGED HEADERS) ----------------------
if not par_df.empty and "note" not in par_df.columns:

    def _get(row, key):
        return row[key] if key in row and pd.notna(row[key]) else np.nan

    keys = ["Coal sample","Gas type","Mixture"]
    base = par_df[keys].drop_duplicates().copy()

    lookup = {}
    for _, r in par_df.iterrows():
        k = (r["Coal sample"], r["Gas type"], r["Mixture"], r["Model"])
        lookup[k] = r.to_dict()

    rows = []
    for _, b in base.iterrows():
        coal = b["Coal sample"]; gas = b["Gas type"]; mix = b["Mixture"]
        rL = lookup.get((coal, gas, mix, "Langmuir"), {})
        rF = lookup.get((coal, gas, mix, "Freundlich"), {})
        rT = lookup.get((coal, gas, mix, "Toth"), {})
        rS = lookup.get((coal, gas, mix, "Sips"), {})

        rows.append({
            ("", "Coal sample"): coal,
            ("", "Gas type"): gas,
            ("", "Mixture"): mix,

            ("Langmuir", "Q0 (scf/ton)"): _get(rL, "Q0"),
            ("Langmuir", "PL (psia)"):    _get(rL, "PL"),

            ("Freundlich", "KF (scf/ton*Psi^-1/m)"): _get(rF, "KF"),
            ("Freundlich", "m (-)"):                 _get(rF, "m"),

            ("Toth", "Q0 (scf/ton)"): _get(rT, "Q0"),
            ("Toth", "b (Psi^-1)"):   _get(rT, "b"),
            ("Toth", "t (-)"):        _get(rT, "t"),

            ("Sips", "Nsm (scf/ton)"): _get(rS, "Nsm"),
            ("Sips", "KLF (Psi^-1)"):  _get(rS, "KLF"),
            ("Sips", "n (-)"):         _get(rS, "n"),
        })

    table10_df = pd.DataFrame(rows)

    import xlsxwriter
    with pd.ExcelWriter(PAR_XLSX_TABLE10, engine="xlsxwriter") as writer:
        sheet = "Table10"
        table10_df.to_excel(writer, sheet_name=sheet, index=False, startrow=2)

        wb  = writer.book
        ws  = writer.sheets[sheet]

        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        ws.write(0, 0, "Table 10. Model parameters correspond to each of the adsorption isotherm models used in this study", title_fmt)

        hdr_top = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        hdr_sub = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        cell_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})
        left_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1, "bold": True})

        cols = list(table10_df.columns)
        ncol = len(cols)

        top_row = 2
        sub_row = 3

        for j in range(3):
            ws.merge_range(top_row, j, sub_row, j, cols[j][1], left_fmt)

        j = 3
        while j < ncol:
            group = cols[j][0]
            start = j
            while j < ncol and cols[j][0] == group:
                j += 1
            end = j - 1
            ws.merge_range(top_row, start, top_row, end, group, hdr_top)
            for k in range(start, end + 1):
                ws.write(sub_row, k, cols[k][1], hdr_sub)

        nrows = len(table10_df)
        for r in range(4, 4 + nrows):
            for c in range(ncol):
                ws.write(r, c, table10_df.iloc[r-4, c], cell_fmt)

        ws.set_column(0, 0, 16)
        ws.set_column(1, 1, 10)
        ws.set_column(2, 2, 10)
        ws.set_column(3, ncol-1, 18)

        ws.set_row(top_row, 24)
        ws.set_row(sub_row, 24)

else:
    pd.DataFrame({"note":["no pure cases were fitted"]}).to_excel(PAR_XLSX_TABLE10, index=False)

print(f"Saved Table-10 style parameter sheet -> {PAR_XLSX_TABLE10}")

# ---------------------- WRITE "TABLE 9" STYLE ERROR METRICS (NO DR/DA) ----------------------
pure_err_df = err_df
if not pure_err_df.empty and "note" not in pure_err_df.columns:

    df9 = pure_err_df.copy()
    df9["Gas Composition"] = df9["Mixture"].astype(str)

    key_cols = ["Coal sample","Gas type","Gas Composition"]

    order_metrics = ["MAE","MSE","R²"]
    rows = []
    for (coal, gas, comp), g in df9.groupby(key_cols, dropna=False):
        model_row = {r["Model"]: r for _, r in g.iterrows()}

        for em in order_metrics:
            def grab(model, metric):
                r = model_row.get(model, None)
                if r is None: return np.nan
                if metric == "R²":   return r.get("Model_R2", np.nan)
                if metric == "MAE":  return r.get("Model_MAE", np.nan)
                if metric == "MSE":  return r.get("Model_MSE", np.nan)
                return np.nan

            r0 = g.iloc[0].to_dict() if len(g) else {}

            def grab_ml(prefix, metric):
                if metric == "R²":  return r0.get(f"{prefix}_R2", np.nan)
                if metric == "MAE": return r0.get(f"{prefix}_MAE", np.nan)
                if metric == "MSE": return r0.get(f"{prefix}_MSE", np.nan)
                return np.nan

            rows.append({
                "Coal sample": coal,
                "Gas type": gas,
                "Gas Composition": comp,
                "Error metrics": em,

                "Langmuir": grab("Langmuir", em),
                "Freundlich": grab("Freundlich", em),
                "Toth": grab("Toth", em),
                "Sips": grab("Sips", em),

                "Previous study": grab_ml("Spivakova", em),
                "XGBoost predicted (This study)": grab_ml("ThisStudy", em),
            })

    out9 = pd.DataFrame(rows)
    out9["Error metrics"] = pd.Categorical(out9["Error metrics"], categories=order_metrics, ordered=True)
    out9 = out9.sort_values(["Coal sample","Gas type","Gas Composition","Error metrics"]).reset_index(drop=True)

    import xlsxwriter
    with pd.ExcelWriter(ERR_XLSX_TABLE9, engine="xlsxwriter") as writer:
        sheet = "Table9"
        out9.to_excel(writer, sheet_name=sheet, index=False, startrow=2)

        wb = writer.book
        ws = writer.sheets[sheet]

        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        ws.write(0, 0, "Table 9. Performance metrics for all models across various coal samples", title_fmt)

        hdr_fmt  = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        cell_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})
        left_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})

        header_row = 2
        for j, col in enumerate(out9.columns):
            ws.write(header_row, j, col, hdr_fmt)

        nrows, ncols = out9.shape
        start_data = 3
        for r in range(nrows):
            for c in range(ncols):
                v = out9.iat[r, c]
                if isinstance(v, (float, np.floating)) and np.isfinite(v):
                    ws.write_number(start_data + r, c, float(v), cell_fmt)
                else:
                    ws.write(start_data + r, c, "" if pd.isna(v) else v, cell_fmt)

        i = 0
        while i < nrows:
            coalv = out9.iat[i, 0]
            gasv  = out9.iat[i, 1]
            compv = out9.iat[i, 2]
            j = i
            while j < nrows and out9.iat[j, 0] == coalv and out9.iat[j, 1] == gasv and out9.iat[j, 2] == compv:
                j += 1
            r0 = start_data + i
            r1 = start_data + (j - 1)
            if r1 > r0:
                ws.merge_range(r0, 0, r1, 0, coalv, left_fmt)
                ws.merge_range(r0, 1, r1, 1, gasv,  left_fmt)
                ws.merge_range(r0, 2, r1, 2, compv, left_fmt)
            i = j

        ws.set_column(0, 0, 14)
        ws.set_column(1, 1, 10)
        ws.set_column(2, 2, 14)
        ws.set_column(3, 3, 12)
        ws.set_column(4, ncols-1, 20)

        ws.set_row(2, 22)
        ws.set_row(0, 22)

else:
    pd.DataFrame({"note":["no PURE error metrics (no pure cases matched)"]}).to_excel(ERR_XLSX_TABLE9, index=False)

print(f"Saved Table-9 style error sheet -> {ERR_XLSX_TABLE9}")

# ============================================================
# ADDITION: MIXTURE EL METRICS TABLE (SAME STYLE AS TABLE 8)
                                                         
# ============================================================

MIXTURE_ERR_XLSX_TABLE8_STYLE = "mixture_el_error_metrics_table_like_table8.xlsx"

# ---- 1) Build a "Table-8 style" dataframe from the mixture rows ----
                                                                                           
                                                                                     

order_metrics = ["MAE", "MSE", "R²"]

                                               
ONLY_EL_COLUMN = False

rows_out = []

for coal, co2pct in MIXTURE_CASES:
    pure_pair = _get_pure_lang(coal)
    if pure_pair is None:
        continue

    (VL_co2, PL_co2), (VL_ch4, PL_ch4) = pure_pair
    b_co2 = 1.0 / max(PL_co2, EPS)
    b_ch4 = 1.0 / max(PL_ch4, EPS)

    for gas in MIXTURE_PLOT_GASES:
        d0 = df[df[COL_COAL].astype(str).str.strip().eq(str(coal).strip())].copy()
        d0 = d0[d0["Gas_norm"].str.upper().eq(str(gas).upper())].copy()
        d0 = d0[co2_match_mask(d0["CO2_pct"], co2pct, tol=MIXTURE_CO2_TOL)].copy()

        dsel = choose_best_T_group(d0)
        if dsel is None or len(dsel) < MIN_PTS:
            continue

        dsel = dsel.sort_values("P_psia")
        P = dsel["P_psia"].to_numpy(float)
        q = dsel["q_exp"].to_numpy(float)

        y_co2 = (pd.to_numeric(dsel["CO2_pct"], errors="coerce").to_numpy(float)) / 100.0
        y_co2 = np.clip(y_co2, 0.0, 1.0)

                                                
        q_el_pts = extended_langmuir_binary(
            P_total=P,
            y_co2=y_co2,
            VL_co2=VL_co2, b_co2=b_co2,
            VL_ch4=VL_ch4, b_ch4=b_ch4,
            component=gas
        )
        mel = metrics(q, q_el_pts)

                                                
        preds_ml_raw = {}
        if IRINA_XGB_COL is not None and dsel[IRINA_XGB_COL].notna().any():
            preds_ml_raw["Previous study"] = pd.to_numeric(dsel[IRINA_XGB_COL], errors="coerce").to_numpy(float)
        if THIS_XGB_COL is not None and dsel[THIS_XGB_COL].notna().any():
            preds_ml_raw["XGBoost predicted (This study)"] = pd.to_numeric(dsel[THIS_XGB_COL], errors="coerce").to_numpy(float)

                                         
        m_prev = metrics(q, preds_ml_raw["Previous study"]) if "Previous study" in preds_ml_raw else None
        m_this = metrics(q, preds_ml_raw["XGBoost predicted (This study)"]) if "XGBoost predicted (This study)" in preds_ml_raw else None

        gas_label = "CO2" if str(gas).upper().startswith("CO2") else "CH4"
        comp_label = f"{int(round(co2pct))}%"

        for em in order_metrics:
            def pick_m(mdict, em):
                if mdict is None: return np.nan
                if em == "R²": return mdict["R2"]
                if em == "MAE": return mdict["MAE"]
                if em == "MSE": return mdict["MSE"]
                return np.nan

            row = {
                "Coal sample": coal,
                "Gas type": gas_label,
                "CO2 concentration (%)": comp_label,
                "Error metrics": em,
                "Extended Langmuir": pick_m(mel, em),
            }

            if not ONLY_EL_COLUMN:
                row["Previous study"] = pick_m(m_prev, em)
                row["XGBoost predicted (This study)"] = pick_m(m_this, em)

            rows_out.append(row)

out_mix8 = pd.DataFrame(rows_out)

if out_mix8.empty:
    pd.DataFrame({"note": ["no mixture rows found for EL table"]}).to_excel(MIXTURE_ERR_XLSX_TABLE8_STYLE, index=False)
    print(f"Saved -> {MIXTURE_ERR_XLSX_TABLE8_STYLE} (empty)")
else:
    out_mix8["Error metrics"] = pd.Categorical(out_mix8["Error metrics"], categories=order_metrics, ordered=True)
    out_mix8 = out_mix8.sort_values(["Coal sample","Gas type","CO2 concentration (%)","Error metrics"]).reset_index(drop=True)

    # ---- 2) Write with same "merge rows by group" formatting as your Table 9 writer ----
    import xlsxwriter
    with pd.ExcelWriter(MIXTURE_ERR_XLSX_TABLE8_STYLE, engine="xlsxwriter") as writer:
        sheet = "Table_EL"
        out_mix8.to_excel(writer, sheet_name=sheet, index=False, startrow=2)

        wb = writer.book
        ws = writer.sheets[sheet]

        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        ws.write(0, 0, "Mixture performance metrics (Extended Langmuir)", title_fmt)

        hdr_fmt  = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        cell_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})
        left_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})

        header_row = 2
        for j, col in enumerate(out_mix8.columns):
            ws.write(header_row, j, col, hdr_fmt)

        nrows, ncols = out_mix8.shape
        start_data = 3
        for r in range(nrows):
            for c in range(ncols):
                v = out_mix8.iat[r, c]
                if isinstance(v, (float, np.floating)) and np.isfinite(v):
                    ws.write_number(start_data + r, c, float(v), cell_fmt)
                else:
                    ws.write(start_data + r, c, "" if pd.isna(v) else v, cell_fmt)

                                                                                     
        i = 0
        while i < nrows:
            coalv = out_mix8.iat[i, 0]
            gasv  = out_mix8.iat[i, 1]
            compv = out_mix8.iat[i, 2]
            j = i
            while j < nrows and out_mix8.iat[j, 0] == coalv and out_mix8.iat[j, 1] == gasv and out_mix8.iat[j, 2] == compv:
                j += 1
            r0 = start_data + i
            r1 = start_data + (j - 1)
            if r1 > r0:
                ws.merge_range(r0, 0, r1, 0, coalv, left_fmt)
                ws.merge_range(r0, 1, r1, 1, gasv,  left_fmt)
                ws.merge_range(r0, 2, r1, 2, compv, left_fmt)
            i = j

                                                  
        ws.set_column(0, 0, 14)                
        ws.set_column(1, 1, 10)             
        ws.set_column(2, 2, 22)                      
        ws.set_column(3, 3, 12)                  
        ws.set_column(4, ncols-1, 26)

        ws.set_row(0, 22)
        ws.set_row(2, 22)

    print(f"Saved mixture EL table (Table-8 style) -> {MIXTURE_ERR_XLSX_TABLE8_STYLE}")
# ============================================================
                                                               
                                                       
# ============================================================

EL_PAR_XLSX_TABLE10_STYLE = "mixture_el_parameters_table_like_table10.xlsx"

                                                          
el_rows = []
for coal, _ in MIXTURE_CASES:
    k_co2 = (str(coal).strip(), "CO2")
    k_ch4 = (str(coal).strip(), "CH4")

    if k_co2 not in pure_lang or k_ch4 not in pure_lang:
        print(f"[WARN] EL param table: missing pure Langmuir for {coal} (need CO2 & CH4). Skipping.")
        continue

    (VL_co2, PL_co2) = pure_lang[k_co2]
    (VL_ch4, PL_ch4) = pure_lang[k_ch4]

    el_rows.append({
        ("", "Coal sample"): str(coal).strip(),

        ("CO2", "VL (Q0) (scf/ton)"): float(VL_co2),
        ("CO2", "PL (psia)"):        float(PL_co2),
        ("CO2", "b = 1/PL (psia⁻¹)"): float(1.0 / max(PL_co2, EPS)),

        ("CH4", "VL (Q0) (scf/ton)"): float(VL_ch4),
        ("CH4", "PL (psia)"):         float(PL_ch4),
        ("CH4", "b = 1/PL (psia⁻¹)"):  float(1.0 / max(PL_ch4, EPS)),
    })

el_table = pd.DataFrame(el_rows)

if el_table.empty:
    pd.DataFrame({"note": ["no EL parameters available (missing pure Langmuir fits)"]}).to_excel(
        EL_PAR_XLSX_TABLE10_STYLE, index=False
    )
    print(f"Saved -> {EL_PAR_XLSX_TABLE10_STYLE} (empty)")
else:
    import xlsxwriter
    with pd.ExcelWriter(EL_PAR_XLSX_TABLE10_STYLE, engine="xlsxwriter") as writer:
        sheet = "Table_EL_Params"
        el_table.to_excel(writer, sheet_name=sheet, index=False, startrow=2)

        wb = writer.book
        ws = writer.sheets[sheet]

        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        ws.write(0, 0, "EL input parameters (from pure Langmuir fits)", title_fmt)

        hdr_top = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        hdr_sub = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        cell_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})
        left_fmt = wb.add_format({"align": "center", "valign": "vcenter", "border": 1, "bold": True})

        cols = list(el_table.columns)
        ncol = len(cols)

        top_row = 2
        sub_row = 3

                                                      
        ws.merge_range(top_row, 0, sub_row, 0, cols[0][1], left_fmt)

                                  
        j = 1
        while j < ncol:
            group = cols[j][0]
            start = j
            while j < ncol and cols[j][0] == group:
                j += 1
            end = j - 1
            ws.merge_range(top_row, start, top_row, end, group, hdr_top)
            for k in range(start, end + 1):
                ws.write(sub_row, k, cols[k][1], hdr_sub)

                                 
        nrows = len(el_table)
        for r in range(4, 4 + nrows):
            for c in range(ncol):
                v = el_table.iloc[r-4, c]
                if isinstance(v, (float, np.floating)) and np.isfinite(v):
                    ws.write_number(r, c, float(v), cell_fmt)
                else:
                    ws.write(r, c, "" if pd.isna(v) else v, cell_fmt)

        ws.set_column(0, 0, 14)
        ws.set_column(1, ncol-1, 20)
        ws.set_row(top_row, 24)
        ws.set_row(sub_row, 24)

    print(f"Saved EL parameter table (Table-10 style) -> {EL_PAR_XLSX_TABLE10_STYLE}")
