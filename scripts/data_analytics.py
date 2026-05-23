# ============================ Imports ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, gaussian_kde
from sklearn.model_selection import train_test_split
from math import ceil

# ============================ Config ===========================================
EXCEL_IN   = 'Dataset.xlsx'
SHEET_NAME = 'Data'   

TARGET = 'Gas Adsorption, scf/ton'

                                                           
CO2_COL_CANDIDATES  = ['CO2 concentration mol%', 'Composition CO2%', 'CO₂ concentration mol%']
TEMP_COL_CANDIDATES = [
    'Temperature, °C', 'T, °C', 'Temperature (°C)', 'Temp, °C',
    'Temperature, C', 'T, C', 'Temperature (C)', 'Temp, C'
]

BASE_FEATURES = [
    'P, psi',
    'Ash content, wt.%',
    'Fuel_Ratio',
                                                          
]

# ============================ Small helpers ====================================
def find_first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
                                                                    
    return candidates[0]

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def make_strat_bins(y: pd.Series, max_bins: int = 8, min_count: int = 2) -> pd.Series:
    y = pd.to_numeric(y, errors='coerce').dropna()
    idx = y.index
    for q in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=q, duplicates="drop")
        except Exception:
            continue
        if bins.value_counts().min() >= min_count:
            z = pd.Series(index=idx, data=bins.astype(str))
            return z.reindex(y.index)
    return pd.Series(["all"] * len(y), index=idx)

# ============================ Load data ========================================
Total = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
Total.columns = [c.strip() for c in Total.columns]

CO2_COL  = find_first_present(Total.columns, CO2_COL_CANDIDATES)
TEMP_COL = find_first_present(Total.columns, TEMP_COL_CANDIDATES)

FEATURES = BASE_FEATURES.copy()
FEATURES.append(TEMP_COL)                        
FEATURES.append(CO2_COL)                 

needed = FEATURES + [TARGET]
missing = [c for c in needed if c not in Total.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

Total = coerce_numeric(Total, needed)

# ============================ Pretty label mapping =============================
                                                                   
pretty = {
    'P, psi'            : 'Pressure, psi',
    TEMP_COL            : 'Temperature, °C',
    'Ash content, wt.%' : 'Ash, wt.%',
    'Fuel_Ratio'        : 'Fuel Ratio',
    CO2_COL             : r'CO$_2$ concentration (mol%)',
    TARGET              : 'Gas Adsorption, scf/ton',
}

                                         
ORDER_FOR_ALL = [
    'P, psi',
    TEMP_COL,
    'Ash content, wt.%',
    'Fuel_Ratio',
    CO2_COL,
    TARGET,
]

# ============================ Stratified split (for KDE grid only) =============
X = Total[FEATURES].copy()
y = Total[TARGET].copy()

y_bins = make_strat_bins(y, max_bins=8, min_count=2).reindex(y.index).fillna("all")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y_bins
)

# ============================ Paper-style stats table (ALL DATA) ===============
def paper_stats_table(df: pd.DataFrame, order, label_map) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    desc = num.describe().T                                             
    variance = num.var(ddof=1)

    out = pd.DataFrame({
        'count'              : desc['count'].astype(int),
        'mean'               : desc['mean'],
        'standard deviation' : desc['std'],
        'variance'           : variance,
        'kurtosis'           : num.apply(lambda s: kurtosis(pd.to_numeric(s, errors='coerce').dropna(), bias=False)),
        'skewness'           : num.apply(lambda s: skew(pd.to_numeric(s, errors='coerce').dropna(), bias=False)),
        'range'              : desc['max'] - desc['min'],
        'lower bound'        : desc['min'],
        'upper bound'        : desc['max'],
        '25%'                : desc['25%'],
        '50% (Median)'       : desc['50%'],
        '75%'                : desc['75%'],
    })

    existing_in_order = [c for c in order if c in out.index]
    out = out.loc[existing_in_order].round(3)
    out.index = [label_map.get(c, c) for c in out.index]
    return out

                                                           
all_df_for_stats = Total[[c for c in ORDER_FOR_ALL if c in Total.columns]].copy()
stats_tbl = paper_stats_table(all_df_for_stats, ORDER_FOR_ALL, pretty)
stats_tbl.to_excel("Paper_Stats_Table.xlsx")
print("Saved: Paper_Stats_Table.xlsx (computed on ALL DATA)")
print(stats_tbl)

# ============================ Correlation heatmap ==============================
def correlation_heatmap(
    df: pd.DataFrame, order, label_map,
    outfile="Correlation_Heatmap.png",
    title="Pearson correlation",
    figsize=(8.0, 6.2),                            
    annot_size=13,                                          
    tick_fs=13,                                          
    title_fs=20,                               
    cbar_label_fs=13
):
    sns.set_theme(style="white")
    plt.figure(figsize=figsize)
    corr = df[[c for c in order if c in df.columns]].corr(numeric_only=True)

    ax = sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap=plt.cm.RdBu_r, vmin=-1.0, vmax=1.0,
        square=True, linewidths=0.6, linecolor="white",
        cbar=True, cbar_kws={"shrink": 0.9, "label": "Correlation coefficient"},
        annot_kws={"size": annot_size, "weight": "bold"}
    )
    ax.set_title(title, fontsize=title_fs, weight="bold", pad=10)

                        
    ax.set_xticklabels(
        [label_map.get(c.get_text(), c.get_text()) for c in ax.get_xticklabels()],
        rotation=30, ha="right", fontsize=tick_fs, fontweight="bold"
    )
    ax.set_yticklabels(
        [label_map.get(c.get_text(), c.get_text()) for c in ax.get_yticklabels()],
        rotation=30, va="center", fontsize=tick_fs, fontweight="bold"
    )

                        
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.set_label("Correlation coefficient", size=cbar_label_fs, weight="bold")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()


# ============================ KDE helpers & flexible grid ======================
TRAIN_BAR  = "#2563EB"                 
TEST_BAR   = "#F97316"                  
KDE_TRAIN  = "#1F2937"                                      
KDE_TEST   = "#7C2D12"                                  

def _scaled_kde(yvals: pd.Series, bins: np.ndarray):
    y = pd.to_numeric(yvals, errors="coerce").dropna().astype(float)
    if y.size < 3:
        xg = np.linspace(bins.min(), bins.max(), 256)
        return xg, np.zeros_like(xg)
    kde = gaussian_kde(y)
    xg = np.linspace(bins.min(), bins.max(), 512)
    density = kde(xg)
    binw = np.diff(bins).mean()
    return xg, density * (y.size * binw)

def _nice_axes(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(1.1); s.set_color("#111")
    ax.tick_params(labelsize=10)
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")

def plot_train_test_grid_kde_colored(
    Xtr: pd.DataFrame, Xte: pd.DataFrame, ytr: pd.Series, yte: pd.Series,
    cols_in_order: list, label_map: dict, outfile: str
):
    """
    Flexible layout: uses 3 columns; rows = ceil(n/3)
    The LAST entry is assumed to be TARGET.
    """
    n = len(cols_in_order)
    assert n >= 2, "Need at least 2 columns (one feature + target)"
    ncols = 3
    nrows = ceil(n / ncols)

    plt.close("all")
    fig = plt.figure(figsize=(5*ncols, 3.4*nrows), facecolor="white")
    fig.suptitle("Training vs Testing distributions (KDE scaled to bin counts)",
                 fontsize=18, weight="bold", y=0.98)
    gs = plt.GridSpec(nrows, ncols, figure=fig, hspace=0.38, wspace=0.28)

    series_tr = {c: (ytr if c == TARGET else Xtr[c]) for c in cols_in_order}
    series_te = {c: (yte if c == TARGET else Xte[c]) for c in cols_in_order}

    k = 0
    for r in range(nrows):
        for c in range(ncols):
            if k >= n:
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")
                continue
            col = cols_in_order[k]
            ax = fig.add_subplot(gs[r, c])

            tr = pd.to_numeric(series_tr[col], errors="coerce").dropna().astype(float)
            te = pd.to_numeric(series_te[col], errors="coerce").dropna().astype(float)
            allv = pd.concat([tr, te]) if not te.empty else tr
            if allv.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                k += 1
                continue
            bins = np.linspace(allv.min(), allv.max(), 26)

            ax.hist(tr, bins=bins, color=TRAIN_BAR, alpha=0.88,
                    edgecolor="#065f46", linewidth=0.6, label="Training")
            if not te.empty:
                ax.hist(te, bins=bins, color=TEST_BAR, alpha=0.70, hatch="////",
                        edgecolor="#064e3b", linewidth=0.6, label="Testing")

            xg, yk_tr = _scaled_kde(tr, bins);  ax.plot(xg, yk_tr, color=KDE_TRAIN, lw=2.2)
            if not te.empty:
                xg2, yk_te = _scaled_kde(te, bins); ax.plot(xg2, yk_te, color=KDE_TEST, lw=2.0, ls="--")

            _nice_axes(ax)
            ax.set_title(label_map.get(col, col), fontsize=14, weight="bold", pad=6)
            ax.margins(y=0.08)
            ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=10)

            k += 1

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

# ============================ Pairplot =========================================
def green_pairplot(
    df: pd.DataFrame, order, label_map,
    outfile="PairPlot_Green.png",
    title="Pairwise Relationships",
    panel_height=2.1,                            
    font_scale=1.25,                                   
    label_fs=14,                                                         
    tick_fs=11                                                   
):
    cols  = [c for c in order if c in df.columns]
    numdf = df[cols].select_dtypes(include=[np.number]).copy()

                                                                     
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=font_scale)

    GREEN      = "#2e7d32"
    GREEN_EDGE = "#0b3d2e"

    g = sns.pairplot(
        numdf,
        diag_kind="kde",
        plot_kws=dict(color=GREEN, alpha=0.85, s=32, edgecolor=GREEN_EDGE, linewidth=0.4),
        diag_kws=dict(color=GREEN, fill=True, alpha=0.35, linewidth=1.0),
        height=panel_height
    )

           
    g.fig.suptitle(title, y=1.02, fontsize=int(label_fs*1.3), fontweight="bold")

                                                                           
    try:
        for j, col in enumerate(numdf.columns):
            ax = g.axes[-1, j]
            ax.set_xlabel(label_map.get(col, col), fontsize=label_fs, fontweight="bold", labelpad=10)
            ax.xaxis.get_label().set_rotation(30)
            ax.xaxis.get_label().set_horizontalalignment("right")
        for i, row in enumerate(numdf.columns):
            ax = g.axes[i, 0]
            ax.set_ylabel(label_map.get(row, row), fontsize=label_fs, fontweight="bold", labelpad=14)
            ax.yaxis.get_label().set_rotation(30)
            ax.yaxis.get_label().set_horizontalalignment("right")
            ax.yaxis.get_label().set_verticalalignment("center")
    except Exception:
        pass

                                                         
    for ax in g.axes.flat:
        if ax is None:
            continue
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for s in ax.spines.values():
            s.set_linewidth(0.9); s.set_color(GREEN_EDGE)

                                                       
                                                                            
    w, h = g.fig.get_size_inches()
    g.fig.set_size_inches(w*0.95, h*0.95)

    plt.tight_layout()
    g.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()

# ============================ Run visuals ======================================
                                                          
cols_for_grid = [c for c in ORDER_FOR_ALL if c != TARGET] + [TARGET]

plot_train_test_grid_kde_colored(
    X_train, X_test, y_train, y_test,
    cols_for_grid, pretty,
    outfile="train_test_hists_kde_colored.png"
)

                                        
correlation_heatmap(
    all_df_for_stats, ORDER_FOR_ALL, pretty,
    outfile="Correlation_Heatmap_ALL.png",
    title="Pearson correlation (ALL DATA)"
)

green_pairplot(
    all_df_for_stats, ORDER_FOR_ALL, pretty,
    outfile="PairPlot_Green_ALL.png",
    title="Pairwise Relationships (ALL DATA)"
)

print("Done. Saved files:")
print(" - Paper_Stats_Table.xlsx (ALL DATA)")
print(" - train_test_hists_kde_colored.png (train/test)")
print(" - Correlation_Heatmap_ALL.png (ALL DATA)")
print(" - PairPlot_Green_ALL.png (ALL DATA)")
