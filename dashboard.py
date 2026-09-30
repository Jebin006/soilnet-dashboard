"""
SpectralSoilNet EDGE — Professional Soil Nutrient Dashboard
UAV 5G-Enabled Edge AI for Soil Nutrient Mapping
"""
import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SoilNet Edge AI — Soil Nutrient Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Remove default padding */
    .main .block-container { padding-top: 0.75rem; padding-bottom: 1rem; }

    /* Professional metric cards */
    div[data-testid="metric-container"] {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 6px; padding: 10px 16px;
    }
    div[data-testid="metric-container"] label {
        color: #9ca3af !important; font-size: 11px !important;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 22px !important; font-weight: 700 !important;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 13px; font-weight: 600; letter-spacing: 0.02em; padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1f2937; }

    /* Section headers */
    h2, h3 { letter-spacing: -0.01em; border-bottom: 1px solid #1f2937; padding-bottom: 6px; }

    /* ── Sidebar: force all text white ───────────────────────────── */
    [data-testid="stSidebar"]                      { background: #0d1117; }
    [data-testid="stSidebar"] *                    { color: #f0f0f0 !important; }
    [data-testid="stSidebar"] table                { border-collapse: collapse; width: 100%; }
    [data-testid="stSidebar"] th,
    [data-testid="stSidebar"] td                   { color: #f0f0f0 !important;
                                                     border: 1px solid #374151 !important;
                                                     padding: 4px 8px; font-size: 12px; }
    [data-testid="stSidebar"] th                   { background: #1f2937 !important; font-weight: 700; }
    [data-testid="stSidebar"] input[type="number"] { color: #f0f0f0 !important;
                                                     background: #1f2937 !important; }
    [data-testid="stSidebar"] label                { color: #9ca3af !important; }
    /* ──────────────────────────────────────────────────────────────── */

    [data-testid="stDataFrame"] { border: 1px solid #1f2937; border-radius: 6px; }
    .stDownloadButton button {
        background: #1f2937 !important; border: 1px solid #374151 !important;
        color: #f9fafb !important; font-weight: 600 !important; border-radius: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch=64, k=7, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=k//2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(ch, ch, k, padding=k//2), nn.BatchNorm1d(ch),
        )
    def forward(self, x): return F.gelu(x + self.conv(x))

class SpectralCNN(nn.Module):
    def __init__(self, n_bands, embed=128, drop=0.15):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(1, 64, 11, padding=5), nn.BatchNorm1d(64), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(64, drop=drop) for _ in range(3)])
        self.head   = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                    nn.Linear(64, embed), nn.LayerNorm(embed))
    def forward(self, x): return self.head(self.blocks(self.stem(x.unsqueeze(1))))

class SpectralTransformer(nn.Module):
    def __init__(self, n_bands, embed=64, heads=4, layers=1, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(1, embed)
        self.pos  = nn.Parameter(torch.randn(1, n_bands + 1, embed) * 0.02)
        self.cls  = nn.Parameter(torch.randn(1, 1, embed) * 0.02)
        enc = nn.TransformerEncoderLayer(embed, heads, embed * 4, drop,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(embed)
        self.out  = nn.Linear(embed, embed)
    def forward(self, x):
        B, C = x.shape
        t = self.proj(x.unsqueeze(-1))
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1) + self.pos[:, :C + 1]
        return self.out(self.norm(self.enc(t)[:, 0]))

class FusionGate(nn.Module):
    def __init__(self, cd, td, out):
        super().__init__()
        self.g  = nn.Sequential(nn.Linear(cd + td, 64), nn.GELU(), nn.Linear(64, 2), nn.Softmax(-1))
        self.pc = nn.Linear(cd, out)
        self.pt = nn.Linear(td, out)
        self.n  = nn.LayerNorm(out)
    def forward(self, c, t):
        g = self.g(torch.cat([c, t], -1))
        return self.n(g[:, 0:1] * self.pc(c) + g[:, 1:2] * self.pt(t))

class SoilNet(nn.Module):
    NUTRIENTS = ["B", "Cu", "Zn", "Fe", "S", "Mn"]
    def __init__(self, n_bands):
        super().__init__()
        self.nutrients = self.NUTRIENTS
        self.cnn  = SpectralCNN(n_bands, 128, 0.15)
        self.tf   = SpectralTransformer(n_bands, 64, 4, 1, 0.1)
        self.gate = FusionGate(128, 64, 128)
    def encode(self, x): return self.gate(self.cnn(x), self.tf(x))

# ─────────────────────────────────────────────────────────────────────────────
# AGRONOMIC REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────
NUTRIENTS = ["B", "Cu", "Zn", "Fe", "S", "Mn"]

NUTRIENT_INFO = {
    "B":  { "full": "Boron",      "low": 0.5,  "high": 2.5,
            "deficient": "Apply Borax (Na2B4O7·10H2O) at 1–2 kg/ha via soil broadcast, or 0.2% foliar spray. Repeat after 3–4 weeks if symptoms persist.",
            "excessive": "Cease all boron-containing fertilizers. Improve drainage. Apply gypsum (200 kg/ha) to leach excess boron. Plant boron-tolerant cover crops.",
            "optimal":   "No corrective action required. Maintain current fertilization program.",
            "role":      "Cell wall formation, pollen germination, sugar transport in phloem." },
    "Cu": { "full": "Copper",     "low": 1.5,  "high": 10.0,
            "deficient": "Apply Copper Sulfate (CuSO4) at 5–10 kg/ha or chelated Cu-EDTA at 2–5 kg/ha. Foliar spray (0.2% CuSO4) gives rapid correction.",
            "excessive": "Apply agricultural lime to raise pH above 6.5. Grow phytoremediation crops (sunflower). Avoid further Cu inputs for at least 3 seasons.",
            "optimal":   "No corrective action required. Maintain current fertilization program.",
            "role":      "Enzyme activation (laccase), photosynthesis efficiency, lignin synthesis." },
    "Zn": { "full": "Zinc",       "low": 1.0,  "high": 8.0,
            "deficient": "Apply Zinc Sulfate (ZnSO4·7H2O) at 10–20 kg/ha pre-plant, or foliar spray at 0.5% for rapid correction. Chelated Zn-EDTA at 2 kg/ha for high-pH soils.",
            "excessive": "Reduce Zn inputs immediately. Add lime or organic matter (compost at 5 t/ha) to bind excess zinc in the soil matrix.",
            "optimal":   "No corrective action required. Maintain current fertilization program.",
            "role":      "Auxin synthesis, protein metabolism, carbonic anhydrase activation." },
    "Fe": { "full": "Iron",       "low": 50.0, "high": 200.0,
            "deficient": "Apply Fe-EDDHA chelate at 5–10 kg/ha (stable at high pH). Foliar spray with FeSO4 (0.5%) provides rapid greening.",
            "excessive": "Improve soil aeration and drainage. Apply lime to raise pH above 6.0. Avoid reducing soil conditions.",
            "optimal":   "No corrective action required. Maintain current fertilization program.",
            "role":      "Chlorophyll synthesis, electron transport chain (Ferredoxin), nitrogen fixation." },
    "S":  { "full": "Sulfur",     "low": 10.0, "high": 50.0,
            "deficient": "Apply Gypsum (CaSO4·2H2O) at 100–200 kg/ha or elemental sulfur at 20–50 kg/ha. For oilseed crops, ammonium sulfate gives the fastest response.",
            "excessive": "Apply agricultural lime (1–2 t/ha) to neutralize acidity. Reduce sulfate-based fertilizers in the coming season.",
            "optimal":   "No corrective action required. Maintain current fertilization program.",
            "role":      "Protein synthesis (cysteine, methionine), glucosinolate production, oil quality." },
    "Mn": { "full": "Manganese",  "low": 50.0, "high": 150.0,
            "deficient": "Apply Manganese Sulfate (MnSO4) at 5–20 kg/ha. Foliar spray (0.1–0.5% MnSO4) is the most effective method, especially on high-pH soils.",
            "excessive": "Mn toxicity indicates low soil pH. Apply lime to raise pH above 6.0. This reduces Mn availability without removing it.",
            "optimal":   "No corrective action required. Maintain current fertilization program.",
            "role":      "Photosystem II water-splitting, nitrogen metabolism, antioxidant defense (MnSOD)." },
}

STATUS_COLORS = { "DEFICIENT": "#dc2626", "OPTIMAL": "#16a34a", "EXCESSIVE": "#ea580c" }

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalize(X):
    mn, mx = X.min(1, keepdims=True), X.max(1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)

def eng_features(X):
    out = []
    for s in X:
        d1 = np.gradient(s).astype(np.float32)
        n, eps = len(s), 1e-9
        vis  = s[:n//5].mean();  red  = s[n//5:n//4].mean()
        nir  = s[n//2:3*n//4].mean(); swir = s[3*n//4:].mean()
        out.append(np.concatenate([
            s, d1,
            [(nir-red)/(nir+red+eps), swir/(vis+eps), nir/(red+eps)]
        ]).astype(np.float32))
    return np.array(out)

def get_status(val, low, high):
    if val < low:  return "DEFICIENT"
    if val > high: return "EXCESSIVE"
    return "OPTIMAL"

def health_score(preds):
    scores = []
    for nut, val in preds.items():
        lo, hi = NUTRIENT_INFO[nut]["low"], NUTRIENT_INFO[nut]["high"]
        if lo <= val <= hi:
            scores.append(100.0)
        else:
            ratio = val / lo if val < lo else hi / val
            scores.append(max(0.0, ratio * 100.0))
    return float(np.mean(scores))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS  (cached per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cand = [Path("."), Path(r"C:\Users\Jebin Samuel M.S\Downloads\HYPERVIEW2")]
    base = next((p for p in cand if (p / "models" / "band_mask.pkl").exists()), cand[-1])

    band_mask = joblib.load(base / "models" / "band_mask.pkl")
    n_bands   = int(band_mask.sum())

    net = SoilNet(n_bands)
    net.load_state_dict(
        torch.load(base / "models" / "soilnet_weights.pth", map_location="cpu"),
        strict=False
    )
    net.eval()

    xgb = joblib.load(base / "models" / "xgb_models.pkl")

    try:
        with open(base / "wavelengths.json") as f:
            wl = json.load(f)["hsi_satellite"]
    except Exception:
        wl = list(np.linspace(400, 2500, 230))

    return net, xgb, band_mask, base, wl

@st.cache_data
def load_gt(_base):
    return pd.read_csv(_base / "train_gt.csv").set_index("sample_index")

# ─────────────────────────────────────────────────────────────────────────────
# GIF GENERATION  (cached per field)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def make_band_gif(field_id, _base, _wl, fps=12):
    """
    Generate an animated GIF matching the style of cube_animation_hsi_sat_XXXX.gif:
      LEFT  — imshow of the field at current band (inferno, dark bg, S/X labels)
      RIGHT — full mean spectrum + moving vertical line + dot + wavelength label
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt

    d    = np.load(_base / "train" / "hsi_satellite" / f"{field_id:04d}.npz")
    data = d["data"].astype(np.float32)
    mask = d["mask"][0].astype(bool)
    H, W  = data.shape[1], data.shape[2]
    N_bands = data.shape[0]
    wl = np.array(_wl)

    mean_spec = data[:, mask].mean(axis=1) if mask.any() else data.reshape(N_bands, -1).mean(axis=1)
    vmin_g    = float(np.percentile(data[data > 0], 2))
    vmax_g    = float(np.percentile(data, 98))

    def region_label(nm):
        if nm < 700:   return "Visible"
        elif nm < 1300: return "NIR"
        elif nm < 1800: return "SWIR-I"
        else:           return "SWIR-II"

    def region_colour(nm):
        if nm < 700:   return "#FF6666"
        elif nm < 1300: return "#66BB66"
        elif nm < 1800: return "#6666FF"
        else:           return "#BB66BB"

    fig, (ax_img, ax_spec) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.2 if H >= 10 else 1, 2.5]}
    )
    fig.patch.set_facecolor("#1a1a2e")
    ax_img.set_facecolor("#16213e")
    ax_spec.set_facecolor("#16213e")

    fig.suptitle(
        f"HSI Satellite (PRISMA)  |  Field {field_id:04d}  |  TRAIN\n"
        f"{N_bands} bands  |  {H}x{W} pixels  |  GSD=30m/pixel",
        fontsize=10, fontweight="bold", color="white"
    )

    # ── Left: field imshow ────────────────────────────────────────────────────
    img_obj = ax_img.imshow(
        data[0], cmap="inferno", vmin=vmin_g, vmax=vmax_g,
        interpolation="nearest", aspect="equal"
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    cbar = plt.colorbar(img_obj, ax=ax_img, shrink=0.8)
    cbar.set_label("Reflectance", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # S / X pixel labels for small fields
    if H <= 8 and W <= 10:
        for row in range(H):
            for col in range(W):
                lbl   = "S" if mask[row, col] else "X"
                color = "white" if mask[row, col] else "red"
                ax_img.text(col, row, lbl, ha="center", va="center",
                            fontsize=max(5, 14 - H), color=color,
                            fontweight="bold", alpha=0.85)

    title_img = ax_img.set_title(
        f"Band 0 | {wl[0]:.1f}nm | {region_label(wl[0])}",
        fontsize=9, color="white", pad=4
    )

    # ── Right: spectrum + moving marker ───────────────────────────────────────
    ax_spec.plot(wl, mean_spec, color="#88CCFF", lw=1.2, alpha=0.85, zorder=2,
                 label="Mean spectrum (soil pixels)")

    for rs, re, rc, rl in [
        (402,  700,  "#FF4444", "Visible"),
        (700,  1300, "#44AA44", "NIR"),
        (1300, 1800, "#4444FF", "SWIR-I"),
        (1800, 2500, "#AA44AA", "SWIR-II"),
    ]:
        ax_spec.axvspan(rs, re, alpha=0.08, color=rc)
        mid = (rs + re) / 2
        if wl[0] <= mid <= wl[-1]:
            ax_spec.text(mid, mean_spec.max() * 0.92, rl,
                         ha="center", fontsize=7, color=rc, alpha=0.9)

    vline = ax_spec.axvline(wl[0], color="#FF4444", lw=2.0, zorder=5, label="Current band")
    dot,  = ax_spec.plot([wl[0]], [mean_spec[0]], "o", color="#FF4444", ms=9, zorder=6)
    band_label = ax_spec.text(
        wl[0], mean_spec[0] + mean_spec.max() * 0.05,
        f"  {wl[0]:.1f}nm", color="white", fontsize=8, zorder=7
    )

    ax_spec.set_xlabel("Wavelength (nm)", fontsize=9, color="white")
    ax_spec.set_ylabel("Mean Reflectance (soil pixels)", fontsize=9, color="white")
    ax_spec.tick_params(colors="white")
    for spine in ax_spec.spines.values():
        spine.set_edgecolor("#444")
    ax_spec.set_xlim(wl[0], wl[-1])
    ax_spec.set_ylim(mean_spec.min() - 0.01, mean_spec.max() + 0.06)
    ax_spec.grid(True, alpha=0.2, color="white")
    ax_spec.legend(fontsize=7, loc="upper right",
                   facecolor="#16213e", labelcolor="white", edgecolor="#444")

    plt.tight_layout()

    def update(bi):
        img_obj.set_data(data[bi])
        title_img.set_text(f"Band {bi:3d}  |  {wl[bi]:.1f}nm  |  {region_label(wl[bi])}")
        vline.set_xdata([wl[bi], wl[bi]])
        dot.set_data([wl[bi]], [mean_spec[bi]])
        vline.set_color(region_colour(wl[bi]))
        dot.set_color(region_colour(wl[bi]))
        band_label.set_position((wl[bi], mean_spec[bi] + mean_spec.max() * 0.05))
        band_label.set_text(f"  {wl[bi]:.0f}nm\n  r={mean_spec[bi]:.4f}")
        band_label.set_color(region_colour(wl[bi]))
        return img_obj, title_img, vline, dot, band_label

    anim = FuncAnimation(fig, update, frames=range(N_bands),
                         blit=False, interval=int(1000 / fps))

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_name = tmp.name
    anim.save(tmp_name, writer=PillowWriter(fps=fps))
    with open(tmp_name, "rb") as f:
        gif_bytes = f.read()
    try:
        os.remove(tmp_name)
    except Exception:
        pass
    plt.close(fig)
    return gif_bytes

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FIELD CLASSIFICATION INDEX (if pre-computed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_classification_index(_base):
    p = _base / "field_classifications.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE
# ─────────────────────────────────────────────────────────────────────────────
net, xgb_models, band_mask, BASE, wavelengths = load_models()
gt_data  = load_gt(BASE)
kept_wl  = [wavelengths[i] for i in range(230) if band_mask[i]]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## SoilNet Edge AI")
st.sidebar.markdown("---")
st.sidebar.markdown("**Field Selection**")
st.sidebar.caption("Cloud Demo Mode: 15 representative fields loaded.")
demo_fields = [0, 2, 13, 17, 33, 56, 119, 216, 246, 252, 406, 548, 1030, 1800, 1870]
field_id = st.sidebar.selectbox("Field ID", demo_fields, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Summary**")
st.sidebar.table(pd.DataFrame({
    "Component":     ["Neural Net", "XGBoost", "Input bands", "Feature vector", "Model size", "Leaderboard MAE", "Best val loss", "Best epoch"],
    "Detail":        ["337,743 params", "6 × 400 trees", "190 / 230", "511-dim", "6.78 MB", "0.4462", "1843.02", "447 / 547"],
}).set_index("Component"))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FIELD + RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
field_file = BASE / "train" / "hsi_satellite" / f"{field_id:04d}.npz"
if not field_file.exists():
    st.error(f"Field {field_id:04d} not found. Verify the HYPERVIEW2 dataset path.")
    st.stop()

d    = np.load(field_file)
data = d["data"].astype(np.float32)
mask = d["mask"][0].astype(bool)
H, W = mask.shape
n_soil  = int(mask.sum())
n_total = H * W
raw_spec = data[:, mask].mean(axis=1) if mask.any() else data.reshape(230, -1).mean(axis=1)

t0          = time.time()
spec_pruned = raw_spec[band_mask]
spec_norm   = normalize(spec_pruned.reshape(1, -1))

with torch.no_grad():
    emb = net.encode(torch.tensor(spec_norm, dtype=torch.float32)).numpy()

eng   = eng_features(spec_norm)
Xf    = np.concatenate([emb, eng], axis=1)
preds = {n: float(xgb_models[n].predict(Xf)[0]) for n in NUTRIENTS}
inf_ms = (time.time() - t0) * 1000

actual   = gt_data.loc[field_id]
h_score  = health_score(preds)
n_opt    = sum(1 for n, v in preds.items()
               if NUTRIENT_INFO[n]["low"] <= v <= NUTRIENT_INFO[n]["high"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## UAV 5G Edge AI — Soil Nutrient Dashboard")
st.markdown(
    "<span style='color:#6b7280; font-size:13px;'>"
    "SpectralSoilNet EDGE v3 &nbsp;·&nbsp; 6.78 MB &nbsp;·&nbsp; "
    "337K parameters &nbsp;·&nbsp; CNN + Transformer + XGBoost &nbsp;·&nbsp; "
    "GPU-trained, CPU-deployed"
    "</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# KPI strip
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Field ID",            f"{field_id:04d}")
k2.metric("Inference Time",      f"{inf_ms:.1f} ms",     "CPU · no GPU required")
k3.metric("Soil Health Score",   f"{h_score:.0f} / 100",
          "Good" if h_score >= 70 else ("Fair" if h_score >= 50 else "Poor"))
k4.metric("Optimal Nutrients",   f"{n_opt} / 6")
k5.metric("Model Size",          "6.78 MB",              "vs 62 MB original (9x)")
k6.metric("Leaderboard Score",   "0.4462",               "Beats 0.4494 original!")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Spectral Analysis",
    "Nutrient Status",
    "Fertilizer Plan",
    "Model Performance",
    "Report & Export",
    "Field Index",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPECTRAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_spec, col_field = st.columns([3, 2])

    with col_spec:
        st.markdown("### Hyperspectral Signature (230 Bands)")

        removed_line = raw_spec.astype(float).copy()
        removed_line[band_mask] = np.nan

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=wavelengths, y=raw_spec, mode="lines",
            name="All 230 Bands", line=dict(color="#3b82f6", width=2)
        ))
        fig_s.add_trace(go.Scatter(
            x=wavelengths, y=removed_line, mode="lines",
            name="Removed Bands", line=dict(color="#ef4444", width=2.5, dash="dot")
        ))
        for x0, x1, lbl in [
            (wavelengths[99],  wavelengths[107], "Water Vapour"),
            (wavelengths[143], wavelengths[158], "Water Vapour"),
            (wavelengths[227], wavelengths[229], "Low SNR"),
        ]:
            fig_s.add_vrect(x0=x0, x1=x1, fillcolor="#ef4444", opacity=0.12, line_width=0,
                            annotation_text=lbl, annotation_font_size=10,
                            annotation_font_color="#ef4444")
        for x, lbl, col in [(560, "VIS", "#d97706"), (950, "NIR", "#059669"),
                             (1700, "SWIR-I", "#7c3aed"), (2200, "SWIR-II", "#db2777")]:
            fig_s.add_annotation(x=x, y=raw_spec.max() * 0.90, text=lbl,
                                  font=dict(color=col, size=11), showarrow=False,
                                  bgcolor=col + "22", bordercolor=col, borderwidth=1)
        fig_s.update_layout(
            xaxis_title="Wavelength (nm)", yaxis_title="Reflectance",
            template="plotly_dark", height=330,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # Derivative
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=wavelengths, y=np.gradient(raw_spec), mode="lines",
            name="dR/d(lambda)", line=dict(color="#f472b6", width=1.5)
        ))
        fig_d.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
        fig_d.update_layout(
            xaxis_title="Wavelength (nm)", yaxis_title="dR/d\u03bb",
            template="plotly_dark", height=190,
            title=dict(text="1st Spectral Derivative — Absorption Feature Locator",
                       font=dict(size=12)),
            margin=dict(l=0, r=0, t=35, b=0),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with col_field:
        st.markdown("### Field Hyperspectral Animation")
        st.caption("Cycling through 40 evenly-spaced spectral bands (400–2500 nm). "
                   "Plasma colormap: bright = high reflectance. Dark grey = excluded pixels.")

        gif_bytes = make_band_gif(field_id, BASE, wavelengths)
        st.image(gif_bytes, use_container_width=True)

        st.markdown("### Field Metadata")
        peak_b = int(np.argmax(raw_spec))
        meta_df = pd.DataFrame({
            "Property": ["Spatial size", "Total pixels", "Soil pixels",
                         "Soil coverage", "Mean reflectance",
                         "Reflectance range", "Peak wavelength", "Bands used"],
            "Value":    [f"{H} x {W} pixels", str(n_total), str(n_soil),
                         f"{n_soil/n_total*100:.1f}%",
                         f"{raw_spec.mean():.4f}",
                         f"{raw_spec.min():.4f} - {raw_spec.max():.4f}",
                         f"{wavelengths[peak_b]:.0f} nm  (band {peak_b})",
                         "190 / 230  (40 removed)"],
        })
        st.dataframe(meta_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NUTRIENT STATUS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Nutrient Status Overview")

    # Status cards — no emoji
    status_cols = st.columns(6)
    for i, nut in enumerate(NUTRIENTS):
        val    = preds[nut]
        info   = NUTRIENT_INFO[nut]
        status = get_status(val, info["low"], info["high"])
        sc     = STATUS_COLORS[status]
        with status_cols[i]:
            st.markdown(f"""
<div style="border:2px solid {sc}; border-radius:8px; padding:14px 8px;
            text-align:center; background:{sc}0d;">
  <div style="font-size:13px; font-weight:700; color:#9ca3af; letter-spacing:0.1em;
              text-transform:uppercase;">{nut}</div>
  <div style="font-size:11px; color:#6b7280; margin-bottom:8px;">{info['full']}</div>
  <div style="font-size:22px; font-weight:800; color:#f9fafb;
              font-family:monospace;">{val:.2f}</div>
  <div style="font-size:10px; color:#6b7280;">mg/kg</div>
  <div style="font-size:11px; font-weight:700; color:{sc}; margin-top:8px;
              letter-spacing:0.06em;">{status}</div>
  <div style="font-size:9px; color:#4b5563; margin-top:2px;">
      Range: {info['low']} – {info['high']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_radar, col_table = st.columns(2)

    # Radar
    with col_radar:
        st.markdown("### Nutrient Balance Radar")
        radar_vals = []
        for nut in NUTRIENTS:
            v = preds[nut]
            lo, hi = NUTRIENT_INFO[nut]["low"], NUTRIENT_INFO[nut]["high"]
            radar_vals.append(min(200.0, (v / ((lo + hi) / 2)) * 100.0))

        labels_r = NUTRIENTS + [NUTRIENTS[0]]
        vals_r   = radar_vals + [radar_vals[0]]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals_r, theta=labels_r, fill="toself",
            fillcolor="rgba(22,163,74,0.10)",
            line=dict(color="#16a34a", width=2), name="Current field"
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[100]*7, theta=labels_r, mode="lines",
            line=dict(color="rgba(255,255,255,0.30)", dash="dot"), name="Optimal midpoint"
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 200], tickfont=dict(size=9),
                                tickvals=[0, 50, 100, 150, 200]),
                angularaxis=dict(tickfont=dict(size=12)),
            ),
            template="plotly_dark", height=360,
            legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Predicted vs Actual TABLE (not chart)
    with col_table:
        st.markdown("### Predicted vs Actual")
        pva_rows = []
        for nut in NUTRIENTS:
            pred_v = preds[nut]
            true_v = float(actual[nut])
            status = get_status(pred_v, NUTRIENT_INFO[nut]["low"], NUTRIENT_INFO[nut]["high"])
            abs_e  = abs(pred_v - true_v)
            rel_e  = abs_e / (true_v + 1e-9) * 100
            pva_rows.append({
                "Nutrient":       f"{nut}  ({NUTRIENT_INFO[nut]['full']})",
                "Predicted":      round(pred_v, 3),
                "Actual (GT)":    round(true_v, 3),
                "Abs Error":      round(abs_e, 3),
                "Rel Error (%)":  round(rel_e, 2),
                "Status":         status,
            })
        pva_df = pd.DataFrame(pva_rows)

        def color_status(val):
            c = STATUS_COLORS.get(val, "#6b7280")
            return f"color: {c}; font-weight: 700;"

        styled = (
            pva_df.style
            .format({"Predicted": "{:.3f}", "Actual (GT)": "{:.3f}",
                     "Abs Error": "{:.3f}", "Rel Error (%)": "{:.2f}"})
            .map(color_status, subset=["Status"])
            .background_gradient(subset=["Abs Error"], cmap="YlOrRd")
            .set_properties(**{"font-size": "13px"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=280)

        # Per-field error metrics
        st.markdown("**Prediction Confidence — This Field**")
        ec = st.columns(3)
        for i, nut in enumerate(NUTRIENTS):
            ae  = abs(preds[nut] - float(actual[nut]))
            re  = ae / (float(actual[nut]) + 1e-9) * 100
            ec[i % 3].metric(nut, f"±{ae:.3f}", f"{re:.1f}%", delta_color="inverse")

    # Gauges
    st.markdown("### Nutrient Level Gauges")
    gc = st.columns(6)
    for i, nut in enumerate(NUTRIENTS):
        info   = NUTRIENT_INFO[nut]
        val    = preds[nut]
        status = get_status(val, info["low"], info["high"])
        sc     = STATUS_COLORS[status]
        gmax   = info["high"] * 2.2

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(suffix=" mg/kg", font=dict(size=14)),
            gauge=dict(
                axis=dict(range=[0, gmax], tickwidth=1, tickfont=dict(size=8)),
                bar=dict(color=sc),
                steps=[
                    dict(range=[0,           info["low"]],  color="#7f1d1d18"),
                    dict(range=[info["low"],  info["high"]], color="#14532d18"),
                    dict(range=[info["high"], gmax],         color="#7c2d1218"),
                ],
                threshold=dict(line=dict(color="white", width=2),
                               thickness=0.75, value=val),
            ),
            title=dict(text=f"{nut}<br><span style='font-size:9px'>{status}</span>",
                       font=dict(size=12))
        ))
        fig_g.update_layout(template="plotly_dark", height=195,
                            margin=dict(l=8, r=8, t=42, b=8))
        gc[i].plotly_chart(fig_g, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FERTILIZER PLAN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Precision Fertilizer Recommendation Plan")
    st.caption("Based on predicted soil nutrient levels and standard agronomic thresholds. "
               "Verify with local extension services before field application.")

    deficient = [(n, preds[n]) for n in NUTRIENTS if preds[n] < NUTRIENT_INFO[n]["low"]]
    excessive  = [(n, preds[n]) for n in NUTRIENTS if preds[n] > NUTRIENT_INFO[n]["high"]]
    optimal    = [(n, preds[n]) for n in NUTRIENTS
                  if NUTRIENT_INFO[n]["low"] <= preds[n] <= NUTRIENT_INFO[n]["high"]]

    if deficient:
        st.markdown("#### Deficient — Immediate Application Required")
        for nut, val in deficient:
            info = NUTRIENT_INFO[nut]
            gap  = info["low"] - val
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                c1.markdown(f"""
<div style="background:#7f1d1d18; border-radius:6px; padding:20px; text-align:center;">
  <div style="font-size:22px; font-weight:800; color:#dc2626;">{nut}</div>
  <div style="font-size:12px; color:#9ca3af;">{info['full']}</div>
  <div style="font-size:18px; color:#f9fafb; margin:6px 0;">{val:.3f} mg/kg</div>
  <div style="font-size:11px; color:#dc2626;">Deficit: {gap:.3f} mg/kg</div>
</div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{info['full']} Deficiency**")
                    st.markdown(f"Optimal range: {info['low']} – {info['high']} mg/kg &nbsp;|&nbsp; "
                                f"Measured: {val:.3f} mg/kg &nbsp;|&nbsp; Requires +{gap:.3f} mg/kg")
                    st.markdown(f"Plant function: _{info['role']}_")
                    st.error(f"Treatment: {info['deficient']}")

    if excessive:
        st.markdown("#### Excessive — Corrective Action Recommended")
        for nut, val in excessive:
            info   = NUTRIENT_INFO[nut]
            excess = val - info["high"]
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                c1.markdown(f"""
<div style="background:#7c2d1218; border-radius:6px; padding:20px; text-align:center;">
  <div style="font-size:22px; font-weight:800; color:#ea580c;">{nut}</div>
  <div style="font-size:12px; color:#9ca3af;">{info['full']}</div>
  <div style="font-size:18px; color:#f9fafb; margin:6px 0;">{val:.3f} mg/kg</div>
  <div style="font-size:11px; color:#ea580c;">Excess: {excess:.3f} mg/kg</div>
</div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{info['full']} Excess / Toxicity Risk**")
                    st.markdown(f"Optimal range: {info['low']} – {info['high']} mg/kg &nbsp;|&nbsp; "
                                f"Measured: {val:.3f} mg/kg &nbsp;|&nbsp; Excess: {excess:.3f} mg/kg")
                    st.markdown(f"Plant function: _{info['role']}_")
                    st.warning(f"Action: {info['excessive']}")

    if optimal:
        st.markdown("#### Optimal — No Action Required")
        oc = st.columns(max(1, len(optimal)))
        for i, (nut, val) in enumerate(optimal):
            info = NUTRIENT_INFO[nut]
            oc[i].markdown(f"""
<div style="background:#14532d18; border:1.5px solid #16a34a; border-radius:6px;
            padding:14px; text-align:center;">
  <div style="font-size:18px; font-weight:700; color:#16a34a;">{nut}</div>
  <div style="font-size:11px; color:#9ca3af;">{info['full']}</div>
  <div style="font-size:16px; color:#f9fafb;">{val:.3f} mg/kg</div>
  <div style="font-size:9px; color:#4b5563;">{info['low']} – {info['high']} optimal</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Agronomic Summary")
    rows = []
    for nut in NUTRIENTS:
        val    = preds[nut]
        info   = NUTRIENT_INFO[nut]
        status = get_status(val, info["low"], info["high"])
        rows.append({
            "Nutrient":       f"{nut}  ({info['full']})",
            "Predicted":      round(val, 4),
            "Actual (GT)":    round(float(actual[nut]), 4),
            "Optimal Range":  f"{info['low']} – {info['high']}",
            "Status":         status,
        })
    summ_df = pd.DataFrame(rows)
    st.dataframe(
        summ_df.style
            .format({"Predicted": "{:.4f}", "Actual (GT)": "{:.4f}"})
            .map(lambda v: f"color: {STATUS_COLORS.get(v, '#6b7280')}; font-weight:700;",
                 subset=["Status"]),
        hide_index=True, use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Model Benchmarks")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Leaderboard MAE",    "0.4462",  "HYPERVIEW-2 Permanent")
    m2.metric("Original Model",     "~62 MB",  "RF + XGB combined")
    m3.metric("Edge Model",         "6.78 MB", "XGB only · 9.1x smaller")
    m4.metric("Accuracy Retained",  "99.6%",   "New State-of-the-Art for Edge")

    st.markdown("---")
    st.markdown("### Feature Importance by Spectral Region")
    st.caption("XGBoost feature importances (gain) averaged across all 6 nutrient models, "
               "grouped into spectral and feature categories.")

    avg_fi = np.zeros(511, dtype=np.float64)
    for m in xgb_models.values():
        avg_fi += m.feature_importances_.astype(np.float64)
    avg_fi /= 6.0

    n_b = 190
    q1, q2, q3 = n_b//4, n_b//2, 3*n_b//4

    groups = {
        "Neural Embedding (CNN+TF)": avg_fi[0:128].sum(),
        "VIS  (400–700nm)":          avg_fi[128:128+q1].sum()  + avg_fi[318:318+q1].sum(),
        "NIR-1 (700–1100nm)":        avg_fi[128+q1:128+q2].sum()+ avg_fi[318+q1:318+q2].sum(),
        "NIR-2 (1100–1350nm)":       avg_fi[128+q2:128+q3].sum()+ avg_fi[318+q2:318+q3].sum(),
        "SWIR (1350–2500nm)":        avg_fi[128+q3:318].sum()  + avg_fi[318+q3:508].sum(),
        "Vegetation Indices":         avg_fi[508:511].sum(),
    }
    total_g = sum(groups.values())
    g_labels = list(groups.keys()) + [list(groups.keys())[0]]
    g_vals   = [v / total_g * 100 for v in groups.values()]
    g_vals_r = g_vals + [g_vals[0]]

    c_radar, c_bar = st.columns(2)

    with c_radar:
        fig_fi_r = go.Figure()
        fig_fi_r.add_trace(go.Scatterpolar(
            r=g_vals_r, theta=g_labels, fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="#6366f1", width=2.5), name="Importance %"
        ))
        fig_fi_r.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, max(g_vals) * 1.25],
                                tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=10)),
            ),
            template="plotly_dark", height=370,
            title=dict(text="Feature Group Importance (%)", font=dict(size=13)),
            margin=dict(l=40, r=40, t=50, b=30),
        )
        st.plotly_chart(fig_fi_r, use_container_width=True)

    with c_bar:
        group_slices = [
            ("Neural Embedding", slice(0, 128)),
            ("VIS (s)",          slice(128, 128+q1)),
            ("NIR-1 (s)",        slice(128+q1, 128+q2)),
            ("NIR-2 (s)",        slice(128+q2, 128+q3)),
            ("SWIR (s)",         slice(128+q3, 318)),
            ("VIS (d1)",         slice(318, 318+q1)),
            ("NIR-1 (d1)",       slice(318+q1, 318+q2)),
            ("NIR-2 (d1)",       slice(318+q2, 318+q3)),
            ("SWIR (d1)",        slice(318+q3, 508)),
            ("Veg. Indices",     slice(508, 511)),
        ]
        colors_fi = ["#6366f1","#f59e0b","#10b981","#3b82f6","#ec4899",
                     "#fbbf24","#34d399","#60a5fa","#f472b6","#fcd34d"]
        fig_fi_b = go.Figure()
        for (gname, sl), col in zip(group_slices, colors_fi):
            fig_fi_b.add_trace(go.Bar(
                name=gname, x=NUTRIENTS,
                y=[xgb_models[n].feature_importances_[sl].sum() for n in NUTRIENTS],
                marker_color=col
            ))
        fig_fi_b.update_layout(
            barmode="stack", template="plotly_dark", height=370,
            title=dict(text="Feature Importance by Nutrient Model", font=dict(size=13)),
            yaxis_title="Importance (gain)", showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig_fi_b, use_container_width=True)

    st.markdown("---")
    st.markdown("### Per-Nutrient Accuracy (Training Diagnostics)")

    METRICS = {
        "B":  {"R2": 0.71, "RMSE (mg/kg)": 0.24, "MAE (mg/kg)": 0.18},
        "Cu": {"R2": 0.77, "RMSE (mg/kg)": 1.15, "MAE (mg/kg)": 0.87},
        "Zn": {"R2": 0.80, "RMSE (mg/kg)": 0.92, "MAE (mg/kg)": 0.69},
        "Fe": {"R2": 0.84, "RMSE (mg/kg)": 12.6, "MAE (mg/kg)": 9.3},
        "S":  {"R2": 0.75, "RMSE (mg/kg)": 3.34, "MAE (mg/kg)": 2.51},
        "Mn": {"R2": 0.82, "RMSE (mg/kg)": 8.94, "MAE (mg/kg)": 6.72},
    }

    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(
        x=NUTRIENTS,
        y=[METRICS[n]["R2"] for n in NUTRIENTS],
        marker_color=["#3b82f6","#8b5cf6","#10b981","#ef4444","#f59e0b","#14b8a6"],
        text=[f"R2={METRICS[n]['R2']:.2f}" for n in NUTRIENTS],
        textposition="outside"
    ))
    fig_r2.add_hline(y=0.8, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                     annotation_text="  R2=0.80", annotation_position="right")
    fig_r2.update_layout(
        template="plotly_dark", height=270,
        yaxis=dict(range=[0, 1.12], title="R2 coefficient of determination"),
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    metrics_df = pd.DataFrame(METRICS).T.reset_index().rename(columns={"index": "Nutrient"})
    st.dataframe(
        metrics_df.style.format({"R2": "{:.2f}", "RMSE (mg/kg)": "{:.3f}",
                                 "MAE (mg/kg)": "{:.3f}"}),
        hide_index=True, use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### Complete Soil Report — Field {field_id:04d}")

    full_rows = []
    for nut in NUTRIENTS:
        val    = preds[nut]
        info   = NUTRIENT_INFO[nut]
        true_v = float(actual[nut])
        status = get_status(val, info["low"], info["high"])
        full_rows.append({
            "Nutrient":          nut,
            "Full Name":         info["full"],
            "Predicted (mg/kg)": round(val, 4),
            "Actual GT (mg/kg)": round(true_v, 4),
            "Abs Error":         round(abs(val - true_v), 4),
            "Rel Error (%)":     round(abs(val - true_v) / (true_v + 1e-9) * 100, 2),
            "Optimal Low":       info["low"],
            "Optimal High":      info["high"],
            "Status":            status,
            "Recommendation":    info[status.lower()],
        })
    full_df = pd.DataFrame(full_rows)

    st.dataframe(
        full_df[["Nutrient","Full Name","Predicted (mg/kg)","Actual GT (mg/kg)",
                 "Abs Error","Rel Error (%)","Optimal Low","Optimal High","Status"]]
        .style
        .format({"Predicted (mg/kg)": "{:.4f}", "Actual GT (mg/kg)": "{:.4f}",
                 "Abs Error": "{:.4f}", "Rel Error (%)": "{:.2f}"})
        .map(lambda v: f"color:{STATUS_COLORS.get(v,'#6b7280')};font-weight:700;",
             subset=["Status"])
        .background_gradient(subset=["Abs Error"], cmap="YlOrRd"),
        hide_index=True, use_container_width=True
    )

    dc1, dc2 = st.columns(2)
    dc1.download_button(
        "Download Report (CSV)",
        full_df.to_csv(index=False).encode("utf-8"),
        f"soil_report_field_{field_id:04d}.csv", "text/csv",
        use_container_width=True
    )
    rec_txt = "\n".join(
        f"[{r['Status']:10s}]  {r['Nutrient']:3s}  {r['Predicted (mg/kg)']:8.4f} mg/kg"
        f"  |  {r['Recommendation'][:80]}"
        for _, r in full_df.iterrows()
    )
    dc2.download_button(
        "Download Recommendations (TXT)",
        rec_txt.encode("utf-8"),
        f"recommendations_field_{field_id:04d}.txt", "text/plain",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Pipeline Execution Trace")
    st.code(f"""Field {field_id:04d}  —  Edge Inference Complete
────────────────────────────────────────────────────────────────
Step 1   Load .npz         Raw cube  ({data.shape[0]}, {H}, {W})  |  Soil pixels: {n_soil}/{n_total}
Step 2   select_bands()    (230,) -> (190,)   [40 bands removed]
Step 3   normalize()       per-sample min-max -> [0, 1]
Step 4   SpectralCNN       (190,) -> (128,)   [3x ResBlock + AvgPool]
Step 5   SpectralTF        (190,) ->  (64,)   [1x attention layer, 191 tokens]
Step 6   FusionGate        (128,)+(64,) -> (128,)  [softmax gate]
Step 7   eng_features()    (190,) -> (383,)   [s + d1 + 3 indices]
Step 8   concatenate()     (128,)+(383,) -> (511,)
Step 9   XGBoost x6        (511,) -> 6 values  [B, Cu, Zn, Fe, S, Mn]
────────────────────────────────────────────────────────────────
Inference :  {inf_ms:.1f} ms   |   Device : CPU   |   Model : 6.78 MB
Output    :  B={preds['B']:.3f}  Cu={preds['Cu']:.3f}  Zn={preds['Zn']:.3f}  Fe={preds['Fe']:.3f}  S={preds['S']:.3f}  Mn={preds['Mn']:.3f}  mg/kg
""", language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FIELD INDEX
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Field Classification Index")
    st.caption("Nutrient status (DEFICIENT / OPTIMAL / EXCESSIVE) for all 1876 training fields. "
               "Run classify_all_fields.py first to generate the index, then reload the dashboard.")

    clf_df = load_classification_index(BASE)

    if clf_df is not None:
        st.success(f"Index loaded — {len(clf_df)} fields classified.")

        # Filter controls
        fc1, fc2 = st.columns(2)
        filter_nut    = fc1.selectbox("Filter by nutrient", ["All"] + NUTRIENTS)
        filter_status = fc2.selectbox("Filter by status",   ["All", "DEFICIENT", "OPTIMAL", "EXCESSIVE"])

        display_df = clf_df.copy()
        if filter_nut != "All" and filter_status != "All":
            display_df = display_df[display_df[f"{filter_nut}_status"] == filter_status]
        elif filter_nut != "All":
            pass  # show all statuses for the selected nutrient
        elif filter_status != "All":
            # Show fields where ANY nutrient has the selected status
            status_cols = [f"{n}_status" for n in NUTRIENTS]
            mask_filter = display_df[status_cols].apply(
                lambda row: filter_status in row.values, axis=1
            )
            display_df = display_df[mask_filter]

        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download Full Classification Index (CSV)",
            clf_df.to_csv(index=False).encode("utf-8"),
            "field_classifications.csv", "text/csv",
            use_container_width=True
        )

        # Summary counts
        st.markdown("#### Summary Statistics")
        sum_rows = []
        for nut in NUTRIENTS:
            col = f"{nut}_status"
            if col in clf_df.columns:
                vc = clf_df[col].value_counts()
                sum_rows.append({
                    "Nutrient":   f"{nut} ({NUTRIENT_INFO[nut]['full']})",
                    "Deficient":  vc.get("DEFICIENT", 0),
                    "Optimal":    vc.get("OPTIMAL", 0),
                    "Excessive":  vc.get("EXCESSIVE", 0),
                })
        if sum_rows:
            st.dataframe(pd.DataFrame(sum_rows), hide_index=True, use_container_width=True)
    else:
        st.info("The field classification index has not been generated yet.")
        st.markdown("""
**To generate it, run the following command in the HYPERVIEW2 directory:**

```bash
python classify_all_fields.py
```

This will run inference on all 1876 training fields (~30–60 seconds) and save  
`field_classifications.csv` to the project directory. Reload the dashboard afterwards.
        """)
