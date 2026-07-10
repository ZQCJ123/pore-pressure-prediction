# -*- coding: utf-8 -*-
"""
DTCO Hybrid Soft Sensing (ABS/RES Dual-domain)
Modified version:
- Keep ABS experts as: TrendResidualQuantile + XGBExpertABS + SeqExpertABS
- Replace RES experts with DIFFERENT model types to increase inductive diversity:
    1) RidgeExpert (linear, robust baseline on residual domain)
    2) GBDTQuantileExpert (direct quantile modeling on residual domain)
    3) SeqExpertRES (sequence-aware residual expert, unchanged)

The rest of the pipeline (proxy, OOD+proxy adaptive weighting, depth-bin smoothing/shrink, auto-lambda)
remains consistent with your original design.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.signal import savgol_filter
from xgboost import XGBRegressor


# =========================
# 参数区（你主要改路径 + 可选的lambda策略）
# =========================

# --- 关键：Hybrid 的 lambda 策略 ---
# 1) "auto"：运行时实时计算 λ（无监督；不使用测试井DTCO真值）
# 2) "manual"：手动指定每口井 λ（仅做消融/上界用）
LAMBDA_MODE = "manual"  # "auto" 或 "manual"

# 手动 λ（当 LAMBDA_MODE="manual" 生效）
LAMBDA_MANUAL = {
    "Well_A": 0.0,
    "Well_B": 0.0,
    "Well_C": 0.0,
}

# auto 模式下 sigmoid 参数（越大越“硬切换”）
AUTO_LAMBDA_K = 8.0

# auto λ 由多信号组合（都不需要测试井DTCO真值）
# - vel: 速度自洽（弱信号）
# - resid: RES 通道预测残差分布外程度（强信号，能保护Well_B）
# - proxyabs: proxy_LF 与 ABS低频趋势冲突程度（中等信号）
AUTO_LAMBDA_WEIGHTS = {"vel": 0.2, "resid": 0.6, "proxyabs": 0.2}
AUTO_LAMBDA_ABS_LF_WIN = 201
AUTO_LAMBDA_SCORE_CLIP = 10.0
AUTO_LAMBDA_FALLBACK = 0.5

# B/C 权重调参（可关掉：先固定 alpha=beta=1 更稳）
TUNE_ALPHA_BETA = False
ALPHA_GRID = [0.5, 1.0, 2.0]
BETA_GRID  = [0.5, 1.0, 2.0]
ALPHA0, BETA0 = 1.0, 1.0

# 画图平滑（只影响显示）
PLOT_SMOOTH = False
PLOT_SMOOTH_WIN = 9

# XGB 网格（跑慢就缩小）
XGB_PARAM_GRID = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [7, 9, 11],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}

# DTCO单位（us/ft 常见）
US_PER_S = 1e6
M_PER_FT = 0.3048

# COLS = ["TDEP","TT","TWT","Vave","Vrms","TT Gradient","ROP","WOB","RPM","TORQUE","FLOWIN","DXC","SEDP","DTCO"]
COLS = ["TDEP","TT","TWT","Vave","Vrms","TT Gradient","ROP","WOB","RPM","TORQUE","FLOWIN","DXC","SEDP","DTCO_P50","RHO8","DTCO"]
QUANTILES = (0.1, 0.5, 0.9)

# RESID_FEATURES = ["LOG_ROP","LOG_WOB","LOG_RPM","LOG_TORQUE","LOG_FLOWIN","LOG_DXC","LOG_SEDP"]
RESID_FEATURES = COLS
TREND_V4 = ["DTCO_PROXY","TWT_GRAD_RE","TT_GRAD_RE"]

# XGB_FEATURES_ABS = ["TDEP","TT","TWT","Vave","Vrms","TT Gradient","FLOWIN"]
XGB_FEATURES_ABS = COLS
XGB_FEATURES_RES = XGB_FEATURES_ABS + ["DTCO_PROXY_LF"]

# NEW: 更有“域差异”的 RES 专家特征（包含工程对数项，利于残差解释）
RIDGE_FEATURES_RES = XGB_FEATURES_RES + RESID_FEATURES
GBDT_FEATURES_RES  = XGB_FEATURES_RES + RESID_FEATURES + ["DTCO_PROXY"]  # 可按需删减/扩展

GBDT_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=2,
    min_samples_leaf=20, subsample=0.7, random_state=42
)

# 融合分箱参数
BIN_SIZE = 100.0
MIN_PTS = 40
SMOOTH_BINS = 5
LAMBDA_SHRINK = 0.5

# proxy贴合时：哪些专家用 P50 贴合 proxy（其它用 trend）
# UPDATED: 增加 ridge_res / gbdt_res
USE_P50_FOR_PROXY = {"xgb_abs", "seq_abs", "xgb_res", "seq_res", "ridge_res", "gbdt_res"}

# ★ 改进：P10/P90 用深度分箱残差（更稳、更合理）
USE_BINNED_P10P90 = True
P10P90_BIN_SIZE = 400.0
P10P90_MIN_COUNT = 40

# --- depth-varying lambda (ABS-RES fusion) ---
LAMBDA_DEPTH_VARYING = True   # True: λ(z); False: 井级标量λ
LAMBDA_BIN_SIZE = 100.0       # 深度分箱大小（与 BIN_SIZE 可相同也可不同）
LAMBDA_MIN_PTS = 40          # bin内最少点数，不够就回退全局λ
LAMBDA_SMOOTH_BINS = 5        # 对bin λ做rolling平滑

# =========================
# 工具函数
# =========================
def coerce_numeric(df):
    df = df.drop(columns=['trend', 'PP'], errors='ignore')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_columns(df, name):
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}")

def winsorize(s: pd.Series, ql=0.02, qh=0.98):
    lo, hi = s.quantile(ql), s.quantile(qh)
    return s.clip(lo, hi)

def compute_vint_dix(z, twt_ms, vrms_mps, eps=1e-9):
    z = np.asarray(z, float)
    t = np.asarray(twt_ms, float) / 1000.0
    v = np.asarray(vrms_mps, float)
    S = (v**2) * t
    dSdz = np.gradient(S, z)
    dtdz = np.gradient(t, z)
    vint2 = dSdz / (dtdz + eps)
    vint2 = np.maximum(vint2, eps)
    return np.sqrt(vint2)

def robust_stats(df, cols):
    med = df[cols].median()
    iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)
    iqr = iqr.replace(0, 1.0)
    return med, iqr

def ood_score(df, cols, med, iqr):
    z = ((df[cols] - med) / iqr).abs()
    return float(z.median().median())

def softmax_neg(scores):
    s = np.asarray(scores, float)
    s = s - np.min(s)
    w = np.exp(-s)
    return w / (w.sum() + 1e-12)

def mad_proxy(proxy, series):
    proxy = np.asarray(proxy, float)
    series = np.asarray(series, float)
    m = np.isfinite(proxy) & np.isfinite(series)
    if m.sum() < 50:
        return 1e6
    return float(np.median(np.abs(series[m] - proxy[m])))

# ★ 返回 MAE / RMSE / MAPE / R2
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs((y_true[denom] - y_pred[denom]) / y_true[denom])) * 100) if np.any(denom) else np.nan
    return mae, rmse, mape, r2

def _roll_med(a, win):
    return pd.Series(a).rolling(win, center=True, min_periods=1).median().values

def plot_dtco(depth, true, p50, p10, p90, title, path, smooth=False, win=9):
    idx = np.argsort(depth)
    depth, true, p50, p10, p90 = depth[idx], true[idx], p50[idx], p10[idx], p90[idx]

    if smooth:
        true_s  = _roll_med(true,  win)
        p50_s   = _roll_med(p50,   win)
        p10_s   = _roll_med(p10,   win)
        p90_s   = _roll_med(p90,   win)
    else:
        true_s, p50_s, p10_s, p90_s = true, p50, p10, p90

    plt.figure(figsize=(7,10))
    plt.fill_betweenx(depth, p10_s, p90_s, alpha=0.25, label="P10-P90")
    plt.plot(p50_s, depth, lw=1.5, label="Pred (P50)")
    plt.plot(true_s, depth, lw=1.2, label="True")
    plt.gca().invert_yaxis()
    plt.xlabel("DTCO"); plt.ylabel("TDEP"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

def align_to_common(idx: pd.Index, arr: np.ndarray, common: pd.Index) -> np.ndarray:
    pos = idx.get_indexer(common)
    if np.any(pos < 0):
        raise ValueError("common 不是 idx 的子集（对齐失败）")
    return np.asarray(arr)[pos]

def depth_bin_ids(depth, bin_size, d0):
    depth = np.asarray(depth, float)
    return np.floor((depth - d0) / bin_size).astype(int)

def build_binned_quantiles(depth, resid_err, bin_size, min_count):
    """返回：d0, global_q10, global_q90, dict(bin->(q10,q90))"""
    depth = np.asarray(depth, float)
    resid_err = np.asarray(resid_err, float)
    m = np.isfinite(depth) & np.isfinite(resid_err)
    if m.sum() < 50:
        return float(np.nanmin(depth)), float(np.quantile(resid_err[m], 0.1)), float(np.quantile(resid_err[m], 0.9)), {}

    d0 = float(np.nanmin(depth[m]))
    bid = depth_bin_ids(depth[m], bin_size, d0)

    gq10 = float(np.quantile(resid_err[m], 0.1))
    gq90 = float(np.quantile(resid_err[m], 0.9))

    qmap = {}
    for b in np.unique(bid):
        mm = (bid == b)
        if mm.sum() >= min_count:
            qmap[int(b)] = (float(np.quantile(resid_err[m][mm], 0.1)),
                            float(np.quantile(resid_err[m][mm], 0.9)))
    return d0, gq10, gq90, qmap

def apply_binned_quantiles(depth, d0, bin_size, gq10, gq90, qmap):
    depth = np.asarray(depth, float)
    bid = depth_bin_ids(depth, bin_size, d0)
    q10 = np.full(len(depth), gq10, float)
    q90 = np.full(len(depth), gq90, float)
    for i, b in enumerate(bid):
        t = qmap.get(int(b))
        if t is not None:
            q10[i], q90[i] = t
    return q10, q90

def _safe_ratio(x, ref, default=1.0):
    x = float(x)
    ref = float(ref)
    if (not np.isfinite(x)) or (not np.isfinite(ref)) or ref <= 0:
        return float(default)
    return float(x / (ref + 1e-12))


# =========================
# 特征工程：保留原始行 + DTCO_PROXY + DTCO_PROXY_LF
# =========================
def add_features_keep_rows(df_raw, smooth_win=11, grad_clip=(0.02,0.98), proxy_lf_win=201):
    df = df_raw.copy()
    if "ROW_ID" not in df.columns:
        df["ROW_ID"] = np.arange(len(df), dtype=int)

    num_cols = [c for c in df.columns if c != "ROW_ID"]
    df_u = df.groupby("TDEP", as_index=False)[num_cols].median().sort_values("TDEP").reset_index(drop=True)

    tt_s  = df_u["TT"].rolling(smooth_win, center=True, min_periods=1).median()
    twt_s = df_u["TWT"].rolling(smooth_win, center=True, min_periods=1).median()
    vrms_s = df_u["Vrms"].rolling(smooth_win, center=True, min_periods=1).median()

    z = df_u["TDEP"].values.astype(float)
    df_u["TT_GRAD_RE"]  = np.gradient(tt_s.values.astype(float),  z)
    df_u["TWT_GRAD_RE"] = np.gradient(twt_s.values.astype(float), z)
    df_u["TT_GRAD_RE"]  = winsorize(df_u["TT_GRAD_RE"],  *grad_clip)
    df_u["TWT_GRAD_RE"] = winsorize(df_u["TWT_GRAD_RE"], *grad_clip)

    vint = compute_vint_dix(z, twt_s.values.astype(float), vrms_s.values.astype(float))
    df_u["VINT_DIX"] = vint
    df_u["DTCO_PROXY"] = (1.0/(df_u["VINT_DIX"]+1e-12)) * US_PER_S * M_PER_FT
    df_u["DTCO_PROXY"] = winsorize(df_u["DTCO_PROXY"], 0.02, 0.98)

    df_u["DTCO_PROXY_LF"] = df_u["DTCO_PROXY"].rolling(proxy_lf_win, center=True, min_periods=1).median()

    feat_cols = ["TDEP","TT_GRAD_RE","TWT_GRAD_RE","DTCO_PROXY","DTCO_PROXY_LF"]
    df = df.merge(df_u[feat_cols], on="TDEP", how="left")

    for c in ["ROP","WOB","RPM","TORQUE","FLOWIN","DXC","SEDP"]:
        df[f"LOG_{c}"] = np.log1p(np.clip(df[c].values, 0, None))

    df = df.set_index("ROW_ID", drop=False)
    return df


# =========================
# Expert 1：Trend+Residual Quantiles（ABS 用）
# =========================
class TrendResidualQuantile:
    def __init__(self, trend_features, name="expert", base_col=None, clip_y=True):
        self.name = name
        self.trend_features = list(trend_features)
        self.resid_features = list(RESID_FEATURES)
        self.base_col = base_col
        self.clip_y = clip_y

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.trend = RidgeCV(alphas=np.logspace(-3,3,40))
        self.resid_models = [GradientBoostingRegressor(loss="quantile", alpha=q, **GBDT_PARAMS) for q in QUANTILES]

    def fit(self, df_train):
        need = ["DTCO"] + self.trend_features + self.resid_features
        if self.base_col:
            need += [self.base_col]
        df = df_train.dropna(subset=need).copy()

        X = df[self.trend_features + self.resid_features].values
        y = df[["DTCO"]].values.astype(float)

        if self.base_col:
            y = y - df[[self.base_col]].values.astype(float)

        Xn = self.x_scaler.fit_transform(X)
        yn = self.y_scaler.fit_transform(y).ravel()
        if self.clip_y:
            yn = np.clip(yn, 0.0, 1.0)

        Xt = Xn[:, :len(self.trend_features)]
        Xr = Xn[:, len(self.trend_features):]

        self.trend.fit(Xt, yn)
        trend_n = self.trend.predict(Xt)
        resid = yn - trend_n

        for m in self.resid_models:
            m.fit(Xr, resid)
        return self

    def predict(self, df_in):
        need = self.trend_features + self.resid_features
        if self.base_col:
            need += [self.base_col]
        df = df_in.dropna(subset=need).copy()

        X = df[self.trend_features + self.resid_features].values
        Xn = self.x_scaler.transform(X)
        Xt = Xn[:, :len(self.trend_features)]
        Xr = Xn[:, len(self.trend_features):]

        trend_n = self.trend.predict(Xt)
        resid_q = np.vstack([m.predict(Xr) for m in self.resid_models])
        pred_q_n = trend_n.reshape(1,-1) + resid_q

        if self.clip_y:
            pred_q_n = np.clip(pred_q_n, 0.0, 1.0)
        pred_q_n.sort(axis=0)

        pred_q = {q: self.y_scaler.inverse_transform(pred_q_n[i].reshape(-1,1)).ravel()
                  for i,q in enumerate(QUANTILES)}
        trend = self.y_scaler.inverse_transform(trend_n.reshape(-1,1)).ravel()

        if self.base_col:
            base = df[self.base_col].values.astype(float)
            trend = trend + base
            for q in QUANTILES:
                pred_q[q] = pred_q[q] + base

        return df.index, trend, pred_q


# =========================
# Expert 2：XGB ABS（保留原版）
# =========================
class XGBExpertABS:
    def __init__(self, name="xgb_abs"):
        self.name = name
        self.features = list(XGB_FEATURES_ABS)
        self.model = None

        self.gq10 = 0.0
        self.gq90 = 0.0
        self.q_d0 = 0.0
        self.qmap = {}

    def fit(self, df_train):
        need = ["DTCO"] + self.features
        df = df_train.dropna(subset=need).copy()
        X = df[self.features].values
        y = df["DTCO"].values.astype(float)

        base = XGBRegressor(objective="reg:squarederror", random_state=42)
        gs = GridSearchCV(base, XGB_PARAM_GRID, cv=5, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1)
        gs.fit(X, y)
        self.model = gs.best_estimator_
        print(f"[XGB_ABS] Best Parameters: {gs.best_params_}")

        yhat = self.model.predict(X)
        resid_err = y - yhat

        if USE_BINNED_P10P90:
            self.q_d0, self.gq10, self.gq90, self.qmap = build_binned_quantiles(
                depth=df["TDEP"].values, resid_err=resid_err,
                bin_size=P10P90_BIN_SIZE, min_count=P10P90_MIN_COUNT
            )
        else:
            self.gq10 = float(np.quantile(resid_err, 0.1))
            self.gq90 = float(np.quantile(resid_err, 0.9))
            self.q_d0 = float(np.nanmin(df["TDEP"].values))
            self.qmap = {}

        return self

    def predict(self, df_in):
        df = df_in.dropna(subset=self.features).copy()
        X = df[self.features].values
        p50 = self.model.predict(X)

        if USE_BINNED_P10P90:
            q10, q90 = apply_binned_quantiles(
                depth=df["TDEP"].values, d0=self.q_d0, bin_size=P10P90_BIN_SIZE,
                gq10=self.gq10, gq90=self.gq90, qmap=self.qmap
            )
        else:
            q10 = np.full(len(df), self.gq10, float)
            q90 = np.full(len(df), self.gq90, float)

        pred_q = {0.5: p50, 0.1: p50 + q10, 0.9: p50 + q90}
        stacked = np.vstack([pred_q[0.1], pred_q[0.5], pred_q[0.9]])
        stacked.sort(axis=0)
        pred_q[0.1], pred_q[0.5], pred_q[0.9] = stacked[0], stacked[1], stacked[2]
        return df.index, p50.copy(), pred_q


# =========================
# NEW Expert：Ridge（线性基线）—用于 RES 域
# =========================
class RidgeExpert:
    """
    线性Ridge专家：既可用于ABS也可用于RES
    - 若 base_col 不为 None：学习 (DTCO - base_col) 的残差，并在输出时回加 base_col
    - P10/P90：用训练残差误差的深度分箱分位数来构造区间（与你原脚本一致的逻辑）
    """
    def __init__(self, features, name="ridge", base_col=None):
        self.name = name
        self.base_col = base_col

        # 关键：特征去重，确保 fit/predict 维度一致
        self.features = dedup_keep_order(list(features))

        self.x_scaler = MinMaxScaler()
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 40))

        self.gq10 = 0.0
        self.gq90 = 0.0
        self.q_d0 = 0.0
        self.qmap = {}

    def fit(self, df_train):
        need = ["DTCO"] + self.features
        if self.base_col:
            need += [self.base_col]
        df = df_train.dropna(subset=need).copy()

        X = df[self.features].values.astype(float)   # ★ 只用 features
        y = df["DTCO"].values.astype(float)

        if self.base_col:
            y = y - df[self.base_col].values.astype(float)

        Xn = self.x_scaler.fit_transform(X)
        self.model.fit(Xn, y)

        yhat = self.model.predict(Xn)
        resid_err = y - yhat   # 用于区间的误差分布

        if USE_BINNED_P10P90:
            self.q_d0, self.gq10, self.gq90, self.qmap = build_binned_quantiles(
                depth=df["TDEP"].values, resid_err=resid_err,
                bin_size=P10P90_BIN_SIZE, min_count=P10P90_MIN_COUNT
            )
        else:
            self.gq10 = float(np.quantile(resid_err, 0.1))
            self.gq90 = float(np.quantile(resid_err, 0.9))
            self.q_d0 = float(np.nanmin(df["TDEP"].values))
            self.qmap = {}

        return self

    def predict(self, df_in):
        need = self.features
        if self.base_col:
            need = need + [self.base_col]
        df = df_in.dropna(subset=need).copy()

        X = df[self.features].values.astype(float)   # ★ 只用 features（不要拼 base_col）
        Xn = self.x_scaler.transform(X)

        p50_res = self.model.predict(Xn)
        if self.base_col:
            base = df[self.base_col].values.astype(float)
            p50 = base + p50_res
        else:
            p50 = p50_res

        if USE_BINNED_P10P90:
            q10, q90 = apply_binned_quantiles(
                depth=df["TDEP"].values, d0=self.q_d0, bin_size=P10P90_BIN_SIZE,
                gq10=self.gq10, gq90=self.gq90, qmap=self.qmap
            )
        else:
            q10 = np.full(len(df), self.gq10, float)
            q90 = np.full(len(df), self.gq90, float)

        pred_q = {0.5: p50, 0.1: p50 + q10, 0.9: p50 + q90}
        stacked = np.vstack([pred_q[0.1], pred_q[0.5], pred_q[0.9]])
        stacked.sort(axis=0)
        pred_q[0.1], pred_q[0.5], pred_q[0.9] = stacked[0], stacked[1], stacked[2]

        # 这里 trend 没必要强行定义为“趋势项”，直接用 p50 更安全（避免 proxy-score 失真）
        trend = p50.copy()
        return df.index, trend, pred_q


# =========================
# NEW Expert：GBDT Quantile（直接分位数）—用于 RES 域
# =========================
class GBDTQuantileExpert:
    """
    直接训练三条分位数回归曲线（q=0.1/0.5/0.9）。
    用于 RES 域时，训练目标为 DTCO-base_col；输出时回加 base_col。
    """
    def __init__(self, features, name="gbdt_q", base_col=None):
        self.name = name
        # self.features = list(features)
        self.features = dedup_keep_order(list(features))
        self.base_col = base_col

        self.x_scaler = MinMaxScaler()
        self.models = {q: GradientBoostingRegressor(loss="quantile", alpha=q, **GBDT_PARAMS) for q in QUANTILES}

    def fit(self, df_train):
        need = ["DTCO"] + self.features
        if self.base_col:
            need += [self.base_col]
        df = df_train.dropna(subset=need).copy()

        X = df[self.features].values
        y = df["DTCO"].values.astype(float)
        if self.base_col:
            y = y - df[self.base_col].values.astype(float)

        Xn = self.x_scaler.fit_transform(X)

        for q, m in self.models.items():
            m.fit(Xn, y)
        return self

    def predict(self, df_in):
        feat_cols = list(self.features)  # ★复制，避免被改坏
        need_cols = feat_cols + ([self.base_col] if self.base_col else [])

        df = df_in.dropna(subset=need_cols).copy()

        X = df[feat_cols].values  # ★这里只用 features，不包含 base_col
        Xn = self.x_scaler.transform(X)

        base = df[self.base_col].values.astype(float) if self.base_col else 0.0

        pred_q = {q: self.models[q].predict(Xn) + base for q in QUANTILES}
        stacked = np.vstack([pred_q[0.1], pred_q[0.5], pred_q[0.9]])
        stacked.sort(axis=0)
        pred_q[0.1], pred_q[0.5], pred_q[0.9] = stacked[0], stacked[1], stacked[2]

        trend = pred_q[0.5].copy()
        return df.index, trend, pred_q


# =========================
# Seq Expert：ABS / RES（原版保留）
# =========================
class SeqExpertABS:
    def __init__(self, name="seq_abs"):
        self.name = name
        self.lag_k = 6
        self.roll_win = 9
        self.tr_c2 = 0.0
        self.tr_c1 = 0.0
        self.tr_c0 = 0.0
        self.model = None
        self.feature_cols_ = None

        self.gq10 = 0.0
        self.gq90 = 0.0
        self.q_d0 = 0.0
        self.qmap = {}

    def _add_sequence_features(self, df, cols):
        df = df.sort_values("TDEP").copy()
        for c in cols:
            for k in range(1, self.lag_k + 1):
                df[f"{c}_L{k}"] = df[c].shift(k)
            df[f"{c}_RMEAN"] = df[c].shift(1).rolling(self.roll_win, min_periods=max(3, self.roll_win//3)).mean()
            df[f"{c}_RSTD"]  = df[c].shift(1).rolling(self.roll_win, min_periods=max(3, self.roll_win//3)).std()
        df["DZ"] = df["TDEP"].diff()
        return df

    def _fit_depth_trend_quadratic(self, df):
        du = df.groupby("TDEP", as_index=False)["DTCO"].median().sort_values("TDEP")
        du["DTCO_LF"] = du["DTCO"].rolling(201, center=True, min_periods=1).median()
        z = du["TDEP"].values.astype(float)
        y = du["DTCO_LF"].values.astype(float)
        c2, c1, c0 = np.polyfit(z, y, deg=2)
        self.tr_c2, self.tr_c1, self.tr_c0 = float(c2), float(c1), float(c0)

    def _trend(self, tdep):
        t = np.asarray(tdep, float)
        return self.tr_c2 * t**2 + self.tr_c1 * t + self.tr_c0

    def fit(self, df_train):
        need = ["DTCO","TDEP","TT","TWT","Vave","Vrms","TT Gradient","FLOWIN","DTCO_PROXY","TT_GRAD_RE","TWT_GRAD_RE"]
        df = df_train.dropna(subset=need).copy()

        self._fit_depth_trend_quadratic(df)
        df["DTCO_TREND"] = self._trend(df["TDEP"].values)
        df["RESID"] = df["DTCO"].values.astype(float) - df["DTCO_TREND"].values.astype(float)

        cols_for_seq = ["TT","TWT","Vrms","Vave","FLOWIN","TT Gradient","DTCO_PROXY","TT_GRAD_RE","TWT_GRAD_RE"]
        df = self._add_sequence_features(df, cols_for_seq)

        feat_cols = []
        for c in cols_for_seq:
            feat_cols.append(c)
            for k in range(1, self.lag_k+1):
                feat_cols.append(f"{c}_L{k}")
            feat_cols.append(f"{c}_RMEAN")
            feat_cols.append(f"{c}_RSTD")
        feat_cols += ["DZ", "TDEP"]
        self.feature_cols_ = feat_cols

        df2 = df.dropna(subset=self.feature_cols_ + ["RESID"]).copy()
        X = df2[self.feature_cols_].values
        y = df2["RESID"].values.astype(float)

        self.model = XGBRegressor(
            colsample_bytree=0.9, learning_rate=0.05, max_depth=11,
            n_estimators=400, subsample=0.7,
            random_state=42, objective="reg:squarederror", n_jobs=-1
        )
        self.model.fit(X, y)

        resid_hat = self.model.predict(X)
        resid_err = y - resid_hat

        if USE_BINNED_P10P90:
            self.q_d0, self.gq10, self.gq90, self.qmap = build_binned_quantiles(
                depth=df2["TDEP"].values, resid_err=resid_err,
                bin_size=P10P90_BIN_SIZE, min_count=P10P90_MIN_COUNT
            )
        else:
            self.gq10 = float(np.quantile(resid_err, 0.1))
            self.gq90 = float(np.quantile(resid_err, 0.9))
            self.q_d0 = float(np.nanmin(df2["TDEP"].values))
            self.qmap = {}

        return self

    def predict(self, df_in):
        need = ["TDEP","TT","TWT","Vave","Vrms","TT Gradient","FLOWIN","DTCO_PROXY","TT_GRAD_RE","TWT_GRAD_RE"]
        df = df_in.dropna(subset=need).copy().sort_values("TDEP")
        trend = self._trend(df["TDEP"].values)

        cols_for_seq = ["TT","TWT","Vrms","Vave","FLOWIN","TT Gradient","DTCO_PROXY","TT_GRAD_RE","TWT_GRAD_RE"]
        df = self._add_sequence_features(df, cols_for_seq)

        ok = df[self.feature_cols_].notna().all(axis=1).values
        resid_pred = np.zeros(len(df), float)
        if ok.sum() > 0:
            resid_pred[ok] = self.model.predict(df.loc[ok, self.feature_cols_].values)

        p50 = trend + resid_pred

        if USE_BINNED_P10P90:
            q10, q90 = apply_binned_quantiles(
                depth=df["TDEP"].values, d0=self.q_d0, bin_size=P10P90_BIN_SIZE,
                gq10=self.gq10, gq90=self.gq90, qmap=self.qmap
            )
        else:
            q10 = np.full(len(df), self.gq10, float)
            q90 = np.full(len(df), self.gq90, float)

        pred_q = {0.5: p50, 0.1: p50 + q10, 0.9: p50 + q90}
        stacked = np.vstack([pred_q[0.1], pred_q[0.5], pred_q[0.9]])
        stacked.sort(axis=0)
        pred_q[0.1], pred_q[0.5], pred_q[0.9] = stacked[0], stacked[1], stacked[2]
        return df.index, trend, pred_q


class SeqExpertRES:
    def __init__(self, name="seq_res"):
        self.name = name
        self.lag_k = 6
        self.roll_win = 9
        self.model = None
        self.feature_cols_ = None

        self.gq10 = 0.0
        self.gq90 = 0.0
        self.q_d0 = 0.0
        self.qmap = {}

    def _add_sequence_features(self, df, cols):
        df = df.sort_values("TDEP").copy()
        for c in cols:
            for k in range(1, self.lag_k + 1):
                df[f"{c}_L{k}"] = df[c].shift(k)
            df[f"{c}_RMEAN"] = df[c].shift(1).rolling(self.roll_win, min_periods=max(3, self.roll_win//3)).mean()
            df[f"{c}_RSTD"]  = df[c].shift(1).rolling(self.roll_win, min_periods=max(3, self.roll_win//3)).std()
        df["DZ"] = df["TDEP"].diff()
        return df

    def fit(self, df_train):
        need = ["DTCO","TDEP","TT","TWT","Vave","Vrms","TT Gradient","FLOWIN",
                "DTCO_PROXY","DTCO_PROXY_LF","TT_GRAD_RE","TWT_GRAD_RE"]
        df = df_train.dropna(subset=need).copy()

        df["DTCO_TREND"] = df["DTCO_PROXY_LF"].values.astype(float)
        df["RESID"] = df["DTCO"].values.astype(float) - df["DTCO_TREND"].values.astype(float)

        cols_for_seq = ["TT","TWT","Vrms","Vave","FLOWIN","TT Gradient",
                        "DTCO_PROXY","DTCO_PROXY_LF","TT_GRAD_RE","TWT_GRAD_RE"]
        df = self._add_sequence_features(df, cols_for_seq)

        feat_cols = []
        for c in cols_for_seq:
            feat_cols.append(c)
            for k in range(1, self.lag_k+1):
                feat_cols.append(f"{c}_L{k}")
            feat_cols.append(f"{c}_RMEAN")
            feat_cols.append(f"{c}_RSTD")
        feat_cols += ["DZ", "TDEP"]
        self.feature_cols_ = feat_cols

        df2 = df.dropna(subset=self.feature_cols_ + ["RESID"]).copy()
        X = df2[self.feature_cols_].values
        y = df2["RESID"].values.astype(float)

        self.model = XGBRegressor(
            colsample_bytree=0.9, learning_rate=0.05, max_depth=11,
            n_estimators=400, subsample=0.7,
            random_state=42, objective="reg:squarederror", n_jobs=-1
        )
        self.model.fit(X, y)

        resid_hat = self.model.predict(X)
        resid_err = y - resid_hat

        if USE_BINNED_P10P90:
            self.q_d0, self.gq10, self.gq90, self.qmap = build_binned_quantiles(
                depth=df2["TDEP"].values, resid_err=resid_err,
                bin_size=P10P90_BIN_SIZE, min_count=P10P90_MIN_COUNT
            )
        else:
            self.gq10 = float(np.quantile(resid_err, 0.1))
            self.gq90 = float(np.quantile(resid_err, 0.9))
            self.q_d0 = float(np.nanmin(df2["TDEP"].values))
            self.qmap = {}

        return self

    def predict(self, df_in):
        need = ["TDEP","TT","TWT","Vave","Vrms","TT Gradient","FLOWIN",
                "DTCO_PROXY","DTCO_PROXY_LF","TT_GRAD_RE","TWT_GRAD_RE"]
        df = df_in.dropna(subset=need).copy().sort_values("TDEP")

        trend = df["DTCO_PROXY_LF"].values.astype(float)

        cols_for_seq = ["TT","TWT","Vrms","Vave","FLOWIN","TT Gradient",
                        "DTCO_PROXY","DTCO_PROXY_LF","TT_GRAD_RE","TWT_GRAD_RE"]
        df = self._add_sequence_features(df, cols_for_seq)

        ok = df[self.feature_cols_].notna().all(axis=1).values
        resid_pred = np.zeros(len(df), float)
        if ok.sum() > 0:
            resid_pred[ok] = self.model.predict(df.loc[ok, self.feature_cols_].values)

        p50 = trend + resid_pred

        if USE_BINNED_P10P90:
            q10, q90 = apply_binned_quantiles(
                depth=df["TDEP"].values, d0=self.q_d0, bin_size=P10P90_BIN_SIZE,
                gq10=self.gq10, gq90=self.gq90, qmap=self.qmap
            )
        else:
            q10 = np.full(len(df), self.gq10, float)
            q90 = np.full(len(df), self.gq90, float)

        pred_q = {0.5: p50, 0.1: p50 + q10, 0.9: p50 + q90}
        stacked = np.vstack([pred_q[0.1], pred_q[0.5], pred_q[0.9]])
        stacked.sort(axis=0)
        pred_q[0.1], pred_q[0.5], pred_q[0.9] = stacked[0], stacked[1], stacked[2]
        return df.index, trend, pred_q


# =========================
# 融合：proxy + OOD（分箱 + 平滑 + shrink）
# =========================
def compute_weights(df_feat, common, exp_outs, ood_infos, alpha=1.0, beta=1.0,
                    domain="abs", resid_ref=None):
    """
    domain="abs": score1 = proxy MAD( DTCO_PROXY vs trend_like ), score2 = input OOD
    domain="res": score1 = residual-output OOD( rhat vs training residual stats ), score2 = input OOD
                 （完全不使用 DTCO_PROXY）
    """
    dfc = df_feat.loc[common]

    # 输入侧 OOD（对输入特征）
    ood_scores = []
    for (cols, A_med, A_iqr) in ood_infos:
        ood_scores.append(ood_score(dfc, cols, A_med, A_iqr))
    ood_scores = np.asarray(ood_scores, float)
    ood_n = ood_scores / (np.median(ood_scores) + 1e-12)

    if domain == "abs":
        proxy = dfc["DTCO_PROXY"].values.astype(float)

        proxy_scores = []
        for (name, idx, trend, pred_q) in exp_outs:
            p50_c = align_to_common(idx, pred_q[0.5], common)
            tr_c  = align_to_common(idx, trend, common)
            trend_like = p50_c if name in USE_P50_FOR_PROXY else tr_c
            proxy_scores.append(mad_proxy(proxy, trend_like))

        proxy_scores = np.asarray(proxy_scores, float)
        proxy_n = proxy_scores / (np.median(proxy_scores) + 1e-12)

        total = alpha * proxy_n + beta * ood_n
        return softmax_neg(total)

    elif domain == "res":
        if resid_ref is None:
            raise ValueError("domain='res' 时必须传 resid_ref={'rA_med':..., 'rA_iqr':...}")

        proxy_lf = dfc["DTCO_PROXY_LF"].values.astype(float)
        r_med = float(resid_ref["rA_med"])
        r_iqr = float(resid_ref["rA_iqr"])

        resid_scores = []
        for (name, idx, trend, pred_q) in exp_outs:
            p50_c = align_to_common(idx, pred_q[0.5], common)
            resid_scores.append(resid_output_ood_score(p50_c, proxy_lf, r_med, r_iqr))

        resid_scores = np.asarray(resid_scores, float)
        resid_n = resid_scores / (np.median(resid_scores) + 1e-12)

        # RES 域：用 alpha 表示“残差输出OOD”的权重，beta 表示“输入OOD”的权重
        total = alpha * resid_n + beta * ood_n
        return softmax_neg(total)

    else:
        raise ValueError("domain 必须是 'abs' 或 'res'")

def apply_weights_pointwise(exp_outs, common, w_mat):
    n = len(common)
    preds = {q: np.zeros(n, float) for q in QUANTILES}
    trend_b = np.zeros(n, float)

    for j, (name, idx, trend, pred_q) in enumerate(exp_outs):
        tr_c = align_to_common(idx, trend, common)
        for q in QUANTILES:
            preds[q] += w_mat[:, j] * align_to_common(idx, pred_q[q], common)
        trend_b += w_mat[:, j] * tr_c

    stacked = np.vstack([preds[0.1], preds[0.5], preds[0.9]])
    stacked.sort(axis=0)
    preds[0.1], preds[0.5], preds[0.9] = stacked[0], stacked[1], stacked[2]
    return preds, trend_b

def weights_by_depth_bins(df_feat, exp_outs, ood_infos, alpha=1.0, beta=1.0,
                          bin_size=200.0, min_pts=400, smooth_bins=5, lambda_shrink=0.35,
                          domain="abs", resid_ref=None):
    common = exp_outs[0][1]
    for _, idx, _, _ in exp_outs[1:]:
        common = common.intersection(idx)
    common = common.sort_values()

    dfc = df_feat.loc[common].copy()
    depth = dfc["TDEP"].values.astype(float)

    # 全局权重（bin 样本不足时回退）
    w_global = compute_weights(df_feat, common, exp_outs, ood_infos,
                               alpha=alpha, beta=beta, domain=domain, resid_ref=resid_ref)

    d0 = np.nanmin(depth)
    bin_id = np.floor((depth - d0) / bin_size).astype(int)

    k = len(exp_outs)
    n = len(common)

    # 预取每个专家的 p50 / trend（后面 bin 内要用）
    p50_mat = np.zeros((k, n), float)
    tr_mat  = np.zeros((k, n), float)
    for j, (name, idx, trend, pred_q) in enumerate(exp_outs):
        p50_mat[j] = align_to_common(idx, pred_q[0.5], common)
        tr_mat[j]  = align_to_common(idx, trend, common)

    uniq_bins = np.unique(bin_id)
    W_bin = np.zeros((len(uniq_bins), k), float)

    for bi, b in enumerate(uniq_bins):
        mask = (bin_id == b)
        if mask.sum() < min_pts:
            W_bin[bi, :] = w_global
            continue

        df_sub = dfc.iloc[np.where(mask)[0]]

        # 输入侧 OOD（每个专家一份）
        ood_scores = []
        for (cols, A_med, A_iqr) in ood_infos:
            ood_scores.append(ood_score(df_sub, cols, A_med, A_iqr))
        ood_scores = np.asarray(ood_scores, float)
        ood_n = ood_scores / (np.median(ood_scores) + 1e-12)

        if domain == "abs":
            proxy = df_sub["DTCO_PROXY"].values.astype(float)

            proxy_scores = []
            for j, (nm, *_rest) in enumerate(exp_outs):
                trend_like = p50_mat[j] if nm in USE_P50_FOR_PROXY else tr_mat[j]
                proxy_scores.append(mad_proxy(proxy, trend_like[mask]))
            proxy_scores = np.asarray(proxy_scores, float)
            proxy_n = proxy_scores / (np.median(proxy_scores) + 1e-12)

            total = alpha * proxy_n + beta * ood_n
            W_bin[bi, :] = softmax_neg(total)

        elif domain == "res":
            if resid_ref is None:
                raise ValueError("domain='res' 时必须传 resid_ref")

            proxy_lf = df_sub["DTCO_PROXY_LF"].values.astype(float)
            r_med = float(resid_ref["rA_med"])
            r_iqr = float(resid_ref["rA_iqr"])

            resid_scores = []
            for j in range(k):
                p50_b = p50_mat[j][mask]
                resid_scores.append(resid_output_ood_score(p50_b, proxy_lf, r_med, r_iqr))

            resid_scores = np.asarray(resid_scores, float)
            resid_n = resid_scores / (np.median(resid_scores) + 1e-12)

            total = alpha * resid_n + beta * ood_n
            W_bin[bi, :] = softmax_neg(total)

        else:
            raise ValueError("domain 必须是 'abs' 或 'res'")

    # bin 权重平滑
    if smooth_bins and smooth_bins > 1 and len(uniq_bins) >= smooth_bins:
        W_s = pd.DataFrame(W_bin).rolling(smooth_bins, center=True, min_periods=1).mean().values
        W_s = np.clip(W_s, 1e-12, None)
        W_s = W_s / (W_s.sum(axis=1, keepdims=True) + 1e-12)
        W_bin = W_s

    # 收缩到全局权重，防止 bin 不稳定
    W_bin = (1 - lambda_shrink) * W_bin + lambda_shrink * w_global.reshape(1, -1)
    W_bin = np.clip(W_bin, 1e-12, None)
    W_bin = W_bin / (W_bin.sum(axis=1, keepdims=True) + 1e-12)

    bin_to_row = {b: i for i, b in enumerate(uniq_bins)}
    w_mat = np.zeros((n, k), float)
    for i in range(n):
        w_mat[i, :] = W_bin[bin_to_row[bin_id[i]], :]

    return common, w_mat

def dedup_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def run_ensemble(name, df_feat, experts, ood_infos, alpha=1.0, beta=1.0, verbose=True,
                 domain="abs", resid_ref=None):
    exp_outs = []
    for e in experts:
        idx, trend, pred_q = e.predict(df_feat)
        exp_outs.append((e.name, idx, trend, pred_q))

    common, w_mat = weights_by_depth_bins(
        df_feat, exp_outs, ood_infos,
        alpha=alpha, beta=beta,
        bin_size=BIN_SIZE, min_pts=MIN_PTS,
        smooth_bins=SMOOTH_BINS, lambda_shrink=LAMBDA_SHRINK,
        domain=domain, resid_ref=resid_ref
    )
    preds, trend_b = apply_weights_pointwise(exp_outs, common, w_mat)

    y_true = df_feat.loc[common, "DTCO"].values.astype(float)
    mae, rmse, mape, r2 = metrics(y_true, preds[0.5])

    if verbose:
        w_mean = w_mat.mean(axis=0)
        print(
            f"\n[{name}] alpha={alpha}, beta={beta}, "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}, weights:",
            {experts[i].name: float(w_mean[i]) for i in range(len(experts))}
        )
    return common, preds, trend_b, w_mat, (mae, rmse, mape, r2)


# =========================
# Hybrid λ：auto 模式（无监督评分）
# =========================
def velocity_consistency_score(df_feat):
    df = df_feat.dropna(subset=["TDEP","TWT","Vave"]).copy()
    if len(df) < 200:
        return np.inf

    du = df.groupby("TDEP", as_index=False)[["TDEP","TWT","Vave"]].median().sort_values("TDEP")
    z = du["TDEP"].values.astype(float)
    twt_s = du["TWT"].values.astype(float) / 1000.0
    vave = du["Vave"].values.astype(float)

    ok = (z > 0) & (twt_s > 0) & (vave > 0) & np.isfinite(z) & np.isfinite(twt_s) & np.isfinite(vave)
    if ok.sum() < 200:
        return np.inf

    vave_est = 2.0 * z[ok] / twt_s[ok]
    ratio = vave_est / vave[ok]
    ratio = ratio[np.isfinite(ratio) & (ratio > 1e-12)]
    if len(ratio) < 200:
        return np.inf

    return float(np.median(np.abs(np.log(ratio))))

def compute_lambda_multi_signal(
    df_feat: pd.DataFrame,
    common: pd.Index,
    abs_p50: np.ndarray,
    res_p50: np.ndarray,
    ref_pack: dict,
    k: float = 8.0
):
    """
    无监督 λ：结合三类信号（都不使用测试井DTCO真值）
      1) vel: 速度自洽(越大越差)
      2) resid: RES预测残差的分布外程度(越大越差)
      3) proxyabs: proxy_LF 与 ABS低频趋势差异(越大越差)
    """
    if ("DTCO_PROXY_LF" not in df_feat.columns) or (len(common) < 200):
        return AUTO_LAMBDA_FALLBACK, {"vel": np.inf, "resid": np.inf, "proxyabs": np.inf, "S": np.inf}

    proxy_lf = df_feat.loc[common, "DTCO_PROXY_LF"].values.astype(float)

    # 1) vel
    s_vel = velocity_consistency_score(df_feat)

    # 2) resid OOD（预测残差 vs 训练残差分布）
    rhat = np.asarray(res_p50, float) - proxy_lf
    rA_med = float(ref_pack["rA_med"])
    rA_iqr = float(ref_pack["rA_iqr"])
    if (not np.isfinite(rA_iqr)) or rA_iqr <= 1e-9:
        rA_iqr = 1.0
    m = np.isfinite(rhat)
    s_resid = float(np.median(np.abs((rhat[m] - rA_med) / rA_iqr))) if m.sum() >= 50 else np.inf

    # 3) proxy vs ABS low-freq disagreement
    abs_lf = pd.Series(np.asarray(abs_p50, float)).rolling(
        AUTO_LAMBDA_ABS_LF_WIN, center=True, min_periods=1
    ).median().values
    mm = np.isfinite(abs_lf) & np.isfinite(proxy_lf)
    s_proxyabs = float(np.median(np.abs(abs_lf[mm] - proxy_lf[mm]))) if mm.sum() >= 50 else np.inf

    rv = _safe_ratio(s_vel, ref_pack["ref_vel"], default=AUTO_LAMBDA_SCORE_CLIP)
    rr = _safe_ratio(s_resid, ref_pack["ref_resid"], default=AUTO_LAMBDA_SCORE_CLIP)
    rp = _safe_ratio(s_proxyabs, ref_pack["ref_proxyabs"], default=AUTO_LAMBDA_SCORE_CLIP)

    rv = float(np.clip(rv, 0.0, AUTO_LAMBDA_SCORE_CLIP))
    rr = float(np.clip(rr, 0.0, AUTO_LAMBDA_SCORE_CLIP))
    rp = float(np.clip(rp, 0.0, AUTO_LAMBDA_SCORE_CLIP))

    wv = AUTO_LAMBDA_WEIGHTS["vel"]
    wr = AUTO_LAMBDA_WEIGHTS["resid"]
    wp = AUTO_LAMBDA_WEIGHTS["proxyabs"]

    S = wv * rv + wr * rr + wp * rp

    lam = 1.0 / (1.0 + np.exp(k * (S - 1.0)))
    lam = float(np.clip(lam, 0.0, 1.0))

    info = {"vel": s_vel, "resid": s_resid, "proxyabs": s_proxyabs, "S": S, "rv": rv, "rr": rr, "rp": rp}
    return lam, info

import numpy as np
import pandas as pd

def compute_lambda_by_depth_bins(
    df_feat,
    common,
    abs_p50,          # 必须已经对齐到 common
    res_p50,          # 必须已经对齐到 common
    ref_pack,
    k,
    bin_size=200.0,
    min_pts=400,
    smooth_bins=5,
    lambda_shrink=0.35
):
    """
    在每个深度 bin 上计算 lambda_bin，然后映射回每个点得到 lambda(z)。
    关键修复：bin 内 abs_p50/res_p50 用 index 对齐截取，避免长度错配。
    """
    common = common.sort_values()

    # ---- 强制检查：abs_p50/res_p50 必须与 common 同长度 ----
    abs_p50 = np.asarray(abs_p50, float)
    res_p50 = np.asarray(res_p50, float)
    if len(abs_p50) != len(common) or len(res_p50) != len(common):
        raise ValueError(
            f"[compute_lambda_by_depth_bins] abs_p50/res_p50 必须已对齐到 common。\n"
            f"len(common)={len(common)}, len(abs_p50)={len(abs_p50)}, len(res_p50)={len(res_p50)}"
        )

    # 用 Series + index，bin 内用 loc 取子段（最稳）
    abs_s = pd.Series(abs_p50, index=common)
    res_s = pd.Series(res_p50, index=common)

    dfc = df_feat.loc[common].copy()
    depth = dfc["TDEP"].values.astype(float)

    # ---- 1) 全局 lambda（回退+收缩中心）----
    lam_global, lam_info_global = compute_lambda_multi_signal(
        df_feat=df_feat,
        common=common,
        abs_p50=abs_s.loc[common].values,
        res_p50=res_s.loc[common].values,
        ref_pack=ref_pack,
        k=k
    )

    # ---- 2) 分箱 ----
    d0 = np.nanmin(depth)
    bin_id = np.floor((depth - d0) / bin_size).astype(int)
    uniq_bins = np.unique(bin_id)

    lam_bin = np.zeros(len(uniq_bins), float)

    # ---- 3) 每个 bin 单独算 lambda ----
    for bi, b in enumerate(uniq_bins):
        mask = (bin_id == b)
        if mask.sum() < min_pts:
            lam_bin[bi] = lam_global
            print("退回全局lambda")
            continue

        common_b = common[mask]
        abs_b = abs_s.loc[common_b].values
        res_b = res_s.loc[common_b].values

        lam_b, _ = compute_lambda_multi_signal(
            df_feat=df_feat,
            common=common_b,
            abs_p50=abs_b,
            res_p50=res_b,
            ref_pack=ref_pack,
            k=k
        )
        lam_bin[bi] = lam_b

    # ---- 4) bin rolling 平滑 ----
    if smooth_bins and smooth_bins > 1 and len(uniq_bins) >= smooth_bins:
        lam_bin = pd.Series(lam_bin).rolling(smooth_bins, center=True, min_periods=1).mean().values

    # ---- 5) 向全局收缩，防抖 ----
    lam_bin = (1 - lambda_shrink) * lam_bin + lambda_shrink * lam_global
    lam_bin = np.clip(lam_bin, 0.0, 1.0)

    # ---- 6) 映射回点级 lam_vec ----
    bin_to_row = {b: i for i, b in enumerate(uniq_bins)}
    lam_vec = np.zeros(len(common), float)
    for i in range(len(common)):
        lam_vec[i] = lam_bin[bin_to_row[bin_id[i]]]

    info = {
        "lam_global": float(lam_global),
        "lam_bin": lam_bin,
        "uniq_bins": uniq_bins,
        "bin_id": bin_id,
        "lam_info_global": lam_info_global
    }
    return lam_vec, info

def compute_auto_lambda_reference(
    dfA_feat: pd.DataFrame,
    experts_abs, ood_abs,
    experts_res, ood_res,
    alpha: float, beta: float
):
    """
    在训练井A上计算 auto-λ 所需的参考尺度（训练阶段可用DTCO）
    """
    ref_vel = velocity_consistency_score(dfA_feat)

    dfA_r = dfA_feat.dropna(subset=["DTCO", "DTCO_PROXY_LF"]).copy()
    rA = (dfA_r["DTCO"].values.astype(float) - dfA_r["DTCO_PROXY_LF"].values.astype(float))
    rA_med = float(np.median(rA))
    rA_iqr = float(np.quantile(rA, 0.75) - np.quantile(rA, 0.25))
    if (not np.isfinite(rA_iqr)) or rA_iqr <= 1e-9:
        rA_iqr = 1.0

    resid_ref_local = {"rA_med": rA_med, "rA_iqr": rA_iqr}

    common_res, pred_res, _, _, _ = run_ensemble(
        "Well_A_RES_REF", dfA_feat, experts_res, ood_res, alpha, beta,
        verbose=False, domain="res", resid_ref=resid_ref_local
    )

    common_abs, pred_abs, _, _, _ = run_ensemble(
        "Well_A_ABS_REF", dfA_feat, experts_abs, ood_abs, alpha, beta,
        verbose=False, domain="abs"
    )
    common = common_abs.intersection(common_res).sort_values()

    abs_p50 = align_to_common(common_abs, pred_abs[0.5], common)
    res_p50 = align_to_common(common_res, pred_res[0.5], common)
    proxy_lf = dfA_feat.loc[common, "DTCO_PROXY_LF"].values.astype(float)

    rhat_A = res_p50 - proxy_lf
    m = np.isfinite(rhat_A)
    ref_resid = float(np.median(np.abs((rhat_A[m] - rA_med) / rA_iqr))) if m.sum() >= 50 else 1.0
    if (not np.isfinite(ref_resid)) or ref_resid <= 1e-9:
        ref_resid = 1.0

    abs_lf_A = pd.Series(abs_p50).rolling(AUTO_LAMBDA_ABS_LF_WIN, center=True, min_periods=1).median().values
    mm = np.isfinite(abs_lf_A) & np.isfinite(proxy_lf)
    ref_proxyabs = float(np.median(np.abs(abs_lf_A[mm] - proxy_lf[mm]))) if mm.sum() >= 50 else 1.0
    if (not np.isfinite(ref_proxyabs)) or ref_proxyabs <= 1e-9:
        ref_proxyabs = 1.0

    return {
        "ref_vel": ref_vel,
        "rA_med": rA_med,
        "rA_iqr": rA_iqr,
        "ref_resid": ref_resid,
        "ref_proxyabs": ref_proxyabs
    }


# =========================
# 保存输出（Hybrid）
# =========================
def save_hybrid(name, df_feat, common, preds_final, out_dir, title, lambda_val,
                w_abs, w_res, experts_abs, experts_res):
    y_true = df_feat.loc[common, "DTCO"].values.astype(float)
    depth = df_feat.loc[common, "TDEP"].values.astype(float)

    out = pd.DataFrame({
        "ROW_ID": common.values,
        "TDEP": depth,
        "DTCO_TRUE": y_true,
        "DTCO_P10": preds_final[0.1],
        "DTCO_P50": preds_final[0.5],
        "DTCO_P90": preds_final[0.9],
        "LAMBDA_RESID": lambda_val,
    }).sort_values("TDEP")

    for j, e in enumerate(experts_abs):
        out[f"W_ABS_{e.name.upper()}"] = w_abs[:, j]
    for j, e in enumerate(experts_res):
        out[f"W_RES_{e.name.upper()}"] = w_res[:, j]

    csv_path = os.path.join(out_dir, f"{name}_dtco_pred.csv")
    fig_path = os.path.join(out_dir, f"{name}_dtco_pred.png")
    out.to_csv(csv_path, index=False)

    plot_dtco(
        out["TDEP"].values, out["DTCO_TRUE"].values,
        out["DTCO_P50"].values, out["DTCO_P10"].values, out["DTCO_P90"].values,
        title, fig_path, smooth=PLOT_SMOOTH, win=PLOT_SMOOTH_WIN
    )

    print(f"{name} 预测CSV已保存：{csv_path}")
    print(f"{name} 预测图已保存：{fig_path}")


def resid_output_ood_score(p50_pred, proxy_lf, r_med, r_iqr):
    """
    基于输出残差的 OOD：
    rhat = p50_pred - proxy_lf
    score = median( abs((rhat - r_med)/r_iqr) )
    """
    p50_pred = np.asarray(p50_pred, float)
    proxy_lf = np.asarray(proxy_lf, float)
    rhat = p50_pred - proxy_lf

    if (not np.isfinite(r_iqr)) or r_iqr <= 1e-9:
        r_iqr = 1.0

    m = np.isfinite(rhat)
    if m.sum() < 50:
        return float(np.inf)
    return float(np.median(np.abs((rhat[m] - r_med) / r_iqr)))

# =========================
# 主程序
# =========================
def main():
    A_PATH = os.path.expanduser(r"D:\文件\博士\项目\面向深海资源勘探的环境融合感知与智能协同控制\研究\钻前孔隙压力预测\数据\投TII更新后数据\wellA_pr_pp.csv")
    B_PATH = os.path.expanduser(r"~/Desktop/wellD_pr_pp.csv")
    C_PATH = os.path.expanduser(r"~/Desktop/wellE_pr_pp.csv")
    # A_PATH = os.path.expanduser(r"./well_A_processed.csv")
    # B_PATH = os.path.expanduser(r"./well_B_processed.csv")
    # C_PATH = os.path.expanduser(r"./well_C_processed.csv")
    out_dir = os.path.expanduser(r"./pp")

    dfA = coerce_numeric(pd.read_csv(A_PATH));  # ensure_columns(dfA, "Well_A")
    dfB = coerce_numeric(pd.read_csv(B_PATH));  # ensure_columns(dfB, "Well_B")
    dfC = coerce_numeric(pd.read_csv(C_PATH));  # ensure_columns(dfC, "Well_C")

    dfA_feat = add_features_keep_rows(dfA, proxy_lf_win=201)
    dfB_feat = add_features_keep_rows(dfB, proxy_lf_win=201)
    dfC_feat = add_features_keep_rows(dfC, proxy_lf_win=201)

    # ===== 训练 ABS 专家组（仍然只用A训练）=====
    v4_abs  = TrendResidualQuantile(TREND_V4, name="v4_abs", base_col=None, clip_y=True).fit(dfA_feat)
    xgb_abs = XGBExpertABS().fit(dfA_feat)
    seq_abs = SeqExpertABS().fit(dfA_feat)
    experts_abs = [v4_abs, xgb_abs, seq_abs]

    # ===== 训练 RES 专家组：用“不同模型类型”替换原 xgb_res / v4_res =====
    ridge_res = RidgeExpert(features=RIDGE_FEATURES_RES, name="ridge_res", base_col="DTCO_PROXY_LF").fit(dfA_feat)
    gbdt_res  = GBDTQuantileExpert(features=GBDT_FEATURES_RES, name="gbdt_res", base_col="DTCO_PROXY_LF").fit(dfA_feat)
    seq_res   = SeqExpertRES().fit(dfA_feat)
    experts_res = [ridge_res, gbdt_res, seq_res]

    # ===== OOD统计（A上）=====
    A_med_v4, A_iqr_v4 = robust_stats(dfA_feat, TREND_V4)
    A_med_xg_abs, A_iqr_xg_abs = robust_stats(dfA_feat, XGB_FEATURES_ABS)

    SEQ_OOD_ABS = ["TDEP","TT","TWT","Vave","Vrms","TT Gradient","FLOWIN","DTCO_PROXY","TT_GRAD_RE","TWT_GRAD_RE"]
    SEQ_OOD_RES = SEQ_OOD_ABS + ["DTCO_PROXY_LF"]

    A_med_seq_abs, A_iqr_seq_abs = robust_stats(dfA_feat, SEQ_OOD_ABS)
    A_med_seq_res, A_iqr_seq_res = robust_stats(dfA_feat, SEQ_OOD_RES)

    # NEW: RES 专家对应的 OOD 统计
    A_med_ridge_res, A_iqr_ridge_res = robust_stats(dfA_feat, RIDGE_FEATURES_RES)
    A_med_gbdt_res,  A_iqr_gbdt_res  = robust_stats(dfA_feat, GBDT_FEATURES_RES)

    ood_abs = [
        (TREND_V4, A_med_v4, A_iqr_v4),
        (XGB_FEATURES_ABS, A_med_xg_abs, A_iqr_xg_abs),
        (SEQ_OOD_ABS, A_med_seq_abs, A_iqr_seq_abs),
    ]
    ood_res = [
        (RIDGE_FEATURES_RES, A_med_ridge_res, A_iqr_ridge_res),
        (GBDT_FEATURES_RES,  A_med_gbdt_res,  A_iqr_gbdt_res),
        (SEQ_OOD_RES, A_med_seq_res, A_iqr_seq_res),
    ]

    # ===== alpha/beta =====
    alpha, beta = ALPHA0, BETA0
    if TUNE_ALPHA_BETA:
        best = (1e18, alpha, beta)
        for a in ALPHA_GRID:
            for b in BETA_GRID:
                _, _, _, _, (_, rb, _, _) = run_ensemble("B_abs_tmp", dfB_feat, experts_abs, ood_abs, a, b, verbose=False)
                _, _, _, _, (_, rc, _, _) = run_ensemble("C_res_tmp", dfC_feat, experts_res, ood_res, a, b, verbose=False)
                obj = rb + rc
                if obj < best[0]:
                    best = (obj, a, b)
        _, alpha, beta = best
        print(f"[Tuned alpha/beta] alpha={alpha}, beta={beta}")

    # ===== auto lambda 参考尺度（来自训练井A）=====
    ref_pack = compute_auto_lambda_reference(dfA_feat, experts_abs, ood_abs, experts_res, ood_res, alpha, beta)
    resid_ref = {"rA_med": ref_pack["rA_med"], "rA_iqr": ref_pack["rA_iqr"]}
    print(
        f"\n[Auto λ refs] "
        f"ref_vel(A)={ref_pack['ref_vel']:.6f}, "
        f"ref_resid(A)={ref_pack['ref_resid']:.4f}, "
        f"ref_proxyabs(A)={ref_pack['ref_proxyabs']:.4f}, "
        f"rA_med={ref_pack['rA_med']:.4f}, rA_iqr={ref_pack['rA_iqr']:.4f}"
    )

    # ===== 逐井跑 Hybrid =====
    for name, df_feat in [("Well_A", dfA_feat), ("Well_B", dfB_feat), ("Well_C", dfC_feat)]:
        # ABS 预测
        common_abs, pred_abs, _, w_abs, (mae_abs, rmse_abs, mape_abs, r2_abs) = run_ensemble(
            f"{name}_ABS", df_feat, experts_abs, ood_abs, alpha, beta,
            verbose=True, domain="abs"
        )
        # RES 预测
        common_res, pred_res, _, w_res, (mae_res, rmse_res, mape_res, r2_res) = run_ensemble(
            f"{name}_RES", df_feat, experts_res, ood_res, alpha, beta,
            verbose=True, domain="res", resid_ref=resid_ref
        )

        # 共同可用点
        common = common_abs.intersection(common_res).sort_values()

        # 对齐
        abs_q = {q: align_to_common(common_abs, pred_abs[q], common) for q in QUANTILES}
        res_q = {q: align_to_common(common_res, pred_res[q], common) for q in QUANTILES}

        # λ（改为：标量 lam 或随深度变化的 lam_vec）
        if LAMBDA_MODE == "manual":
            lam_scalar = float(LAMBDA_MANUAL.get(name, AUTO_LAMBDA_FALLBACK))
            lam_vec = np.full(len(common), lam_scalar, dtype=float)
            lam_info = {"lam_global": lam_scalar, "mode": "manual"}
            print(f"[{name}] manual lambda_resid={lam_scalar:.3f}")

        else:
            if LAMBDA_DEPTH_VARYING:
                lam_vec, lam_info = compute_lambda_by_depth_bins(
                    df_feat=df_feat,
                    common=common,
                    abs_p50=abs_q[0.5],
                    res_p50=res_q[0.5],
                    ref_pack=ref_pack,
                    k=AUTO_LAMBDA_K,
                    bin_size=LAMBDA_BIN_SIZE,
                    min_pts=LAMBDA_MIN_PTS,
                    smooth_bins=LAMBDA_SMOOTH_BINS,
                    lambda_shrink=LAMBDA_SHRINK
                )
                lam_scalar = float(lam_info["lam_global"])  # 用于打印/标题
                gi = lam_info.get("lam_info_global", None)
                if gi is not None:
                    print(
                        f"[{name}] auto(global) scores: "
                        f"vel={gi['vel']:.6f}(x{gi['rv']:.2f}), "
                        f"residOOD={gi['resid']:.4f}(x{gi['rr']:.2f}), "
                        f"proxyABS={gi['proxyabs']:.4f}(x{gi['rp']:.2f}), "
                        f"S={gi['S']:.3f} -> lambda_resid(global)={lam_scalar:.3f}"
                    )
                print(f"[{name}] depth-varying lambda: mean={lam_vec.mean():.3f}, std={lam_vec.std():.3f}, min={lam_vec.min():.3f}, max={lam_vec.max():.3f}")

            else:
                lam_scalar, lam_info = compute_lambda_multi_signal(
                    df_feat=df_feat,
                    common=common,
                    abs_p50=abs_q[0.5],
                    res_p50=res_q[0.5],
                    ref_pack=ref_pack,
                    k=AUTO_LAMBDA_K
                )
                lam_vec = np.full(len(common), float(lam_scalar), dtype=float)

                if lam_info is not None:
                    print(
                        f"[{name}] auto scores: "
                        f"vel={lam_info['vel']:.6f}(x{lam_info['rv']:.2f}), "
                        f"residOOD={lam_info['resid']:.4f}(x{lam_info['rr']:.2f}), "
                        f"proxyABS={lam_info['proxyabs']:.4f}(x{lam_info['rp']:.2f}), "
                        f"S={lam_info['S']:.3f} -> lambda_resid={lam_scalar:.3f}"
                    )
                else:
                    print(f"[{name}] auto lambda_resid={lam_scalar:.3f}")

        # 混合（用 lam_vec 做逐点融合）
        pred_final = {q: (1 - lam_vec) * abs_q[q] + lam_vec * res_q[q] for q in QUANTILES}

        stacked = np.vstack([pred_final[0.1], pred_final[0.5], pred_final[0.9]])
        stacked.sort(axis=0)
        pred_final[0.1], pred_final[0.5], pred_final[0.9] = stacked[0], stacked[1], stacked[2]

        # window_length 必须是奇数，polyorder 通常为 2 或 3
        pred_final[0.5] = savgol_filter(pred_final[0.5], window_length=51, polyorder=3)

        # Hybrid 评价（四指标）
        y_true = df_feat.loc[common, "DTCO"].values.astype(float)
        mae_h, rmse_h, mape_h, r2_h = metrics(y_true, pred_final[0.5])

        print(f"\n===== {name} HYBRID =====")
        print(f"lambda_resid(global)={lam_scalar:.3f} | lambda(z): mean={lam_vec.mean():.3f}, std={lam_vec.std():.3f}")
        print(f"ABS:   MAE={mae_abs:.4f}, RMSE={rmse_abs:.4f}, MAPE={mape_abs:.2f}%, R2={r2_abs:.4f}")
        print(f"RES:   MAE={mae_res:.4f}, RMSE={rmse_res:.4f}, MAPE={mape_res:.2f}%, R2={r2_res:.4f}")
        print(f"HYBRID MAE={mae_h:.4f}, RMSE={rmse_h:.4f}, MAPE={mape_h:.2f}%, R2={r2_h:.4f}")

        # 保存输出（权重点对点输出）
        idx_abs_in_common = common_abs.get_indexer(common)
        idx_res_in_common = common_res.get_indexer(common)
        w_abs_c = w_abs[idx_abs_in_common, :]
        w_res_c = w_res[idx_res_in_common, :]

        title = f"{name}: HYBRID MAE={mae_h:.2f}, RMSE={rmse_h:.2f}, MAPE={mape_h:.2f}%, R2={r2_h:.3f}, λg={lam_scalar:.2f}"

        save_hybrid(
            name=name,
            df_feat=df_feat,
            common=common,
            preds_final=pred_final,
            out_dir=out_dir,
            title=title,
            lambda_val=lam_vec,          # ★ 这里从标量改成向量
            w_abs=w_abs_c,
            w_res=w_res_c,
            experts_abs=experts_abs,
            experts_res=experts_res
        )

    # 保存模型包
    bundle = {
        "experts_abs": experts_abs,
        "experts_res": experts_res,
        "ood_abs": ood_abs,
        "ood_res": ood_res,
        "alpha": alpha,
        "beta": beta,
        "lambda_mode": LAMBDA_MODE,
        "lambda_manual": LAMBDA_MANUAL,
        "auto_lambda_k": AUTO_LAMBDA_K,
        "auto_lambda_weights": AUTO_LAMBDA_WEIGHTS,
        "auto_lambda_abs_lf_win": AUTO_LAMBDA_ABS_LF_WIN,
        "auto_lambda_refs": ref_pack,
        "bin_size": BIN_SIZE,
        "min_pts": MIN_PTS,
        "smooth_bins": SMOOTH_BINS,
        "lambda_shrink": LAMBDA_SHRINK,
        "use_binned_p10p90": USE_BINNED_P10P90,
        "p10p90_bin_size": P10P90_BIN_SIZE,
        "p10p90_min_count": P10P90_MIN_COUNT
    }
    model_path = os.path.join(out_dir, "hybrid_abs_resid_modified.joblib")
    joblib.dump(bundle, model_path)
    print(f"\n模型已保存：{model_path}")


if __name__ == "__main__":
    main()
