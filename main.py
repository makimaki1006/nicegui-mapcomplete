# -*- coding: utf-8 -*-
# Force reload: 2025-12-25 v8 - Fix market overview: use selected region's age_distribution
"""NiceGUI starter dashboard for migrating from Reflex.

Includes:
- simple email/password check with allowed domains
- Turso -> CSV fallback data loading (summary rows only)
- prefecture/municipality filters
- overview & demographics tabs with basic charts
"""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import List, Dict, Any

import httpx
import pandas as pd
from nicegui import app, ui

# メモリ最適化: 起動時にガベージコレクション
gc.collect()

# db_helper.pyをインポート（Reflexと同じデータアクセスロジック）
try:
    from db_helper import (
        _load_csv_data,
        get_national_stats,
        get_prefecture_stats,
        get_all_prefectures_stats,
        get_municipality_stats,
        get_talent_flow,
        get_distance_stats,
        get_persona_market_share,
        get_qualification_retention_rates,
        get_age_gender_stats,
        get_rarity_analysis,
        get_qualification_options,
        get_persona_employment_breakdown,
        get_qualification_by_gender,
        get_flow_sources,
        get_flow_destinations,
        get_competition_overview,
        get_mobility_type_distribution,
        get_pref_flow_top10,
        get_muni_flow_top10,
        PREFECTURE_ORDER as DB_PREFECTURE_ORDER,
    )
    _DB_HELPER_AVAILABLE = True
    print("[STARTUP] db_helper.py loaded successfully")
except ImportError as e:
    _DB_HELPER_AVAILABLE = False
    print(f"[STARTUP] db_helper.py import failed: {e}")
    # フォールバック用ダミー関数
    _load_csv_data = lambda: pd.DataFrame()
    get_national_stats = lambda: {}
    get_prefecture_stats = lambda pref: {}
    get_all_prefectures_stats = lambda: {}
    get_municipality_stats = lambda pref, muni: {}
    get_talent_flow = lambda pref=None, muni=None: {}
    get_distance_stats = lambda pref=None, muni=None: {}
    get_persona_market_share = lambda pref=None, muni=None: []
    get_qualification_retention_rates = lambda pref=None, muni=None: []
    get_age_gender_stats = lambda pref=None, muni=None: []
    get_rarity_analysis = lambda pref=None, muni=None, ages=None, genders=None, qualifications=None: []
    get_qualification_options = lambda pref=None, muni=None: []
    get_persona_employment_breakdown = lambda pref=None, muni=None: []
    get_qualification_by_gender = lambda pref=None, muni=None: []
    get_flow_sources = lambda pref=None, muni=None, limit=10: []
    get_flow_destinations = lambda pref=None, muni=None, limit=10: []
    get_competition_overview = lambda pref=None, muni=None: {}
    get_mobility_type_distribution = lambda pref=None, muni=None: []
    get_pref_flow_top10 = lambda pref=None: []
    get_muni_flow_top10 = lambda pref=None, muni=None: []
    DB_PREFECTURE_ORDER = []

def log(msg: str) -> None:
    """centralized stdout logger (flush immediately)"""
    print(msg, flush=True)

# NiceGUI <=1.4 compatibility: some releases expose only on_value_change
try:  # pragma: no cover
    from nicegui.elements.select import Select

    if not hasattr(Select, "on_change") and hasattr(Select, "on_value_change"):
        Select.on_change = Select.on_value_change  # type: ignore[attr-defined]
        log("[STARTUP] Added Select.on_change alias for compatibility")
except Exception as exc:  # pragma: no cover
    log(f"[STARTUP] Select compatibility check skipped: {exc}")

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    reflex_env = Path(__file__).parent.parent / "reflex_app" / ".env.production"

    if local_env.exists():
        load_dotenv(local_env)
        log(f"[STARTUP] Loaded: {local_env}")
    elif reflex_env.exists():
        load_dotenv(reflex_env)
        log(f"[STARTUP] Loaded: {reflex_env}")
    else:
        log("[STARTUP] No .env file found, using system environment variables")
except Exception as exc:  # pragma: no cover
    log(f"[STARTUP] dotenv error: {exc}")

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "cyzen_2025")
ALLOWED_DOMAINS = [d.strip() for d in os.getenv("ALLOWED_DOMAINS", "f-a-c.co.jp,cyxen.co.jp").split(",")]

# Prefecture ordering (JIS 北→南)
PREFECTURE_ORDER = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]

# Style - Okabe-Ito Color Palette (色弱対応)
# Reference: https://jfly.uni-koeln.de/color/
BG_COLOR = "#0d1525"                        # 深いネイビー基調
PANEL_BG = "rgba(12, 20, 37, 0.95)"
CARD_BG = "rgba(15, 23, 42, 0.82)"
TEXT_COLOR = "#f8fafc"                      # 文字色
MUTED_COLOR = "rgba(226, 232, 240, 0.75)"   # 薄い文字色
BORDER_COLOR = "rgba(148, 163, 184, 0.22)"  # 境界線

# Okabe-Ito カラーパレット（色弱対応）
PRIMARY_COLOR = "#0072B2"                   # 濃い青（Blue）
SECONDARY_COLOR = "#E69F00"                 # オレンジ（Orange）- 赤緑色弱でも識別◎
ACCENT_PINK = "#CC79A7"                     # 赤紫（Reddish Purple）- 明度高く識別◎
ACCENT_GREEN = "#009E73"                    # 青緑（Bluish Green）- 赤緑色弱でも識別◎
ACCENT_YELLOW = "#F0E442"                   # 黄色（Yellow）- 明度最高、視認性◎
ACCENT_VERMILLION = "#D55E00"               # 朱色（Vermillion）- 赤緑色弱でも識別◎
ACCENT_SKY = "#56B4E9"                      # スカイブルー（Sky Blue）- 明度高く識別◎

# チャート用カラーパレット（Okabe-Ito順序）
COLOR_PALETTE = ['#0072B2', '#E69F00', '#CC79A7', '#009E73', '#F0E442', '#D55E00', '#56B4E9']

# インデックスベースのアクセント色（後方互換）
ACCENT_4 = COLOR_PALETTE[3]  # #009E73 青緑
ACCENT_5 = COLOR_PALETTE[4]  # #F0E442 黄色
ACCENT_6 = COLOR_PALETTE[5]  # #D55E00 朱色
ACCENT_7 = COLOR_PALETTE[6]  # #56B4E9 スカイブルー

# 意味的カラー
WARNING_COLOR = ACCENT_VERMILLION           # 朱色（警告用）
SUCCESS_COLOR = ACCENT_GREEN                # 青緑（成功用）
INFO_COLOR = ACCENT_SKY                     # スカイブルー（情報用）

# Data files
CSV_PATH = Path(__file__).parent.parent / "reflex_app" / "MapComplete_Complete_All_FIXED.csv"
CSV_PATH_GZ = Path(__file__).parent.parent / "reflex_app" / "MapComplete_Complete_All_FIXED.csv.gz"
CSV_PATH_ALT = (
    Path(__file__).parent.parent
    / "python_scripts"
    / "data"
    / "output_v2"
    / "mapcomplete_complete_sheets"
    / "MapComplete_Complete_All_FIXED.csv"
)

_dataframe: pd.DataFrame | None = None
_data_source: str = "not loaded"


# ---------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------
def query_turso(sql: str) -> pd.DataFrame:
    """Run a Turso HTTP query."""
    log(f"[TURSO] query_turso called with SQL: {sql[:100]}...")
    http_url = TURSO_DATABASE_URL
    if http_url.startswith("libsql://"):
        http_url = http_url.replace("libsql://", "https://")
    log(f"[TURSO] HTTP URL: {http_url[:50]}...")

    headers = {
        "Authorization": f"Bearer {TURSO_AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"requests": [{"type": "execute", "stmt": {"sql": sql}}]}

    response = httpx.post(f"{http_url}/v2/pipeline", headers=headers, json=payload, timeout=30.0)
    if response.status_code != 200:
        raise RuntimeError(f"Turso HTTP {response.status_code}: {response.text}")

    data = response.json()
    if not data.get("results"):
        return pd.DataFrame()

    result = data["results"][0]
    if result.get("type") == "error":
        msg = result.get("error", {}).get("message", "Unknown error")
        raise RuntimeError(f"Turso query error: {msg}")

    resp = result["response"]["result"]
    columns = [c["name"] for c in resp["cols"]]
    rows = []
    for row in resp["rows"]:
        row_dict = {}
        for idx, col in enumerate(columns):
            val = row[idx]
            row_dict[col] = val.get("value") if isinstance(val, dict) else val
        rows.append(row_dict)
    return pd.DataFrame(rows, columns=columns)


def load_data() -> pd.DataFrame:
    """Load data from Turso first, then fallback to CSV; cache the result."""
    global _dataframe, _data_source
    print("[DATA] load_data() called", flush=True)
    if _dataframe is not None:
        print("[DATA] Returning cached dataframe", flush=True)
        return _dataframe

    print(f"[DATA] TURSO_DATABASE_URL: {bool(TURSO_DATABASE_URL)}, TURSO_AUTH_TOKEN: {bool(TURSO_AUTH_TOKEN)}", flush=True)
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            print("[DATA] Attempting Turso load...", flush=True)
            log("[DATA] Loading from Turso (SUMMARY only)...")
            _dataframe = query_turso("SELECT * FROM job_seeker_data WHERE row_type = 'SUMMARY'")
            _data_source = "Turso DB"
            print(f"[DATA] Turso SUCCESS: {len(_dataframe):,} rows", flush=True)
            log(f"[DATA] Loaded {len(_dataframe):,} rows from Turso")
            return _dataframe
        except Exception as exc:
            print(f"[DATA] Turso FAILED: {type(exc).__name__}: {exc}", flush=True)
            log(f"[DATA] Turso failed: {type(exc).__name__}: {exc}")
            log("[DATA] Falling back to CSV...")

    for path in [CSV_PATH_GZ, CSV_PATH, CSV_PATH_ALT]:
        if path.exists():
            log(f"[DATA] Loading from CSV: {path}")
            if path.suffix == ".gz":
                _dataframe = pd.read_csv(path, encoding="utf-8-sig", compression="gzip", low_memory=False)
            else:
                _dataframe = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            _data_source = f"CSV ({path.name})"
            log(f"[DATA] Loaded {len(_dataframe):,} rows from CSV")
            return _dataframe

    _data_source = "no data source"
    log("[DATA] No data source found")
    return pd.DataFrame()


# GAP data cache
_gap_dataframe: pd.DataFrame | None = None


def load_gap_data() -> pd.DataFrame:
    """Load GAP row_type data from Turso for supply/demand balance analysis."""
    global _gap_dataframe
    if _gap_dataframe is not None:
        return _gap_dataframe

    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            log("[DATA] Loading GAP data from Turso...")
            _gap_dataframe = query_turso("SELECT * FROM job_seeker_data WHERE row_type = 'GAP'")
            log(f"[DATA] Loaded {len(_gap_dataframe):,} GAP rows from Turso")
            # Ensure numeric columns
            for col in ["demand_count", "supply_count", "gap", "demand_supply_ratio"]:
                if col in _gap_dataframe.columns:
                    _gap_dataframe[col] = pd.to_numeric(_gap_dataframe[col], errors="coerce")
            return _gap_dataframe
        except Exception as exc:
            log(f"[DATA] GAP data load failed: {exc}")
            return pd.DataFrame()
    return pd.DataFrame()


def get_gap_stats(pref: str | None = None, muni: str | None = None) -> dict:
    """Get supply/demand gap statistics for the balance tab (Reflex完全再現)."""
    gap_df = load_gap_data()
    if gap_df.empty:
        return {"demand": 0, "supply": 0, "gap": 0, "ratio": 0, "shortage_count": 0, "surplus_count": 0}

    filtered = gap_df.copy()
    if pref and pref != "全国" and "prefecture" in filtered.columns:
        filtered = filtered[filtered["prefecture"] == pref]
    if muni and muni != "すべて" and "municipality" in filtered.columns:
        filtered = filtered[filtered["municipality"] == muni]

    if filtered.empty:
        return {"demand": 0, "supply": 0, "gap": 0, "ratio": 0, "shortage_count": 0, "surplus_count": 0}

    demand = float(filtered["demand_count"].fillna(0).sum()) if "demand_count" in filtered.columns else 0
    supply = float(filtered["supply_count"].fillna(0).sum()) if "supply_count" in filtered.columns else 0
    gap_val = float(filtered["gap"].fillna(0).sum()) if "gap" in filtered.columns else 0
    ratio = (demand / supply) if supply > 0 else (demand if demand > 0 else 0)

    # 不足地域（需要 > 供給）、過剰地域（供給 > 需要）のカウント
    shortage_count = 0
    surplus_count = 0
    if "demand_count" in filtered.columns and "supply_count" in filtered.columns:
        shortage_count = int((filtered["demand_count"].fillna(0) > filtered["supply_count"].fillna(0)).sum())
        surplus_count = int((filtered["supply_count"].fillna(0) > filtered["demand_count"].fillna(0)).sum())

    return {"demand": demand, "supply": supply, "gap": gap_val, "ratio": ratio,
            "shortage_count": shortage_count, "surplus_count": surplus_count}


def get_gap_rankings(pref: str | None = None, limit: int = 10) -> dict:
    """需給ランキング取得（都道府県内の全市区町村）

    NOTE: 市区町村を選択しても、同じ都道府県内のランキングが表示される（Reflex版仕様）。
    """
    gap_df = load_gap_data()
    if gap_df.empty:
        return {"shortage": [], "surplus": [], "ratio": []}

    # 都道府県フィルタのみ（市区町村は無視＝都道府県内全体のランキング）
    filtered = gap_df.copy()
    if pref and pref != "全国" and "prefecture" in filtered.columns:
        filtered = filtered[filtered["prefecture"] == pref]

    if filtered.empty or "municipality" not in filtered.columns:
        return {"shortage": [], "surplus": [], "ratio": []}

    # 需要超過ランキング（gap > 0 で大きい順）
    shortage_ranking = []
    if "gap" in filtered.columns:
        shortage_df = filtered[filtered["gap"].fillna(0) > 0].copy()
        shortage_df = shortage_df.nlargest(limit, "gap")
        shortage_ranking = [
            {"name": str(row.get("municipality", "不明")), "value": float(row.get("gap", 0))}
            for _, row in shortage_df.iterrows()
        ]

    # 供給超過ランキング（gap < 0 で小さい順＝絶対値大きい順）
    surplus_ranking = []
    if "gap" in filtered.columns:
        surplus_df = filtered[filtered["gap"].fillna(0) < 0].copy()
        surplus_df = surplus_df.nsmallest(limit, "gap")
        surplus_ranking = [
            {"name": str(row.get("municipality", "不明")), "value": abs(float(row.get("gap", 0)))}
            for _, row in surplus_df.iterrows()
        ]

    # 需給比率ランキング（ratio 大きい順）
    ratio_ranking = []
    if "demand_supply_ratio" in filtered.columns:
        ratio_df = filtered.dropna(subset=["demand_supply_ratio"]).copy()
        ratio_df = ratio_df.nlargest(limit, "demand_supply_ratio")
        ratio_ranking = [
            {"name": str(row.get("municipality", "不明")), "value": float(row.get("demand_supply_ratio", 0))}
            for _, row in ratio_df.iterrows()
        ]

    return {"shortage": shortage_ranking, "surplus": surplus_ranking, "ratio": ratio_ranking}


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns/values and keep SUMMARY rows only."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "row_type" in df.columns:
        df = df[df["row_type"] == "SUMMARY"]

    if "prefecture" in df.columns:
        df["prefecture"] = df["prefecture"].astype(str).str.strip()
        df = df[df["prefecture"].astype(bool)]

    if "municipality" in df.columns:
        df["municipality"] = df["municipality"].astype(str).str.strip()

    # convert numeric columns safely
    numeric_cols = [
        "applicant_count", "male_count", "female_count", "avg_age",
        "avg_qualifications", "avg_qualification_count", "count",
        "avg_desired_areas", "national_license_rate", "total_in_municipality",
        "market_share_pct", "avg_mobility_score", "avg_urgency_score",
        "inflow", "outflow", "net_flow", "demand_count", "supply_count",
        "gap", "demand_supply_ratio", "rarity_score", "retention_rate",
        "avg_reference_distance_km", "median_distance_km", "mode_distance_km",
        "min_distance_km", "max_distance_km", "std_distance_km",
        "q25_distance_km", "q75_distance_km", "total_applicants",
        "top_age_ratio", "female_ratio", "male_ratio", "top_employment_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log(f"[DATA] Cleaned dataframe rows: {len(df):,}")
    return df.reset_index(drop=True)


def safe_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].fillna(0).sum()) if col in df else 0.0


def safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].dropna().mean()) if col in df and not df[col].dropna().empty else 0.0


def safe_max(df: pd.DataFrame, col: str) -> float:
    return float(df[col].dropna().max()) if col in df and not df[col].dropna().empty else 0.0


def safe_min(df: pd.DataFrame, col: str) -> float:
    return float(df[col].dropna().min()) if col in df and not df[col].dropna().empty else 0.0


def top_categories(df: pd.DataFrame, col: str, weight_col: str = "applicant_count", limit: int = 10):
    """Return sorted list of (label, value) by summed weight_col."""
    if col not in df.columns or df.empty:
        return []
    if weight_col in df.columns:
        agg = df.groupby(col)[weight_col].sum().sort_values(ascending=False)
    else:
        agg = df[col].value_counts()
    agg = agg.dropna()
    return list(zip(agg.index.tolist()[:limit], agg.values.tolist()[:limit]))


# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------
def is_authenticated() -> bool:
    return app.storage.user.get("authenticated", False)


def get_user_email() -> str:
    return app.storage.user.get("email", "")


def verify_login(email: str, password: str) -> tuple[bool, str]:
    if not email or not password:
        return False, "メールアドレスとパスワードを入力してください"
    if "@" not in email:
        return False, "有効なメールアドレスを入力してください"

    domain = email.split("@")[1].lower()
    if domain not in [d.lower() for d in ALLOWED_DOMAINS]:
        return False, f"ドメイン {domain} は許可されていません"
    if password != AUTH_PASSWORD:
        return False, "パスワードが正しくありません"
    return True, ""


# ---------------------------------------------------------------------
# Health check endpoint (Render用 - 5秒以内に応答必須)
# ---------------------------------------------------------------------
from fastapi import Response

@app.get("/health")
async def health_check():
    """Renderヘルスチェック用の軽量エンドポイント"""
    return Response(content="OK", media_type="text/plain")


# ---------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------
@ui.page("/login")
def login_page() -> None:
    if is_authenticated():
        ui.navigate.to("/")
        return

    ui.query("body").style(f"background-color: {BG_COLOR}")

    with ui.card().classes("absolute-center w-96").style(
        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}"
    ):
        ui.label("MapComplete Dashboard").classes("text-2xl font-bold text-center w-full mb-4").style(
            f"color: {TEXT_COLOR}"
        )
        ui.label("ログイン").classes("text-lg text-center w-full mb-4").style(f"color: {MUTED_COLOR}")

        email_input = ui.input("メールアドレス", placeholder="user@example.com").classes("w-full").props('dark input-style="color: white"')
        password_input = ui.input("パスワード", password=True, password_toggle_button=True).classes("w-full").props('dark input-style="color: white"')
        error_label = ui.label("").classes("text-red-500 text-sm")

        def handle_login() -> None:
            email = email_input.value
            password = password_input.value
            success, message = verify_login(email, password)
            if success:
                app.storage.user["authenticated"] = True
                app.storage.user["email"] = email
                print(f"[AUTH] Login success: {email}")
                ui.navigate.to("/")
            else:
                error_label.text = message
                print(f"[AUTH] Login failed: {message}")

        ui.button("サインイン", on_click=handle_login).classes("w-full mt-4").style(f"background-color: {PRIMARY_COLOR}")
        ui.label("許可ドメイン: " + ", ".join([f"@{d}" for d in ALLOWED_DOMAINS])).classes(
            "text-xs mt-4 text-center w-full"
        ).style(f"color: {MUTED_COLOR}")


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
@ui.page("/")
def dashboard_page() -> None:
    if not is_authenticated():
        ui.navigate.to("/login")
        return

    ui.query("body").style(f"background-color: {BG_COLOR}")

    # カスタムCSS: ドロップダウンを目立たせる
    ui.add_head_html("""
    <style>
    /* 都道府県・市区町村セレクタを目立たせる */
    .dropdown-highlight .q-field__control {
        background-color: rgba(0, 60, 120, 0.95) !important;
        border: 2px solid #00BFFF !important;
        border-radius: 8px !important;
    }
    .dropdown-highlight .q-field__native,
    .dropdown-highlight .q-field__input {
        color: white !important;
    }
    .dropdown-highlight .q-field__label {
        color: #00BFFF !important;
        font-weight: bold !important;
    }
    .dropdown-highlight .q-field--outlined .q-field__control:before {
        border-color: #00BFFF !important;
    }
    .dropdown-highlight .q-field--outlined .q-field__control:hover:before {
        border-color: #56B4E9 !important;
    }
    .dropdown-highlight .q-icon {
        color: #00BFFF !important;
    }
    /* ドロップダウンメニュー */
    .q-menu {
        background-color: #1a2940 !important;
    }
    .q-item {
        color: white !important;
    }
    .q-item:hover {
        background-color: rgba(0, 191, 255, 0.2) !important;
    }
    </style>
    """)

    df = _clean_dataframe(load_data())

    # Build prefecture options with JIS north→south ordering
    # Simple list - browser may sort alphabetically, this is acceptable for now
    prefecture_options: List[str] = ["全国"]
    if "prefecture" in df.columns:
        unique_prefs = [p for p in df["prefecture"].dropna().unique().tolist() if p and p != "全国"]
        order_map = {pref: idx for idx, pref in enumerate(PREFECTURE_ORDER)}
        unique_prefs.sort(key=lambda x: order_map.get(x, len(PREFECTURE_ORDER) + 1))
        prefecture_options.extend(unique_prefs)
        log(f"[DATA] Prefecture options (JIS order in backend): {prefecture_options[:10]} ... total {len(prefecture_options)-1}")
    # Keep reference for validation
    prefectures: List[str] = prefecture_options

    # Municipality dropdown helper - must be inside dashboard_page to access df
    def get_municipality_options(pref_value: str) -> List[str]:
        if pref_value == "全国" or "municipality" not in df.columns:
            return ["すべて"]
        filtered = df[df["prefecture"] == pref_value]
        muni_list = filtered["municipality"].dropna().astype(str).str.strip()
        muni_list = [m for m in muni_list.unique().tolist() if m and m.lower() != "nan"]
        options = ["すべて"] + sorted(muni_list)
        log(f"[DATA] Municipality options for {pref_value}: {options[:10]} ... total {len(options)-1}")
        return options

    state = app.storage.user
    state.setdefault("tab", "overview")

    # ensure prefecture is valid; fallback to first actual pref if available
    current_pref = state.get("prefecture")
    if current_pref not in prefectures:
        if len(prefectures) > 1:
            state["prefecture"] = prefectures[1]
        else:
            state["prefecture"] = "全国"

    # ensure municipality is valid for selected prefecture
    if "municipality" in df.columns and state.get("prefecture") not in ("全国", None, ""):
        munis = get_municipality_options(state["prefecture"])
        current_muni = state.get("municipality")
        if current_muni not in munis:
            if len(munis) > 1:
                state["municipality"] = munis[1]
            else:
                state["municipality"] = munis[0]
    else:
        state["municipality"] = "すべて"

    # Header
    with ui.header().style(f"background-color: {BG_COLOR}; border-bottom: 1px solid {BORDER_COLOR}"):
        ui.label("MapComplete Dashboard").classes("text-xl font-bold").style(f"color: {TEXT_COLOR}")
        if _data_source == "Turso DB":
            ui.label(f"[DB] {_data_source}").classes("text-sm px-2 py-1 rounded").style(
                f"background-color: {ACCENT_GREEN}; color: white"
            )
        else:
            ui.label(f"[!] {_data_source}").classes("text-sm px-2 py-1 rounded").style(
                "background-color: #E69F00; color: white"
            )
        ui.space()
        ui.label(f"ログイン: {get_user_email()}").classes("text-sm").style(f"color: {MUTED_COLOR}")

        def handle_logout() -> None:
            app.storage.user["authenticated"] = False
            app.storage.user["email"] = ""
            ui.navigate.to("/login")

        ui.button("ログアウト", on_click=handle_logout).props("flat").style(f"color: {TEXT_COLOR}")

    # Filters
    def get_filtered_data() -> pd.DataFrame:
        filtered = df.copy()
        if state["prefecture"] != "全国" and "prefecture" in df.columns:
            filtered = filtered[filtered["prefecture"] == state["prefecture"]]
        if state["municipality"] != "すべて" and "municipality" in df.columns:
            filtered = filtered[filtered["municipality"] == state["municipality"]]
        return filtered

    # Container reference for municipality dropdown rebuild
    muni_container = None

    def _get_event_value(e, widget):
        """extract value from NiceGUI event; fallback to widget.value."""
        if hasattr(e, "value") and e.value is not None:
            return e.value
        if hasattr(e, "args") and e.args is not None:
            return e.args
        if widget is not None:
            return widget.value
        return None

    def create_municipality_dropdown():
        """Create municipality dropdown with proper event binding."""
        current_pref = state.get("prefecture", "全国")
        options = get_municipality_options(current_pref)
        current_muni = state.get("municipality", "すべて")
        if current_muni not in options:
            state["municipality"] = "すべて"
            current_muni = "すべて"

        async def on_muni_select(e):
            """Handle municipality selection change."""
            new_val = _get_event_value(e, muni_select)
            if new_val is not None:
                state["municipality"] = new_val
                log(f"[UI] municipality change -> {new_val}")
                ui.notify(f"市区町村: {new_val}")
                show_content.refresh()

        # Use on_change parameter in constructor (NiceGUI correct way)
        # 市区町村セレクタ
        muni_select = ui.select(
            options=options,
            value=current_muni,
            label="市区町村",
            on_change=on_muni_select,
        ).classes("w-48").props(
            'filled dense dark '
            'bg-color="blue-grey-9" '
            'label-color="cyan" '
            'color="white" '
            'popup-content-class="bg-grey-9 text-white"'
        ).style("min-width: 180px;")
        return muni_select

    async def on_pref_select(e):
        """Handle prefecture selection change and rebuild municipality dropdown."""
        nonlocal muni_container
        new_pref = _get_event_value(e, pref_select)
        if new_pref is not None:
            state["prefecture"] = new_pref
            state["municipality"] = "すべて"
            log(f"[UI] prefecture change -> {new_pref}")
            ui.notify(f"都道府県: {new_pref}")

            # Rebuild municipality dropdown with new options
            if muni_container:
                muni_container.clear()
                with muni_container:
                    create_municipality_dropdown()
            show_content.refresh()

    with ui.row().classes("w-full p-4 items-end gap-4").style(f"background-color: {PANEL_BG}"):
        # ドロップダウンを目立つカードで囲む
        with ui.card().classes("p-3").style(
            "background-color: #1a3a5c; "  # 青みがかった濃い背景
            "border: 2px solid #00BFFF; "  # 明るいシアンのボーダー
            "border-radius: 12px; "
            "box-shadow: 0 0 10px rgba(0, 191, 255, 0.3);"  # グロー効果
        ):
            with ui.row().classes("gap-4 items-center"):
                ui.icon("location_on", size="md").style("color: #00BFFF;")
                ui.label("地域選択").classes("text-lg font-bold").style("color: #00BFFF;")

            with ui.row().classes("gap-4 mt-2"):
                # Prefecture selector (show message if data missing)
                if len(prefectures) <= 1:
                    ui.label("都道府県データがありません").style(f"color: {MUTED_COLOR}")

                # 都道府県セレクタ
                pref_select = ui.select(
                    options=prefecture_options,
                    value=state.get("prefecture", "全国"),
                    label="都道府県",
                    on_change=on_pref_select,
                ).classes("w-48").props(
                    'filled dense dark '
                    'bg-color="blue-grey-9" '
                    'label-color="cyan" '
                    'color="white" '
                    'popup-content-class="bg-grey-9 text-white"'
                ).style("min-width: 180px;")

                # Municipality container (rebuilt on prefecture change)
                muni_container = ui.element("div").classes("w-48")
                with muni_container:
                    create_municipality_dropdown()

    # Content
    @ui.refreshable
    def show_content() -> None:
        filtered_df = get_filtered_data()
        tab = state.get("tab", "overview")  # デフォルトをoverviewに変更（軽量・echart無し）
        print(f"[DEBUG] show_content called, tab = {tab}")

        with ui.column().classes("w-full p-4"):
            if filtered_df.empty:
                ui.label("データがありません").style(f"color: {TEXT_COLOR}")
                return

            if tab == "overview":
                # === 市場概況タブ（Reflex完全再現版） ===
                ui.label("総合概要").classes("text-xl font-semibold mb-4").style(f"color: {TEXT_COLOR}")

                # データ取得（Reflexと同じロジック）
                total_applicants = int(safe_sum(filtered_df, "applicant_count")) if "applicant_count" in filtered_df.columns else len(filtered_df)
                male_total = int(safe_sum(filtered_df, "male_count")) if "male_count" in filtered_df.columns else 0
                female_total = int(safe_sum(filtered_df, "female_count")) if "female_count" in filtered_df.columns else 0
                avg_age_val = round(safe_mean(filtered_df, "avg_age"), 1) if "avg_age" in filtered_df.columns else None

                # === KPIカード（3列）：求職者数、平均年齢、男女比 ===
                ui.label("KPI").classes("text-sm font-semibold mb-2").style(f"color: {MUTED_COLOR}")
                with ui.row().classes("w-full gap-4"):
                    # 求職者数
                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 20px; border-radius: 12px"
                    ):
                        ui.label("求職者数").classes("text-sm").style(f"color: {MUTED_COLOR}")
                        ui.label(f"{total_applicants:,}").classes("text-2xl font-bold").style(f"color: {PRIMARY_COLOR}")
                        ui.label("人").classes("text-sm").style(f"color: {MUTED_COLOR}")

                    # 平均年齢
                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 20px; border-radius: 12px"
                    ):
                        ui.label("平均年齢").classes("text-sm").style(f"color: {MUTED_COLOR}")
                        ui.label(f"{avg_age_val if avg_age_val else '-'}").classes("text-2xl font-bold").style(f"color: {ACCENT_GREEN}")
                        ui.label("歳").classes("text-sm").style(f"color: {MUTED_COLOR}")

                    # 男女比
                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 20px; border-radius: 12px"
                    ):
                        ui.label("男女比").classes("text-sm").style(f"color: {MUTED_COLOR}")
                        gender_ratio_text = f"{male_total:,} / {female_total:,}" if (male_total > 0 or female_total > 0) else "-"
                        ui.label(gender_ratio_text).classes("text-2xl font-bold").style(f"color: {ACCENT_PINK}")
                        ui.label("人").classes("text-sm").style(f"color: {MUTED_COLOR}")

                # === 3層比較パネル（全国・都道府県・市区町村） ===
                pref_val = state["prefecture"] if state["prefecture"] != "全国" else None
                muni_val = state["municipality"] if state["municipality"] != "すべて" else None

                # db_helperから統計取得
                nat_stats = get_national_stats()
                pref_stats = get_prefecture_stats(pref_val) if pref_val else {}
                muni_stats = get_municipality_stats(pref_val, muni_val) if pref_val and muni_val else {}

                with ui.card().classes("w-full mt-4").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 20px; border-radius: 12px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("📊").classes("text-lg")
                        ui.label("地域比較").classes("font-semibold").style(f"color: {TEXT_COLOR}")

                    if pref_val and muni_val:
                        ui.label(f"全国 vs {pref_val} vs {muni_val}").classes("text-xs mb-4").style(f"color: {MUTED_COLOR}")
                    else:
                        ui.label("地域を選択してください").classes("text-xs mb-4").style(f"color: {MUTED_COLOR}")

                    # 比較メトリクス（希望勤務地数、平均移動距離、資格保有数）- NaN処理付き
                    import math
                    def safe_val(val, default=0.0):
                        """NaN安全な値変換（明示的チェック）"""
                        if val is None:
                            return default
                        try:
                            f = float(val)
                            if math.isnan(f) or math.isinf(f):
                                return default
                            return f
                        except (ValueError, TypeError):
                            return default

                    def format_val(v, unit):
                        """値のフォーマット（NaN/0は'-'表示）"""
                        if v is None or v == 0 or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                            return "-"
                        return f"{v:.1f}{unit}"

                    comparison_metrics = [
                        ("希望勤務地数", "件", safe_val(nat_stats.get("desired_areas")), safe_val(pref_stats.get("desired_areas")), safe_val(muni_stats.get("desired_areas"))),
                        ("平均移動距離", "km", safe_val(nat_stats.get("distance_km")), safe_val(pref_stats.get("distance_km")), safe_val(muni_stats.get("distance_km"))),
                        ("資格保有数", "個", safe_val(nat_stats.get("qualifications")), safe_val(pref_stats.get("qualifications")), safe_val(muni_stats.get("qualifications"))),
                    ]

                    for label, unit, nat_v, pref_v, muni_v in comparison_metrics:
                        with ui.row().classes("w-full items-center mb-2"):
                            ui.label(f"{label}").classes("w-24 text-sm").style(f"color: {TEXT_COLOR}")
                            with ui.column().classes("flex-1 gap-1"):
                                # 全国
                                with ui.row().classes("items-center gap-2"):
                                    ui.label("全国").classes("w-16 text-xs").style(f"color: {PRIMARY_COLOR}")
                                    with ui.element("div").classes("flex-1 h-4 rounded overflow-hidden").style(f"background-color: {BORDER_COLOR}"):
                                        ui.element("div").classes("h-full").style(f"width: 100%; background-color: {PRIMARY_COLOR}")
                                    ui.label(format_val(nat_v, unit)).classes("w-16 text-xs text-right").style(f"color: {MUTED_COLOR}")
                                # 都道府県
                                if pref_val:
                                    pref_pct = min(int(pref_v / nat_v * 100), 200) if nat_v > 0 else 0
                                    with ui.row().classes("items-center gap-2"):
                                        ui.label(pref_val[:4]).classes("w-16 text-xs").style(f"color: {SECONDARY_COLOR}")
                                        with ui.element("div").classes("flex-1 h-4 rounded overflow-hidden").style(f"background-color: {BORDER_COLOR}"):
                                            ui.element("div").classes("h-full").style(f"width: {pref_pct}%; background-color: {SECONDARY_COLOR}")
                                        ui.label(format_val(pref_v, unit)).classes("w-16 text-xs text-right").style(f"color: {MUTED_COLOR}")
                                # 市区町村
                                if muni_val:
                                    muni_pct = min(int(muni_v / nat_v * 100), 200) if nat_v > 0 else 0
                                    arrow = "▲" if muni_v > nat_v else ("▼" if muni_v < nat_v else "")
                                    muni_display = format_val(muni_v, unit)
                                    with ui.row().classes("items-center gap-2"):
                                        ui.label(muni_val[:4]).classes("w-16 text-xs").style(f"color: {ACCENT_4}")
                                        with ui.element("div").classes("flex-1 h-4 rounded overflow-hidden").style(f"background-color: {BORDER_COLOR}"):
                                            ui.element("div").classes("h-full").style(f"width: {muni_pct}%; background-color: {ACCENT_4}")
                                        ui.label(f"{muni_display} {arrow}" if muni_display != "-" else "-").classes("w-20 text-xs text-right").style(f"color: {MUTED_COLOR}")

                    # 性別比率セクション
                    ui.label("性別比率").classes("text-sm font-semibold mt-4 mb-2").style(f"color: {TEXT_COLOR}")
                    nat_male = nat_stats.get("male_count", 0)
                    nat_female = nat_stats.get("female_count", 0)
                    nat_total_g = nat_male + nat_female
                    nat_male_pct = round(nat_male / nat_total_g * 100, 1) if nat_total_g > 0 else 0
                    nat_female_pct = round(nat_female / nat_total_g * 100, 1) if nat_total_g > 0 else 0

                    pref_male = pref_stats.get("male_count", 0)
                    pref_female = pref_stats.get("female_count", 0)
                    pref_total_g = pref_male + pref_female
                    pref_male_pct = round(pref_male / pref_total_g * 100, 1) if pref_total_g > 0 else 0
                    pref_female_pct = round(pref_female / pref_total_g * 100, 1) if pref_total_g > 0 else 0

                    muni_male = muni_stats.get("male_count", 0)
                    muni_female = muni_stats.get("female_count", 0)
                    muni_total_g = muni_male + muni_female
                    muni_male_pct = round(muni_male / muni_total_g * 100, 1) if muni_total_g > 0 else 0
                    muni_female_pct = round(muni_female / muni_total_g * 100, 1) if muni_total_g > 0 else 0

                    gender_layers = [
                        ("全国", PRIMARY_COLOR, nat_male_pct, nat_female_pct),
                    ]
                    if pref_val:
                        gender_layers.append((pref_val, SECONDARY_COLOR, pref_male_pct, pref_female_pct))
                    if muni_val:
                        gender_layers.append((muni_val, ACCENT_4, muni_male_pct, muni_female_pct))

                    for layer_name, layer_color, m_pct, f_pct in gender_layers:
                        with ui.row().classes("w-full items-center gap-2 mb-1"):
                            ui.label(layer_name[:6]).classes("w-16 text-xs").style(f"color: {layer_color}")
                            with ui.element("div").classes("flex-1 h-4 flex rounded overflow-hidden"):
                                ui.element("div").style(f"width: {m_pct}%; background-color: #3b82f6; height: 100%")
                                ui.element("div").style(f"width: {f_pct}%; background-color: #ec4899; height: 100%")
                            ui.label(f"男{m_pct}% / 女{f_pct}%").classes("text-xs w-28 text-right").style(f"color: {MUTED_COLOR}")

                    # 凡例
                    with ui.row().classes("gap-4 mt-2"):
                        with ui.row().classes("items-center gap-1"):
                            ui.element("div").classes("w-3 h-3 rounded").style("background-color: #3b82f6")
                            ui.label("男性").classes("text-xs").style(f"color: {MUTED_COLOR}")
                        with ui.row().classes("items-center gap-1"):
                            ui.element("div").classes("w-3 h-3 rounded").style("background-color: #ec4899")
                            ui.label("女性").classes("text-xs").style(f"color: {MUTED_COLOR}")

                    # 年齢層分布グラフ（3層比較）
                    ui.label("年齢層分布").classes("text-sm font-semibold mt-4 mb-2").style(f"color: {TEXT_COLOR}")
                    nat_age_dist = nat_stats.get("age_distribution", {})
                    pref_age_dist = pref_stats.get("age_distribution", {})
                    muni_age_dist = muni_stats.get("age_distribution", {})
                    age_order = ["20代", "30代", "40代", "50代", "60代", "70歳以上"]

                    age_chart_data = []
                    for age in age_order:
                        age_chart_data.append({
                            "name": age,
                            "全国": nat_age_dist.get(age, 0),
                            "都道府県": pref_age_dist.get(age, 0) if pref_val else 0,
                            "市区町村": muni_age_dist.get(age, 0) if muni_val else 0,
                        })

                    series_list = [{"name": "全国", "type": "bar", "data": [d["全国"] for d in age_chart_data], "itemStyle": {"color": PRIMARY_COLOR}}]
                    if pref_val:
                        series_list.append({"name": "都道府県", "type": "bar", "data": [d["都道府県"] for d in age_chart_data], "itemStyle": {"color": SECONDARY_COLOR}})
                    if muni_val:
                        series_list.append({"name": "市区町村", "type": "bar", "data": [d["市区町村"] for d in age_chart_data], "itemStyle": {"color": ACCENT_4}})

                    ui.echart({
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "legend": {"textStyle": {"color": MUTED_COLOR}},
                        "xAxis": {"type": "category", "data": age_order, "axisLabel": {"color": MUTED_COLOR}},
                        "yAxis": {"type": "value", "name": "%", "axisLabel": {"color": MUTED_COLOR}},
                        "series": series_list,
                    }).classes("w-full h-80")

                    # 地域凡例
                    with ui.row().classes("gap-4 mt-2"):
                        with ui.row().classes("items-center gap-1"):
                            ui.element("div").classes("w-3 h-3 rounded").style(f"background-color: {PRIMARY_COLOR}")
                            ui.label("全国").classes("text-xs").style(f"color: {MUTED_COLOR}")
                        if pref_val:
                            with ui.row().classes("items-center gap-1"):
                                ui.element("div").classes("w-3 h-3 rounded").style(f"background-color: {SECONDARY_COLOR}")
                                ui.label("都道府県").classes("text-xs").style(f"color: {MUTED_COLOR}")
                        if muni_val:
                            with ui.row().classes("items-center gap-1"):
                                ui.element("div").classes("w-3 h-3 rounded").style(f"background-color: {ACCENT_4}")
                                ui.label("市区町村").classes("text-xs").style(f"color: {MUTED_COLOR}")

                # === グラフ1: 性別構成（ドーナツ/パイチャート） ===
                ui.label("性別構成").classes("text-sm font-semibold mt-6 mb-2").style(f"color: {MUTED_COLOR}")
                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 24px; border-radius: 12px"
                ):
                    gender_pie_data = []
                    if male_total > 0:
                        gender_pie_data.append({"value": male_total, "name": "男性", "itemStyle": {"color": "#0072B2"}})
                    if female_total > 0:
                        gender_pie_data.append({"value": female_total, "name": "女性", "itemStyle": {"color": "#E69F00"}})

                    if gender_pie_data:
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                            "legend": {"orient": "vertical", "left": "left", "textStyle": {"color": MUTED_COLOR}},
                            "series": [{
                                "type": "pie",
                                "radius": ["40%", "70%"],
                                "center": ["50%", "50%"],
                                "data": gender_pie_data,
                                "label": {"show": True, "formatter": "{b}: {d}%", "color": MUTED_COLOR},
                            }],
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # === グラフ2: 年齢帯別分布（棒グラフ） ===
                # filtered_dfから直接年齢分布を計算（Tursoクエリに依存しない）
                ui.label("年齢帯別分布").classes("text-sm font-semibold mt-6 mb-2").style(f"color: {MUTED_COLOR}")
                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 24px; border-radius: 12px"
                ):
                    # filtered_dfから年齢分布を計算
                    df_age_dist = {}
                    if "avg_age" in filtered_df.columns:
                        age_data = filtered_df["avg_age"].dropna()
                        if len(age_data) > 0:
                            # 年齢を年代にバケット化
                            def age_to_bucket(age):
                                try:
                                    a = float(age)
                                    if a < 30: return "20代"
                                    elif a < 40: return "30代"
                                    elif a < 50: return "40代"
                                    elif a < 60: return "50代"
                                    elif a < 70: return "60代"
                                    else: return "70歳以上"
                                except:
                                    return None

                            age_buckets = age_data.apply(age_to_bucket).dropna()
                            df_age_dist = age_buckets.value_counts().to_dict()

                    age_bar_data = [{"name": age, "count": int(df_age_dist.get(age, 0))} for age in age_order]

                    if any(d["count"] > 0 for d in age_bar_data):
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis"},
                            "xAxis": {"type": "category", "data": [d["name"] for d in age_bar_data], "axisLabel": {"color": MUTED_COLOR}},
                            "yAxis": {"type": "value", "name": "人数", "axisLabel": {"color": MUTED_COLOR}},
                            "series": [{
                                "type": "bar",
                                "data": [d["count"] for d in age_bar_data],
                                "name": "人数",
                                "itemStyle": {"color": PRIMARY_COLOR},
                            }],
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # === グラフ3: 年齢層×性別分布（グループ化棒グラフ） ===
                # 選択された地域のデータを使用
                ui.label("年齢層×性別分布").classes("text-sm font-semibold mt-6 mb-2").style(f"color: {MUTED_COLOR}")
                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 24px; border-radius: 12px"
                ):
                    # 選択された地域に応じたage_distributionを使用
                    # 優先順位: 市区町村 > 都道府県 > 全国
                    if muni_val and muni_stats.get("age_distribution"):
                        age_dist_data = muni_stats.get("age_distribution", {})
                    elif pref_val and pref_stats.get("age_distribution"):
                        age_dist_data = pref_stats.get("age_distribution", {})
                    else:
                        age_dist_data = nat_stats.get("age_distribution", {})
                    # df_age_distが空でもage_dist_dataがあれば使用
                    effective_age_dist = age_dist_data if age_dist_data else df_age_dist

                    # 全体の男女比から各年齢層の推定値を計算
                    total_gender = male_total + female_total
                    male_ratio = male_total / total_gender if total_gender > 0 else 0.5
                    female_ratio = female_total / total_gender if total_gender > 0 else 0.5

                    male_by_age = {age: int(effective_age_dist.get(age, 0) * male_ratio) for age in age_order}
                    female_by_age = {age: int(effective_age_dist.get(age, 0) * female_ratio) for age in age_order}

                    if any(male_by_age.values()) or any(female_by_age.values()):
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis"},
                            "legend": {"data": ["男性", "女性"], "textStyle": {"color": MUTED_COLOR}},
                            "xAxis": {"type": "category", "data": age_order, "axisLabel": {"color": MUTED_COLOR}},
                            "yAxis": {"type": "value", "name": "人数", "axisLabel": {"color": MUTED_COLOR}},
                            "series": [
                                {"name": "男性", "type": "bar", "data": [male_by_age.get(age, 0) for age in age_order], "itemStyle": {"color": "#0072B2"}},
                                {"name": "女性", "type": "bar", "data": [female_by_age.get(age, 0) for age in age_order], "itemStyle": {"color": "#E69F00"}},
                            ],
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}")

            elif tab == "demographics":
                ui.label("ペルソナ分析").classes("text-lg font-bold mb-4").style(f"color: {TEXT_COLOR}")

                # db_helper.pyの専用関数を使ってデータ取得（Reflexと同じロジック）
                pref_val = state["prefecture"] if state["prefecture"] != "全国" else None
                muni_val = state["municipality"] if state["municipality"] != "すべて" else None

                # ペルソナシェアデータを取得
                persona_data = get_persona_market_share(pref_val, muni_val)

                # 資格データを取得
                qualification_data = get_qualification_retention_rates(pref_val, muni_val)

                # ----- 1行目: 全ペルソナ内訳 + ペルソナ構成比横棒グラフ -----
                with ui.row().classes("w-full gap-4"):
                    # 左側: 全ペルソナ内訳リスト
                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px; min-width: 300px"
                    ):
                        ui.label("全ペルソナ内訳（100%）").classes("text-sm font-semibold mb-3").style(f"color: {MUTED_COLOR}")

                        if persona_data:
                            with ui.scroll_area().style("max-height: 350px"):
                                for item in persona_data:
                                    with ui.row().classes("w-full justify-between items-center py-1"):
                                        ui.label(item.get("label", "-")).classes("font-semibold").style(f"color: {TEXT_COLOR}; font-size: 0.85rem")
                                        ui.label(f'{item.get("count", 0):,}人 ({item.get("share_pct", "0%")})').style(f"color: {MUTED_COLOR}; font-size: 0.85rem")
                        else:
                            ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                    # 右側: ペルソナ構成比（横棒グラフ）
                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px; min-width: 400px"
                    ):
                        ui.label("ペルソナ構成比（横棒グラフ）").classes("text-sm font-semibold mb-3").style(f"color: {MUTED_COLOR}")

                        if persona_data:
                            # データを横棒グラフ用に変換（上位10件）
                            labels = [item.get("label", "") for item in persona_data[:10]]
                            values = [item.get("count", 0) for item in persona_data[:10]]
                            # 逆順にして上から多い順に表示
                            labels = labels[::-1]
                            values = values[::-1]

                            ui.echart({
                                "backgroundColor": "transparent",
                                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                                "grid": {"left": "25%", "right": "10%", "top": "10%", "bottom": "10%"},
                                "xAxis": {"type": "value", "axisLabel": {"color": MUTED_COLOR}},
                                "yAxis": {
                                    "type": "category",
                                    "data": labels,
                                    "axisLabel": {"color": MUTED_COLOR, "fontSize": 11},
                                },
                                "series": [{
                                    "type": "bar",
                                    "data": values,
                                    "itemStyle": {"color": PRIMARY_COLOR},
                                    "label": {"show": True, "position": "right", "color": TEXT_COLOR}
                                }]
                            }).classes("w-full h-80")
                        else:
                            ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # ----- 2行目: 資格詳細（全資格一覧） -----
                ui.label("資格詳細（全資格一覧）").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {MUTED_COLOR}")

                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                ):
                    if qualification_data:
                        with ui.scroll_area().style("max-height: 350px"):
                            for item in qualification_data:
                                with ui.row().classes("w-full justify-between items-center py-2 border-b").style(f"border-color: {BORDER_COLOR}"):
                                    ui.label(item.get("qualification", "-")).classes("font-semibold").style(f"color: {TEXT_COLOR}; font-size: 0.9rem")
                                    with ui.row().classes("gap-4 items-center"):
                                        ui.label(f'定着率: {item.get("retention_rate", "-")}').style(f"color: {MUTED_COLOR}; font-size: 0.85rem")
                                        interpretation = item.get("interpretation", "-")
                                        interp_color = ACCENT_GREEN if interpretation == "地元志向" else ACCENT_PINK
                                        ui.label(interpretation).style(f"color: {interp_color}; font-size: 0.85rem")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # ----- 3行目: 年齢×性別クロス分析 -----
                ui.label("年齢×性別クロス分析").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {MUTED_COLOR}")

                with ui.row().classes("w-full gap-4"):
                    # 左側: 男女比ドーナツチャート
                    # get_municipality_statsまたはget_prefecture_statsからデータ取得
                    if pref_val and muni_val:
                        demo_stats = get_municipality_stats(pref_val, muni_val)
                    elif pref_val:
                        demo_stats = get_prefecture_stats(pref_val)
                    else:
                        demo_stats = get_national_stats()

                    male_total = demo_stats.get("male_count", 0)
                    female_total = demo_stats.get("female_count", 0)

                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                    ):
                        if male_total > 0 or female_total > 0:
                            ui.echart({
                                "backgroundColor": "transparent",
                                "title": {"text": "男女比", "textStyle": {"color": TEXT_COLOR}},
                                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                                "legend": {"orient": "vertical", "left": "left", "textStyle": {"color": MUTED_COLOR}},
                                "series": [{
                                    "type": "pie",
                                    "radius": ["40%", "70%"],
                                    "data": [
                                        {"value": male_total, "name": "男性", "itemStyle": {"color": "#0072B2"}},
                                        {"value": female_total, "name": "女性", "itemStyle": {"color": "#E69F00"}},
                                    ],
                                }]
                            }).classes("w-full h-80")
                        else:
                            ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                    # 右側: 年齢層×性別分布（積み上げ棒グラフ）
                    with ui.card().classes("flex-1").style(
                        f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                    ):
                        # persona_dataから年齢×性別データを計算（正しいデータソース）
                        # persona_dataのlabelは "50代・女性" のような形式
                        age_order = ["20代", "30代", "40代", "50代", "60代", "70歳以上"]
                        male_by_age = {age: 0 for age in age_order}
                        female_by_age = {age: 0 for age in age_order}

                        if persona_data:
                            for item in persona_data:
                                label = item.get("label", "")
                                count = item.get("count", 0)
                                # labelを解析: "50代・女性" -> age="50代", gender="女性"
                                parts = label.split("・")
                                if len(parts) == 2:
                                    age_part, gender_part = parts
                                    # 年齢表記を正規化
                                    if "70" in age_part or "以上" in age_part:
                                        age_key = "70歳以上"
                                    elif "60" in age_part:
                                        age_key = "60代"
                                    elif "50" in age_part:
                                        age_key = "50代"
                                    elif "40" in age_part:
                                        age_key = "40代"
                                    elif "30" in age_part:
                                        age_key = "30代"
                                    elif "20" in age_part or "10" in age_part:
                                        age_key = "20代"
                                    else:
                                        continue

                                    if "男" in gender_part:
                                        male_by_age[age_key] += count
                                    elif "女" in gender_part:
                                        female_by_age[age_key] += count

                        male_data = [male_by_age.get(age, 0) for age in age_order]
                        female_data = [female_by_age.get(age, 0) for age in age_order]

                        if any(male_data) or any(female_data):
                            ui.echart({
                                "backgroundColor": "transparent",
                                "title": {"text": "年齢層×性別分布", "textStyle": {"color": TEXT_COLOR}},
                                "tooltip": {"trigger": "axis"},
                                "legend": {"data": ["男性", "女性"], "textStyle": {"color": MUTED_COLOR}},
                                "xAxis": {"type": "category", "data": age_order, "axisLabel": {"color": MUTED_COLOR}},
                                "yAxis": {"type": "value", "axisLabel": {"color": MUTED_COLOR}},
                                "series": [
                                    {"name": "男性", "type": "bar", "stack": "total", "data": male_data, "itemStyle": {"color": "#0072B2"}},
                                    {"name": "女性", "type": "bar", "stack": "total", "data": female_data, "itemStyle": {"color": "#E69F00"}},
                                ]
                            }).classes("w-full h-80")
                        else:
                            ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # ----- 4行目: KPIカード -----
                with ui.row().classes("w-full gap-4 mt-6"):
                    # 女性比率
                    fr = 0
                    total = male_total + female_total
                    if total > 0:
                        fr = round((female_total / total * 100), 1)

                    with ui.card().style(f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"):
                        ui.label("女性比率").classes("text-sm").style(f"color: {MUTED_COLOR}")
                        ui.label(f"{fr}%").classes("text-2xl font-bold").style(f"color: #E69F00")

                    # 平均資格保有数
                    avg_quals = demo_stats.get("qualifications", 0)
                    with ui.card().style(f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"):
                        ui.label("平均資格数").classes("text-sm").style(f"color: {MUTED_COLOR}")
                        ui.label(f"{avg_quals:.1f}").classes("text-2xl font-bold").style(f"color: {ACCENT_GREEN}")

                    # 平均移動距離
                    avg_dist = demo_stats.get("distance_km", 0)
                    with ui.card().style(f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"):
                        ui.label("平均移動距離").classes("text-sm").style(f"color: {MUTED_COLOR}")
                        ui.label(f"{avg_dist:.1f}km").classes("text-2xl font-bold").style(f"color: {PRIMARY_COLOR}")

                # ----- 5行目: 年齢・性別×就業状態別内訳 Top 10（積み上げ棒グラフ） -----
                ui.label("年齢・性別×就業状態別内訳 Top 10").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {MUTED_COLOR}")

                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                ):
                    employment_data = get_persona_employment_breakdown(pref_val, muni_val)
                    if employment_data:
                        labels = [item["age_gender"] for item in employment_data]
                        employed = [item["就業中"] for item in employment_data]
                        unemployed = [item["離職中"] for item in employment_data]
                        student = [item["在学中"] for item in employment_data]

                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "legend": {"data": ["就業中", "離職中", "在学中"], "textStyle": {"color": MUTED_COLOR}},
                            "grid": {"left": "15%", "right": "5%", "top": "15%", "bottom": "15%"},
                            "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": MUTED_COLOR, "rotate": 45}},
                            "yAxis": {"type": "value", "axisLabel": {"color": MUTED_COLOR}},
                            "series": [
                                {"name": "就業中", "type": "bar", "stack": "employment", "data": employed, "itemStyle": {"color": "#10b981"}},
                                {"name": "離職中", "type": "bar", "stack": "employment", "data": unemployed, "itemStyle": {"color": "#CC79A7"}},
                                {"name": "在学中", "type": "bar", "stack": "employment", "data": student, "itemStyle": {"color": "#F0E442"}},
                            ],
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # ----- 6行目: 保有資格ペルソナ（主要資格Top10 男女別棒グラフ） -----
                ui.label("保有資格ペルソナ（主要資格Top10 男女別）").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {MUTED_COLOR}")

                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                ):
                    qual_gender_data = get_qualification_by_gender(pref_val, muni_val)
                    if qual_gender_data:
                        labels = [item["qualification"] for item in qual_gender_data]
                        male_counts = [item["male"] for item in qual_gender_data]
                        female_counts = [item["female"] for item in qual_gender_data]

                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "legend": {"data": ["男性", "女性"], "textStyle": {"color": MUTED_COLOR}},
                            "grid": {"left": "20%", "right": "5%", "top": "15%", "bottom": "10%"},
                            "xAxis": {"type": "value", "axisLabel": {"color": MUTED_COLOR}},
                            "yAxis": {"type": "category", "data": labels[::-1], "axisLabel": {"color": MUTED_COLOR, "fontSize": 11}},
                            "series": [
                                {"name": "男性", "type": "bar", "data": male_counts[::-1], "itemStyle": {"color": "#0072B2"}},
                                {"name": "女性", "type": "bar", "data": female_counts[::-1], "itemStyle": {"color": "#E69F00"}},
                            ],
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}")

                # ----- 7行目: ペルソナシェア（年齢×性別）横棒グラフ + バッジ -----
                ui.label("ペルソナシェア（年齢×性別）").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {MUTED_COLOR}")

                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                ):
                    ui.label("この地域の人材構成比（年齢×性別）").classes("text-xs mb-2").style(f"color: {MUTED_COLOR}")

                    if persona_data:
                        # 横棒グラフ
                        labels = [item.get("label", "") for item in persona_data[:10]]
                        values = [item.get("count", 0) for item in persona_data[:10]]

                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "grid": {"left": "25%", "right": "10%", "top": "5%", "bottom": "5%"},
                            "xAxis": {"type": "value", "axisLabel": {"color": MUTED_COLOR}},
                            "yAxis": {"type": "category", "data": labels[::-1], "axisLabel": {"color": MUTED_COLOR}},
                            "series": [{
                                "type": "bar",
                                "data": values[::-1],
                                "itemStyle": {"color": PRIMARY_COLOR},
                                "label": {"show": True, "position": "right", "color": TEXT_COLOR}
                            }]
                        }).classes("w-full h-80")

                        # シェアバッジ表示
                        with ui.row().classes("gap-2 flex-wrap mt-2"):
                            for item in persona_data[:6]:
                                with ui.element("span").style(
                                    f"background-color: rgba(99, 102, 241, 0.1); color: {TEXT_COLOR}; "
                                    "padding: 4px 8px; border-radius: 4px; font-size: 0.75rem"
                                ):
                                    ui.label(f"{item.get('label', '')}: {item.get('share_pct', '')}")
                    else:
                        ui.label("シェアデータがありません").style(f"color: {MUTED_COLOR}")

                # ----- 8行目: 希望勤務地数・資格保有数（年齢×性別リスト） -----
                ui.label("希望勤務地数・資格保有数（年齢×性別）").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {MUTED_COLOR}")

                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                ):
                    ui.label("年齢×性別ごとの平均値").classes("text-xs mb-3").style(f"color: {MUTED_COLOR}")

                    age_gender_stats = get_age_gender_stats(pref_val, muni_val)
                    if age_gender_stats:
                        for item in age_gender_stats:
                            with ui.row().classes("w-full justify-between items-center py-2 border-b").style(f"border-color: {BORDER_COLOR}"):
                                ui.label(item.get("label", "-")).classes("font-semibold").style(f"color: {TEXT_COLOR}; font-size: 0.85rem; min-width: 80px")
                                with ui.row().classes("gap-4"):
                                    with ui.row().classes("gap-1"):
                                        ui.label("希望勤務地:").style(f"color: {MUTED_COLOR}; font-size: 0.75rem")
                                        ui.label(f"{item.get('desired_areas', '-')}箇所").style(f"color: {PRIMARY_COLOR}; font-size: 0.85rem; font-weight: 500")
                                    with ui.row().classes("gap-1"):
                                        ui.label("資格:").style(f"color: {MUTED_COLOR}; font-size: 0.75rem")
                                        ui.label(f"{item.get('qualifications', '-')}個").style(f"color: {ACCENT_GREEN}; font-size: 0.85rem; font-weight: 500")
                    else:
                        ui.label("統計データがありません").style(f"color: {MUTED_COLOR}")

                # ----- 9行目: 人材組み合わせ分析（RARITY）-----
                ui.label("🎯 人材組み合わせ分析").classes("text-sm font-semibold mt-6 mb-3").style(f"color: {TEXT_COLOR}")

                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 16px"
                ):
                    ui.label("年代・性別・資格を選択して検索").classes("text-xs mb-3").style(f"color: {PRIMARY_COLOR}")

                    # RARITY用のローカル状態
                    rarity_state = {"ages": [], "genders": [], "qualifications": [], "results": []}

                    # 年代チェックボックス
                    with ui.element("div").classes("p-2 rounded mb-2").style("background-color: rgba(59, 130, 246, 0.05)"):
                        ui.label("年代（複数選択可）").classes("text-xs font-semibold mb-1").style(f"color: {MUTED_COLOR}")
                        with ui.row().classes("gap-4 flex-wrap"):
                            for age in ["20代", "30代", "40代", "50代", "60代", "70歳以上"]:
                                ui.checkbox(age, on_change=lambda e, a=age: (
                                    rarity_state["ages"].append(a) if e.value else rarity_state["ages"].remove(a) if a in rarity_state["ages"] else None
                                )).classes("text-sm").style(f"color: {TEXT_COLOR}")

                    # 性別チェックボックス
                    with ui.element("div").classes("p-2 rounded mb-2").style("background-color: rgba(34, 197, 94, 0.05)"):
                        ui.label("性別（複数選択可）").classes("text-xs font-semibold mb-1").style(f"color: {MUTED_COLOR}")
                        with ui.row().classes("gap-4"):
                            for gender in ["男性", "女性"]:
                                ui.checkbox(gender, on_change=lambda e, g=gender: (
                                    rarity_state["genders"].append(g) if e.value else rarity_state["genders"].remove(g) if g in rarity_state["genders"] else None
                                )).classes("text-sm").style(f"color: {TEXT_COLOR}")

                    # 資格チェックボックス（レスポンシブグリッド）
                    qual_options = get_qualification_options(pref_val, muni_val)
                    with ui.element("div").classes("p-2 rounded mb-3").style("background-color: rgba(168, 85, 247, 0.05)"):
                        ui.label("資格（複数選択可）").classes("text-xs font-semibold mb-1").style(f"color: {MUTED_COLOR}")
                        # レスポンシブグリッド: 画面幅に応じて2〜4列に自動調整
                        with ui.element("div").classes("w-full").style(
                            "display: grid; "
                            "grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); "
                            "gap: 4px 16px; "
                            "max-height: 200px; "
                            "overflow-y: auto; "
                            "padding-right: 8px"
                        ):
                            for qual in qual_options[:20]:
                                ui.checkbox(qual, on_change=lambda e, q=qual: (
                                    rarity_state["qualifications"].append(q) if e.value else rarity_state["qualifications"].remove(q) if q in rarity_state["qualifications"] else None
                                )).classes("text-sm").style(f"color: {TEXT_COLOR}; white-space: nowrap")

                    # 検索結果表示エリア
                    result_container = ui.column().classes("w-full")

                    def do_rarity_search():
                        result_container.clear()
                        results = get_rarity_analysis(
                            pref_val, muni_val,
                            ages=rarity_state["ages"] or None,
                            genders=rarity_state["genders"] or None,
                            qualifications=rarity_state["qualifications"] or None
                        )
                        with result_container:
                            if results:
                                total = sum(r["count"] for r in results)
                                with ui.row().classes("gap-2 mb-2"):
                                    ui.badge(f"該当: {total:,}人", color="primary")
                                    ui.badge(f"組み合わせ: {len(results)}件", color="gray")
                                with ui.scroll_area().style("max-height: 300px"):
                                    for item in results:
                                        with ui.row().classes("w-full items-center gap-2 py-1"):
                                            ui.label(item["qualification"]).classes("font-semibold").style(f"color: {TEXT_COLOR}; font-size: 0.85rem; min-width: 120px")
                                            ui.label(item["age"]).style(f"color: {MUTED_COLOR}; font-size: 0.8rem; min-width: 50px")
                                            ui.label(item["gender"]).style(f"color: {MUTED_COLOR}; font-size: 0.8rem; min-width: 40px")
                                            ui.element("div").classes("flex-1")
                                            ui.label(f"{item['count']:,}人").style(f"color: {PRIMARY_COLOR}; font-size: 0.85rem; font-weight: 500")
                                            ui.label(f"({item['share_pct']})").style(f"color: {MUTED_COLOR}; font-size: 0.8rem")
                            else:
                                ui.label("条件を選択して検索してください").style(f"color: {MUTED_COLOR}")

                    def clear_rarity_selection():
                        rarity_state["ages"] = []
                        rarity_state["genders"] = []
                        rarity_state["qualifications"] = []
                        result_container.clear()
                        with result_container:
                            ui.label("条件を選択して検索してください").style(f"color: {MUTED_COLOR}")

                    # 検索・クリアボタン
                    with ui.row().classes("gap-2 mb-3"):
                        ui.button("🔍 検索", on_click=do_rarity_search).props("color=primary size=sm")
                        ui.button("🗑️ クリア", on_click=clear_rarity_selection).props("outline color=gray size=sm")

                    with result_container:
                        ui.label("条件を選択して検索してください").style(f"color: {MUTED_COLOR}")

            elif tab == "mobility":
                ui.label("🗺️ 地域・移動パターン").classes("text-xl font-bold mb-4").style(f"color: {TEXT_COLOR}")

                # db_helper.pyの専用関数を使ってデータ取得（Reflexと同じロジック）
                pref_val = state["prefecture"] if state["prefecture"] != "全国" else None
                muni_val = state["municipality"] if state["municipality"] != "すべて" else None

                flow_data = get_talent_flow(pref_val, muni_val)
                dist_data = get_distance_stats(pref_val, muni_val)
                flow_sources = get_flow_sources(pref_val, muni_val, limit=10)
                flow_destinations = get_flow_destinations(pref_val, muni_val, limit=10)
                competition_data = get_competition_overview(pref_val, muni_val)
                mobility_dist = get_mobility_type_distribution(pref_val, muni_val)
                retention_data = get_qualification_retention_rates(pref_val, muni_val)
                # 都道府県/市区町村フローTop10
                print(f"[DEBUG] Calling get_pref_flow_top10({pref_val})")
                pref_flow_list = get_pref_flow_top10(pref_val)
                print(f"[DEBUG] pref_flow_list = {pref_flow_list[:2] if pref_flow_list else 'empty'}")
                muni_flow_list = get_muni_flow_top10(pref_val, muni_val)
                print(f"[DEBUG] muni_flow_list = {muni_flow_list[:2] if muni_flow_list else 'empty'}")

                inflow = flow_data.get("inflow", 0)
                outflow = flow_data.get("outflow", 0)
                applicants = flow_data.get("applicant_count", 0)

                # 地元志向率 = (総求職者 - 流出) / 総求職者 × 100
                local_count = applicants - outflow if applicants > 0 else 0
                local_pct = (local_count / applicants * 100) if applicants > 0 else 0

                # 人材吸引力 = 流入 / 流出
                flow_ratio = f"{inflow / outflow:.2f}x" if outflow > 0 else "∞" if inflow > 0 else "N/A"

                # ========== カード1: 人材フロー分析 ==========
                with ui.card().classes("w-full mb-4").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 24px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("📊").classes("text-xl")
                        ui.label("人材フロー分析").classes("text-lg font-semibold").style(f"color: {TEXT_COLOR}")
                    ui.label("選択エリアへの就職希望者の流入・流出を分析").classes("text-sm mb-4").style(f"color: {MUTED_COLOR}")

                    if applicants > 0:
                        # 4つのKPI
                        with ui.row().classes("w-full gap-4 mb-4"):
                            # 流入（就職希望）
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(16, 185, 129, 0.1)"):
                                ui.label("流入（就職希望）").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(f"{inflow:,}").classes("text-2xl font-bold").style("color: #10b981")
                                    ui.label("人").classes("text-sm").style(f"color: {MUTED_COLOR}")
                            # 地元志向率
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(245, 158, 11, 0.1)"):
                                ui.label("地元志向率").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(f"{local_pct:.1f}").classes("text-2xl font-bold").style("color: #f59e0b")
                                    ui.label("%").classes("text-sm").style(f"color: {MUTED_COLOR}")
                                ui.label(f"({local_count:,}人)").classes("text-xs").style(f"color: {MUTED_COLOR}")
                            # 流出（他地域希望）
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(239, 68, 68, 0.1)"):
                                ui.label("流出（他地域希望）").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(f"{outflow:,}").classes("text-2xl font-bold").style("color: #ef4444")
                                    ui.label("人").classes("text-sm").style(f"color: {MUTED_COLOR}")
                            # 人材吸引力
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(59, 130, 246, 0.1)"):
                                ui.label("人材吸引力").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                ui.label(flow_ratio).classes("text-2xl font-bold").style(f"color: {PRIMARY_COLOR}")

                        # 流入元 / 流出先 2カラム
                        with ui.row().classes("w-full gap-4"):
                            # 流入元（どこから来るか）
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(16, 185, 129, 0.08)"):
                                with ui.row().classes("items-center gap-2 mb-2"):
                                    ui.element("div").classes("w-3 h-3 rounded-sm").style("background-color: #10b981")
                                    ui.label("流入元（どこから来るか）").classes("text-sm font-semibold").style(f"color: {TEXT_COLOR}")
                                if flow_sources:
                                    for item in flow_sources[:5]:
                                        with ui.row().classes("w-full items-center justify-between"):
                                            ui.label(item.get("name", "")).classes("text-sm").style(f"color: {TEXT_COLOR}")
                                            ui.label(f"{item.get('count', 0):,}人").classes("text-sm").style(f"color: {MUTED_COLOR}")
                                else:
                                    ui.label("市区町村を選択すると表示").classes("text-sm").style(f"color: {MUTED_COLOR}")

                            # 流出先（どこへ流れるか）
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(239, 68, 68, 0.08)"):
                                with ui.row().classes("items-center gap-2 mb-2"):
                                    ui.element("div").classes("w-3 h-3 rounded-sm").style("background-color: #ef4444")
                                    ui.label("流出先（どこへ流れるか）").classes("text-sm font-semibold").style(f"color: {TEXT_COLOR}")
                                if flow_destinations and outflow > 0:
                                    for item in flow_destinations[:5]:
                                        with ui.row().classes("w-full items-center justify-between"):
                                            ui.label(item.get("name", "")).classes("text-sm").style(f"color: {TEXT_COLOR}")
                                            ui.label(f"{item.get('count', 0):,}人").classes("text-sm").style(f"color: {MUTED_COLOR}")
                                else:
                                    ui.label("流出データなし（地元志向が高いエリアです）").classes("text-sm").style(f"color: {MUTED_COLOR}")
                    else:
                        ui.label("市区町村を選択すると人材フローを表示します").classes("text-sm").style(f"color: {MUTED_COLOR}")

                # ========== カード2: 居住地→希望地フロー ==========
                with ui.card().classes("w-full mb-4").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 24px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("🔀").classes("text-xl")
                        ui.label("居住地→希望地フロー").classes("text-lg font-semibold").style(f"color: {TEXT_COLOR}")
                    ui.label("現住所からどこへ移動したいかの流れを可視化").classes("text-sm mb-4").style(f"color: {MUTED_COLOR}")

                    with ui.row().classes("w-full gap-4"):
                        # 都道府県フローTop10（リスト形式）
                        with ui.element("div").classes("flex-1 p-4 rounded-lg").style(f"border: 1px solid {BORDER_COLOR}; background-color: rgba(255, 255, 255, 0.03)"):
                            ui.label("都道府県間の移動フロー Top10").classes("text-sm font-semibold mb-2").style(f"color: {TEXT_COLOR}")
                            # 新しい関数を使ってRESIDENCE_FLOWからフローデータを取得
                            if pref_flow_list:
                                for item in pref_flow_list:
                                    with ui.row().classes("w-full items-center"):
                                        ui.label(str(item.get("origin", ""))).classes("text-sm font-medium").style(f"color: {PRIMARY_COLOR}")
                                        ui.label("→").classes("text-sm mx-1").style(f"color: {MUTED_COLOR}")
                                        ui.label(str(item.get("destination", ""))).classes("text-sm font-medium").style(f"color: {SECONDARY_COLOR}")
                                        ui.element("div").classes("flex-grow")
                                        ui.label(f"{item.get('count', 0):,}件").classes("text-sm").style(f"color: {MUTED_COLOR}")
                            else:
                                ui.label("フローデータがありません").classes("text-sm").style(f"color: {MUTED_COLOR}")

                        # 市区町村フローTop10（リスト形式）
                        with ui.column().classes("flex-1 p-4 rounded-lg").style(f"border: 1px solid {BORDER_COLOR}; background-color: rgba(255, 255, 255, 0.03)"):
                            ui.label("市区町村間の移動フロー Top10").classes("text-sm font-semibold mb-2").style(f"color: {TEXT_COLOR}")
                            # 市区町村フローデータを取得して表示
                            muni_flow_list = get_muni_flow_top10(pref_val, muni_val)
                            if muni_flow_list:
                                for item in muni_flow_list:
                                    with ui.row().classes("w-full items-center"):
                                        ui.label(str(item.get("origin", ""))).classes("text-sm font-medium").style(f"color: {PRIMARY_COLOR}")
                                        ui.label("→").classes("text-sm mx-1").style(f"color: {MUTED_COLOR}")
                                        ui.label(str(item.get("destination", ""))).classes("text-sm font-medium").style(f"color: {SECONDARY_COLOR}")
                                        ui.element("div").classes("flex-grow")
                                        ui.label(f"{item.get('count', 0):,}件").classes("text-sm").style(f"color: {MUTED_COLOR}")
                            else:
                                ui.label("市区町村を選択するとフローを表示").classes("text-sm").style(f"color: {MUTED_COLOR}")

                # ========== カード3: 地域サマリー（COMPETITION） ==========
                with ui.card().classes("w-full mb-4").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 24px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("📊").classes("text-xl")
                        ui.label("地域サマリー").classes("text-lg font-semibold").style(f"color: {TEXT_COLOR}")
                    ui.label("選択地域の人材プロファイル概要").classes("text-sm mb-4").style(f"color: {MUTED_COLOR}")

                    if inflow > 0:
                        with ui.row().classes("w-full gap-4"):
                            # 総求職者数（流入数を使用）
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(59, 130, 246, 0.1)"):
                                ui.label("総求職者数").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                ui.label(f"{inflow:,}人").classes("text-xl font-bold").style(f"color: {TEXT_COLOR}")
                            # 女性比率
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(230, 159, 0, 0.1)"):
                                ui.label("女性比率").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                ui.label(competition_data.get("female_ratio", "-")).classes("text-xl font-bold").style("color: #E69F00")
                            # 主要年齢層
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(99, 102, 241, 0.1)"):
                                ui.label("主要年齢層").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                ui.label(competition_data.get("top_age", "-")).classes("text-lg font-bold").style(f"color: {PRIMARY_COLOR}")
                                ui.label(f"({competition_data.get('top_age_ratio', '-')})").classes("text-xs").style(f"color: {MUTED_COLOR}")
                            # 平均資格数
                            with ui.element("div").classes("flex-1 p-4 rounded-lg").style("background-color: rgba(16, 185, 129, 0.1)"):
                                ui.label("平均資格数").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(competition_data.get("avg_qualification_count", "-")).classes("text-xl font-bold").style(f"color: {ACCENT_GREEN}")
                                    ui.label("個").classes("text-sm").style(f"color: {MUTED_COLOR}")
                    else:
                        ui.label("地域データがありません").classes("text-sm").style(f"color: {MUTED_COLOR}")

                # ========== カード4: 移動パターン分布 ==========
                with ui.card().classes("w-full mb-4").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 24px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("🚗").classes("text-xl")
                        ui.label("移動パターン分布").classes("text-lg font-semibold").style(f"color: {TEXT_COLOR}")
                    ui.label("居住地から希望勤務地までの移動距離の傾向").classes("text-sm mb-4").style(f"color: {MUTED_COLOR}")

                    if mobility_dist:
                        # 棒グラフ
                        chart_data = [{"type": item.get("type", ""), "count": item.get("count", 0)} for item in mobility_dist]
                        types = [d["type"] for d in chart_data]
                        counts = [d["count"] for d in chart_data]
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis"},
                            "xAxis": {"type": "category", "data": types, "axisLabel": {"color": MUTED_COLOR}},
                            "yAxis": {"type": "value", "axisLabel": {"color": MUTED_COLOR}},
                            "series": [{"data": counts, "type": "bar", "itemStyle": {"color": PRIMARY_COLOR, "borderRadius": [8, 8, 0, 0]}}]
                        }).classes("w-full h-80")

                        # パーセンテージ表示
                        with ui.row().classes("w-full gap-2 mt-2"):
                            for item in mobility_dist:
                                with ui.element("div").classes("flex-1 p-2 rounded-md text-center").style("background-color: rgba(255, 255, 255, 0.05)"):
                                    ui.label(item.get("type", "")).classes("text-xs").style(f"color: {MUTED_COLOR}")
                                    ui.label(item.get("pct", "-")).classes("text-sm font-semibold").style(f"color: {TEXT_COLOR}")
                    else:
                        ui.label("移動パターンデータがありません").classes("text-sm").style(f"color: {MUTED_COLOR}")

                # ========== カード5: 距離統計 ==========
                with ui.card().classes("w-full mb-4").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 16px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("📏").classes("text-base")
                        ui.label("移動距離の統計").classes("text-sm font-semibold").style(f"color: {TEXT_COLOR}")

                    if dist_data:
                        with ui.row().classes("w-full gap-4"):
                            # Q25（25%点）
                            with ui.element("div").classes("flex-1 p-3 rounded-lg").style("background-color: rgba(20, 184, 166, 0.1)"):
                                ui.label("25%点").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(str(dist_data.get("q25", "-"))).classes("text-lg font-bold").style("color: #14b8a6")
                                    ui.label("km").classes("text-xs").style(f"color: {MUTED_COLOR}")
                            # 中央値
                            with ui.element("div").classes("flex-1 p-3 rounded-lg").style("background-color: rgba(99, 102, 241, 0.1)"):
                                ui.label("中央値").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(str(dist_data.get("median", "-"))).classes("text-lg font-bold").style(f"color: {PRIMARY_COLOR}")
                                    ui.label("km").classes("text-xs").style(f"color: {MUTED_COLOR}")
                            # Q75（75%点）
                            with ui.element("div").classes("flex-1 p-3 rounded-lg").style("background-color: rgba(236, 72, 153, 0.1)"):
                                ui.label("75%点").classes("text-xs").style(f"color: {MUTED_COLOR}")
                                with ui.row().classes("items-end gap-1"):
                                    ui.label(str(dist_data.get("q75", "-"))).classes("text-lg font-bold").style(f"color: {SECONDARY_COLOR}")
                                    ui.label("km").classes("text-xs").style(f"color: {MUTED_COLOR}")
                    else:
                        ui.label("距離データがありません").classes("text-sm").style(f"color: {MUTED_COLOR}")

                # ========== カード6: 資格別定着率 ==========
                with ui.card().classes("w-full").style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 24px"
                ):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("🏠").classes("text-xl")
                        ui.label("資格別定着率").classes("text-lg font-semibold").style(f"color: {TEXT_COLOR}")
                    ui.label("資格保有者の地元定着傾向（1.0以上＝地元志向）").classes("text-sm mb-4").style(f"color: {MUTED_COLOR}")

                    if retention_data:
                        with ui.scroll_area().classes("w-full").style("max-height: 350px"):
                            for item in retention_data:
                                rate = item.get("retention_rate", "-")
                                interp = item.get("interpretation", "平均的")
                                # 色を定着率に応じて変更
                                rate_color = ACCENT_GREEN if interp == "地元志向強" else (
                                    "#10b981" if interp == "地元志向" else (
                                        MUTED_COLOR if interp == "平均的" else "#f59e0b"
                                    )
                                )
                                badge_color = "green" if interp == "地元志向強" else (
                                    "blue" if interp == "地元志向" else (
                                        "gray" if interp == "平均的" else "red"
                                    )
                                )
                                with ui.row().classes("w-full items-center py-1"):
                                    ui.label(item.get("qualification", "")).classes("text-sm font-semibold").style(f"color: {TEXT_COLOR}; min-width: 120px")
                                    ui.element("div").classes("flex-grow")
                                    ui.label(str(rate)).classes("text-sm font-semibold").style(f"color: {rate_color}; min-width: 50px")
                                    ui.badge(interp, color=badge_color).classes("mx-2")
                                    ui.label(f"({item.get('count', 0):,}人)").classes("text-xs").style(f"color: {MUTED_COLOR}; min-width: 60px")

                        # 凡例
                        with ui.row().classes("w-full gap-2 mt-4 flex-wrap"):
                            ui.badge("≥1.1 地元志向強", color="green")
                            ui.badge("≥1.0 地元志向", color="blue")
                            ui.badge("≥0.9 平均的", color="gray")
                            ui.badge("<0.9 流出傾向", color="red")
                    else:
                        ui.label("定着率データがありません").classes("text-sm").style(f"color: {MUTED_COLOR}")

            elif tab == "balance":
                # ==========================================
                # 需給バランスタブ（Reflex完全再現版）
                # ==========================================
                ui.label("需給バランス").classes("text-xl font-bold mb-4").style(f"color: {TEXT_COLOR}")

                pref_val = state["prefecture"] if state["prefecture"] != "全国" else None
                muni_val = state["municipality"] if state["municipality"] != "すべて" else None
                gap_stats = get_gap_stats(pref_val, muni_val)
                gap_rankings = get_gap_rankings(pref_val, limit=10)

                # 選択地域表示
                with ui.row().classes("items-center gap-1 mb-4"):
                    ui.label("📍 選択中:").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                    ui.label(state["prefecture"]).style(f"color: {ACCENT_5}; font-weight: bold; font-size: 0.9rem")
                    if state["municipality"] and state["municipality"] != "すべて":
                        ui.label(" / ").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                        ui.label(state["municipality"]).style(f"color: {WARNING_COLOR}; font-weight: bold; font-size: 0.9rem")
                    else:
                        ui.label(" (都道府県全体)").style(f"color: {MUTED_COLOR}; font-size: 0.85rem; font-style: italic")

                # 5つのKPIカード
                with ui.row().classes("w-full gap-4 flex-wrap"):
                    for label_txt, value, unit in [
                        ("総需要", gap_stats["demand"], "件"),
                        ("総供給", gap_stats["supply"], "件"),
                        ("平均比率", gap_stats["ratio"], ""),
                        ("不足地域", gap_stats["shortage_count"], "箇所"),
                        ("過剰地域", gap_stats["surplus_count"], "箇所"),
                    ]:
                        with ui.card().style(
                            f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; "
                            f"border-radius: 12px; padding: 16px; flex: 1; min-width: 150px"
                        ):
                            ui.label(label_txt).classes("text-sm").style(f"color: {MUTED_COLOR}")
                            formatted = f"{value:,.0f}" if isinstance(value, (int, float)) and value == int(value) else f"{value:.2f}"
                            ui.label(f"{formatted}{unit}").classes("text-2xl font-bold").style(f"color: {PRIMARY_COLOR}")

                # ==========================================
                # 需要超過ランキング Top 10（横棒グラフ）
                # ==========================================
                with ui.card().style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; "
                    f"border-radius: 12px; padding: 24px; margin-top: 24px; width: 100%"
                ):
                    with ui.row().classes("items-baseline gap-2 mb-2"):
                        ui.label("需要超過ランキング Top 10").classes("text-lg font-bold").style(f"color: {TEXT_COLOR}")
                        with ui.row().classes("items-center gap-0"):
                            ui.label("（").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                            ui.label(state["prefecture"]).style(f"color: {ACCENT_5}; font-weight: bold; font-size: 0.9rem")
                            ui.label("内）").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                    ui.label("就業希望者数が居住者数を上回る市区町村（需要超過）").style(
                        f"color: {MUTED_COLOR}; font-size: 0.85rem; margin-bottom: 16px"
                    )

                    shortage_data = gap_rankings.get("shortage", [])
                    if shortage_data:
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "grid": {"left": "25%", "right": "10%", "top": "5%", "bottom": "15%"},
                            "xAxis": {
                                "type": "value",
                                "name": "需要超過（人）",
                                "nameLocation": "middle",
                                "nameGap": 30,
                                "axisLabel": {"color": MUTED_COLOR},
                                "nameTextStyle": {"color": MUTED_COLOR, "fontSize": 12}
                            },
                            "yAxis": {
                                "type": "category",
                                "data": [d["name"] for d in shortage_data][::-1],
                                "axisLabel": {"color": MUTED_COLOR, "fontSize": 11}
                            },
                            "series": [{
                                "type": "bar",
                                "data": [d["value"] for d in shortage_data][::-1],
                                "itemStyle": {"color": WARNING_COLOR, "borderRadius": [0, 8, 8, 0]},
                                "barWidth": 25
                            }]
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}; text-align: center; padding: 48px")

                # ==========================================
                # 供給超過ランキング Top 10（横棒グラフ）
                # ==========================================
                with ui.card().style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; "
                    f"border-radius: 12px; padding: 24px; margin-top: 24px; width: 100%"
                ):
                    with ui.row().classes("items-baseline gap-2 mb-2"):
                        ui.label("供給超過ランキング Top 10").classes("text-lg font-bold").style(f"color: {TEXT_COLOR}")
                        with ui.row().classes("items-center gap-0"):
                            ui.label("（").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                            ui.label(state["prefecture"]).style(f"color: {SUCCESS_COLOR}; font-weight: bold; font-size: 0.9rem")
                            ui.label("内）").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                    ui.label("居住者数が就業希望者数を上回る市区町村（供給超過）").style(
                        f"color: {MUTED_COLOR}; font-size: 0.85rem; margin-bottom: 16px"
                    )

                    surplus_data = gap_rankings.get("surplus", [])
                    if surplus_data:
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "grid": {"left": "25%", "right": "10%", "top": "5%", "bottom": "15%"},
                            "xAxis": {
                                "type": "value",
                                "name": "供給超過（人）",
                                "nameLocation": "middle",
                                "nameGap": 30,
                                "axisLabel": {"color": MUTED_COLOR},
                                "nameTextStyle": {"color": MUTED_COLOR, "fontSize": 12}
                            },
                            "yAxis": {
                                "type": "category",
                                "data": [d["name"] for d in surplus_data][::-1],
                                "axisLabel": {"color": MUTED_COLOR, "fontSize": 11}
                            },
                            "series": [{
                                "type": "bar",
                                "data": [d["value"] for d in surplus_data][::-1],
                                "itemStyle": {"color": SUCCESS_COLOR, "borderRadius": [0, 8, 8, 0]},
                                "barWidth": 25
                            }]
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}; text-align: center; padding: 48px")

                # ==========================================
                # 需給比率ランキング Top 10（横棒グラフ）
                # ==========================================
                with ui.card().style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; "
                    f"border-radius: 12px; padding: 24px; margin-top: 24px; width: 100%"
                ):
                    with ui.row().classes("items-baseline gap-2 mb-2"):
                        ui.label("需給比率ランキング Top 10").classes("text-lg font-bold").style(f"color: {TEXT_COLOR}")
                        with ui.row().classes("items-center gap-0"):
                            ui.label("（").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                            ui.label(state["prefecture"]).style(f"color: {ACCENT_5}; font-weight: bold; font-size: 0.9rem")
                            ui.label("内）").style(f"color: {MUTED_COLOR}; font-size: 0.9rem")
                    ui.label("需要/供給の比率が高い市区町村（採用競争激化）").style(
                        f"color: {MUTED_COLOR}; font-size: 0.85rem; margin-bottom: 16px"
                    )

                    ratio_data = gap_rankings.get("ratio", [])
                    if ratio_data:
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "grid": {"left": "25%", "right": "10%", "top": "5%", "bottom": "15%"},
                            "xAxis": {
                                "type": "value",
                                "name": "需給比率",
                                "nameLocation": "middle",
                                "nameGap": 30,
                                "axisLabel": {"color": MUTED_COLOR},
                                "nameTextStyle": {"color": MUTED_COLOR, "fontSize": 12}
                            },
                            "yAxis": {
                                "type": "category",
                                "data": [d["name"] for d in ratio_data][::-1],
                                "axisLabel": {"color": MUTED_COLOR, "fontSize": 11}
                            },
                            "series": [{
                                "type": "bar",
                                "data": [d["value"] for d in ratio_data][::-1],
                                "itemStyle": {"color": ACCENT_5, "borderRadius": [0, 8, 8, 0]},
                                "barWidth": 25
                            }]
                        }).classes("w-full h-96")
                    else:
                        ui.label("データがありません").style(f"color: {MUTED_COLOR}; text-align: center; padding: 48px")

                # ==========================================
                # 説明パネル
                # ==========================================
                with ui.card().style(
                    f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; "
                    f"border-radius: 12px; padding: 24px; margin-top: 24px; width: 100%"
                ):
                    ui.label("指標の説明").classes("text-lg font-bold mb-4").style(f"color: {TEXT_COLOR}")
                    for desc in [
                        "総需要: 地域内で必要とされる人材数",
                        "総供給: 地域内で利用可能な人材数",
                        "平均比率: 需要 ÷ 供給の平均（比率が高いほど人材獲得が困難）",
                        "不足地域: 需要 > 供給の市区町村数（採用難易度が高い地域）",
                        "過剰地域: 供給 > 需要の市区町村数（人材が余剰している地域）",
                    ]:
                        ui.label(desc).style(f"color: {MUTED_COLOR}; font-size: 0.85rem; margin-bottom: 8px")

            elif tab == "jobmap":
                ui.label("求人地図（GAS連携）").classes("text-lg font-bold mb-4").style(f"color: {TEXT_COLOR}")
                gas_urls = {
                    "介護職": "https://script.google.com/macros/s/AKfycbxd--YaAomrsCpqaLyB40XkTlVOt17bqulrddPVCoFBAOw1FDE7r8mYHMRSKT25D9t7/exec",
                }
                current_job = state.get("jobmap_jobtype", list(gas_urls.keys())[0])
                if current_job not in gas_urls:
                    current_job = list(gas_urls.keys())[0]
                    state["jobmap_jobtype"] = current_job

                def on_job_change(e):
                    state["jobmap_jobtype"] = e.value if hasattr(e, "value") else e.args
                    ui.notify(f"Job: {state['jobmap_jobtype']}")

                ui.select(
                    options=list(gas_urls.keys()),
                    value=current_job,
                    label="職種",
                    on_change=on_job_change,
                ).classes("w-64").props(
                    f'outlined dense color=white text-color=white label-color="{MUTED_COLOR}" popup-content-class="bg-blue-grey-10 text-white"'
                ).style(f"color: {TEXT_COLOR}")

                ui.html(
                    content=f"""
                    <iframe
                        id="jobmap-iframe"
                        src="{gas_urls[current_job]}"
                        width="100%"
                        height="800px"
                        frameborder="0"
                        style="border: 1px solid {BORDER_COLOR}; border-radius: 8px; background: white; min-height: 600px;"
                        allow="geolocation"
                    ></iframe>
                    """,
                    sanitize=False
                ).classes("w-full")

    # Tabs（Reflexと同じ日本語タブ名）
    tab_names = ["📊 市場概況", "👥 人材属性", "🗺️ 地域・移動パターン", "⚖️ 需給バランス", "🗺️ 求人地図"]
    tab_ids = ["overview", "demographics", "mobility", "balance", "jobmap"]

    with ui.row().classes("w-full justify-center gap-2 mb-4 p-2").style(f"background-color: {PANEL_BG}"):
        tab_buttons = []

        def create_tab_click_handler(tab_id: str):
            def handler():
                state["tab"] = tab_id
                for btn, btn_id in tab_buttons:
                    if btn_id == tab_id:
                        btn.style(f"background-color: {PRIMARY_COLOR}; color: white")
                    else:
                        btn.style(f"background-color: {CARD_BG}; color: {TEXT_COLOR}")
                show_content.refresh()

            return handler

        for name, tab_id in zip(tab_names, tab_ids):
            if tab_id == state.get("tab", "overview"):
                btn = ui.button(name, on_click=create_tab_click_handler(tab_id)).style(
                    f"background-color: {PRIMARY_COLOR}; color: white"
                )
            else:
                btn = ui.button(name, on_click=create_tab_click_handler(tab_id)).style(
                    f"background-color: {CARD_BG}; color: {TEXT_COLOR}"
                )
            tab_buttons.append((btn, tab_id))

    with ui.card().classes("w-full").style(f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}"):
        show_content()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ in {"__main__", "__mp_main__"}:
    port = int(os.getenv("PORT", 9090))
    is_production = os.getenv("RENDER") is not None or os.getenv("PORT") is not None
    storage_secret = os.getenv("NICEGUI_STORAGE_SECRET", "nicegui_mapcomplete_secret_key_2025")

    print(f"[STARTUP] Starting NiceGUI app on port {port}...")
    print(f"[STARTUP] Production mode: {is_production}")

    ui.run(
        title="MapComplete Dashboard",
        host="0.0.0.0",
        port=port,
        reload=not is_production,
        storage_secret=storage_secret,
        show=False,
        reconnect_timeout=30.0,
    )
