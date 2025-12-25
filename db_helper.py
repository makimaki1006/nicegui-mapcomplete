# -*- coding: utf-8 -*-
# Force reload: 2025-12-24 v2 - Fix qualification retention count
"""
ハイブリッドデータベースアクセスヘルパー（CSV + Turso + SQLite + PostgreSQL対応）

DashboardStateでのデータベースアクセスを簡潔にするヘルパー関数群。
環境変数で自動切り替え：
- USE_CSV_MODE=true → CSV直接読み込み（Reflex Cloud推奨）
- TURSO_DATABASE_URL設定あり → Turso使用
- DATABASE_URL設定あり → PostgreSQL使用
- どちらも未設定 → SQLite使用
"""

import os
import sys
import sqlite3
import asyncio
import pandas as pd
import httpx  # requests から置き換え（非同期対応）
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta

print("=" * 60)
print("[STARTUP] db_helper.py loading...")
print(f"[STARTUP] Python version: {sys.version}")
print(f"[STARTUP] Current directory: {os.getcwd()}")
print("=" * 60)

# =====================================
# Reflex Cloud用環境変数設定（Secrets代替）
# =====================================
# 注意: setdefaultではなく直接代入で既存のSecrets設定を上書き
if os.getenv("REFLEX_DEPLOYMENT") is not None:
    print("[STARTUP] Reflex Cloud detected - forcing hardcoded env vars")
    os.environ["TURSO_DATABASE_URL"] = "libsql://job-jobseekeranalyzer-makimaki1006.aws-ap-northeast-1.turso.io"
    os.environ["TURSO_AUTH_TOKEN"] = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjM3ODk2NzgsImlkIjoiNTU1YzQwMzgtZjI0MS00ODAwLTgzYTctYmE4MmYzOGQyZGFhIiwicmlkIjoiNjJmNTFkMjgtNTdhNC00M2Y2LWIzZTAtYmRlZGZiM2YxNmY1In0.Q-WMFowjDxhbdUqkeKYG7y4qkhfbsFaVgqKtAyU0y2bdCTNCHOp8IQsHFYgJ6wdT4m96WIXtP4UO-jRbB8woBg"

# dotenvのインポート（エラーハンドリング付き）
try:
    from dotenv import load_dotenv

    # .env.production を優先的に読み込む（Reflex Cloud用）
    env_production = Path(__file__).parent / ".env.production"
    if env_production.exists():
        load_dotenv(env_production)
        print(f"[STARTUP] .env.production loaded: {env_production}")
    else:
        # フォールバック: デフォルトの.env
        load_dotenv()
        print("[STARTUP] .env loaded (default)")
except Exception as e:
    print(f"[STARTUP] dotenv load failed: {e}")
    # dotenvなしでも続行（Reflex Cloudではsecretsから読み込み）

# CSVモード（Reflex Cloud用、外部DB不要）
# ハイブリッドモード: CSVをデフォルト、Tursoはオンデマンド
_csv_mode_env = os.getenv("USE_CSV_MODE", "").lower()
# Reflex Cloud環境ではデフォルトでCSVモード（起動を確実にする）
is_reflex_cloud = os.getenv("REFLEX_DEPLOYMENT") is not None
USE_CSV_MODE = _csv_mode_env == "true" or (_csv_mode_env == "" and is_reflex_cloud)
print(f"[STARTUP] USE_CSV_MODE = {USE_CSV_MODE} (env: '{_csv_mode_env}', cloud: {is_reflex_cloud})")

# CSVファイル名
CSV_FILENAME = "MapComplete_Complete_All_FIXED.csv"
CSV_FILENAME_GZ = "MapComplete_Complete_All_FIXED.csv.gz"

# CSVデータのグローバルキャッシュ（起動時に1回だけ読み込み）
_csv_dataframe: Optional[pd.DataFrame] = None

def _find_csv_path() -> tuple[Optional[Path], bool]:
    """
    CSVファイルを複数の場所から探す（Reflex Cloud対応）
    Returns: (path, is_gzip)
    """
    # 探す場所のリスト（優先順位順）
    search_dirs = [
        Path(__file__).parent,  # db_helper.pyと同じディレクトリ
        Path(__file__).parent.parent,  # 親ディレクトリ
        Path.cwd(),  # カレントディレクトリ
        Path("/app"),  # Reflex Cloud標準パス
        Path("/app/mapcomplete_dashboard"),  # サブディレクトリ
    ]

    # gzip版を優先して探す
    for dir_path in search_dirs:
        gz_path = dir_path / CSV_FILENAME_GZ
        if gz_path.exists():
            print(f"[CSV] Found gzip at: {gz_path}")
            return gz_path, True

    # 通常のCSVを探す
    for dir_path in search_dirs:
        csv_path = dir_path / CSV_FILENAME
        if csv_path.exists():
            print(f"[CSV] Found CSV at: {csv_path}")
            return csv_path, False

    # 見つからない場合はデバッグ情報を出力
    print(f"[CSV] WARNING: CSV not found. Searched directories:")
    for dir_path in search_dirs:
        print(f"  - {dir_path} (exists: {dir_path.exists()})")
        if dir_path.exists():
            try:
                files = list(dir_path.iterdir())[:10]
                print(f"    Files: {[f.name for f in files]}")
            except Exception as e:
                print(f"    Error listing: {e}")

    return None, False

def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのdtypeを最適化してメモリ使用量を削減

    object型カラムのうち、ユニーク値が全行の50%未満のものをcategory型に変換。
    これにより約68%のメモリ削減が可能（299MB → 97MB）。

    更新履歴:
    - 2025-12-22: 追加（DEPLOYMENT_INVESTIGATION_20251222.md Phase 1-2）
    """
    mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024
    converted_cols = []

    for col in df.select_dtypes(include=['object']).columns:
        # ユニーク値が全行の50%未満ならcategoryに変換
        if df[col].nunique() < len(df) * 0.5:
            df[col] = df[col].astype('category')
            converted_cols.append(col)

    mem_after = df.memory_usage(deep=True).sum() / 1024 / 1024
    reduction = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0

    print(f"[CSV] dtype optimization: {mem_before:.1f}MB -> {mem_after:.1f}MB ({reduction:.0f}% reduction)")
    print(f"[CSV] Converted {len(converted_cols)} columns to category")

    return df


def _load_csv_data() -> pd.DataFrame:
    """CSVデータを読み込み（キャッシュ付き、gzip圧縮対応、複数パス検索、dtype最適化）

    更新履歴:
    - 2025-12-22: dtype最適化追加（メモリ68%削減）
    """
    global _csv_dataframe
    if _csv_dataframe is None:
        csv_path, is_gzip = _find_csv_path()

        if csv_path is None:
            raise FileNotFoundError(
                f"CSVファイルが見つかりません: {CSV_FILENAME_GZ} または {CSV_FILENAME}\n"
                "USE_CSV_MODE=true の場合、CSVファイルをデプロイパッケージに含めてください。"
            )

        if is_gzip:
            print(f"[CSV] Loading compressed data from {csv_path}...")
            _csv_dataframe = pd.read_csv(csv_path, encoding='utf-8-sig', compression='gzip', low_memory=False)
        else:
            print(f"[CSV] Loading data from {csv_path}...")
            _csv_dataframe = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)

        print(f"[CSV] Loaded {len(_csv_dataframe):,} rows")

        # dtype最適化（メモリ68%削減）
        _csv_dataframe = _optimize_dtypes(_csv_dataframe)

    return _csv_dataframe

# Turso環境変数
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
print(f"[STARTUP] TURSO_DATABASE_URL set: {bool(TURSO_DATABASE_URL)} (len={len(TURSO_DATABASE_URL)})")
print(f"[STARTUP] TURSO_AUTH_TOKEN set: {bool(TURSO_AUTH_TOKEN)} (len={len(TURSO_AUTH_TOKEN)})")

# PostgreSQL環境変数
DATABASE_URL = os.getenv("DATABASE_URL")
print(f"[STARTUP] DATABASE_URL set: {bool(DATABASE_URL)}")

# SQLiteパス（ローカル開発用）
DB_PATH = Path(__file__).parent / "data" / "job_medley.db"
print(f"[STARTUP] SQLite fallback path: {DB_PATH}")

# データディレクトリ自動作成（SQLiteフォールバック対策）
try:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] Data directory ensured: {DB_PATH.parent}")
except Exception as e:
    print(f"[STARTUP] Data directory creation failed: {e}")

# Turso HTTP API モード
# 本番環境ではTurso設定があれば即座にTrueに設定（遅延初期化の問題を回避）
_HAS_TURSO = bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN)
_HAS_LIBSQL_CLIENT = False
_TURSO_INIT_ERROR = None
_TURSO_INITIALIZED = _HAS_TURSO  # 設定があれば初期化済みとみなす

def _lazy_init_turso():
    """Turso接続を遅延初期化（最初のクエリ時に呼び出し）"""
    global _HAS_TURSO, _HAS_LIBSQL_CLIENT, _TURSO_INIT_ERROR, _TURSO_INITIALIZED, TURSO_DATABASE_URL

    if _TURSO_INITIALIZED:
        return _HAS_TURSO

    _TURSO_INITIALIZED = True
    print(f"[TURSO] Lazy init: URL={bool(TURSO_DATABASE_URL)}, TOKEN={bool(TURSO_AUTH_TOKEN)}")

    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            # HTTP APIでTursoに接続（libsql_clientは不要）
            _HAS_TURSO = True
            # libsql:// を https:// に変換
            if TURSO_DATABASE_URL.startswith('libsql://'):
                TURSO_DATABASE_URL = TURSO_DATABASE_URL.replace('libsql://', 'https://')
            print(f"[TURSO] HTTP API mode enabled: {TURSO_DATABASE_URL[:50]}...")
        except Exception as init_error:
            _HAS_TURSO = False
            _TURSO_INIT_ERROR = str(init_error)
            print(f"[TURSO] Initialization failed (will use CSV fallback): {init_error}")

    return _HAS_TURSO

# 起動時はTurso設定の存在確認のみ（接続テストは行わない）
print(f"[STARTUP] Turso config available: URL={bool(TURSO_DATABASE_URL)}, TOKEN={bool(TURSO_AUTH_TOKEN)}")

# PostgreSQL接続用（必要な場合のみimport）
_HAS_POSTGRES = False
if DATABASE_URL and not _HAS_TURSO:
    try:
        import psycopg2
        import psycopg2.extras
        _HAS_POSTGRES = True
    except ImportError:
        print("WARNING: psycopg2 not installed. Install with: pip install psycopg2-binary")

# ===== STARTUP SUMMARY =====
print("=" * 60)
print("[STARTUP] db_helper.py SUMMARY:")
print(f"  USE_CSV_MODE: {USE_CSV_MODE}")
print(f"  Turso config available: {bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN)}")
print(f"  _HAS_POSTGRES: {_HAS_POSTGRES}")
print(f"  Initial mode: {'CSV (Turso on-demand)' if USE_CSV_MODE else 'Database'}")
print("=" * 60)

# =====================================
# 永続キャッシュ設定（Turso読み書き最適化）
# =====================================
# 目的: 各ユーザーのアクセス毎にDBクエリを繰り返さないよう、
#       アプリケーションレベルでキャッシュを共有
#
# キャッシュ戦略:
# 1. 都道府県リスト: 永続キャッシュ（アプリ再起動まで保持）
# 2. 市区町村リスト: 永続キャッシュ（都道府県ごと）
# 3. フィルタ済みデータ: 永続キャッシュ（prefecture+municipality単位）
# 4. 全ユーザーで共有: 同じ地域へのアクセスは追加DBクエリなし
# 5. 明示的更新: refresh_all_cache() で全キャッシュをクリア&再読み込み
#
# これにより、例えば100人が同時に「東京都」を選択しても、
# DBクエリは最初の1人目の1回のみ

_cache: dict = {}
_cache_time: dict = {}
_max_cache_items = 100  # メモリ最適化: 100件に削減（Render 512MB対応）
_ttl_minutes = 30  # メモリ最適化: 30分に短縮

# 永続キャッシュ（TTLなし、明示的にクリアするまで保持）
_static_cache: dict = {
    "prefectures": None,  # 都道府県リスト
    "municipalities": {},  # 都道府県→市区町村リストのマッピング
    "filtered_data": {},  # prefecture_municipality → DataFrame
}
_cache_initialized: bool = False

# 都道府県の標準順序（JISコード順：北から南）
PREFECTURE_ORDER = [
    "北海道",
    "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]

def _sort_prefectures(prefectures: list) -> list:
    """都道府県リストを標準順序（北から南）でソート"""
    order_map = {pref: i for i, pref in enumerate(PREFECTURE_ORDER)}
    # 標準リストにない都道府県は末尾に配置
    return sorted(prefectures, key=lambda x: order_map.get(x, 999))


def get_db_type() -> str:
    """使用中のデータベースタイプを取得"""
    if USE_CSV_MODE:
        return "csv"
    # Tursoは遅延初期化（最初のクエリ時に初期化）
    elif _lazy_init_turso():
        return "turso"
    elif DATABASE_URL and _HAS_POSTGRES:
        return "postgresql"
    else:
        return "sqlite"


def get_connection() -> Union[sqlite3.Connection, "psycopg2.extensions.connection", object]:
    """データベース接続を取得（環境変数で自動切り替え）"""
    if _HAS_TURSO:
        # Tursoはasyncベースなので、ダミー接続を返す
        class TursoDummyConnection:
            def close(self):
                pass
        return TursoDummyConnection()
    elif DATABASE_URL and _HAS_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    else:
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"データベースファイルが見つかりません: {DB_PATH}\n"
                f"migrate_csv_to_db.py を実行してデータベースを作成してください。"
            )
        return sqlite3.connect(str(DB_PATH))


def _build_turso_args(params: list) -> list:
    """Turso HTTP API用のパラメータ配列を構築（SQLインジェクション対策）

    Turso HTTP API v2は正式なパラメータバインディングをサポート。
    手動エスケープではなく、API側でサニタイズされる安全な方法。

    Args:
        params: パラメータリスト

    Returns:
        Turso args形式のリスト [{"type": "text", "value": "..."}, ...]
    """
    if not params:
        return []

    args = []
    for param in params:
        if param is None:
            args.append({"type": "null"})
        elif isinstance(param, bool):
            args.append({"type": "integer", "value": str(1 if param else 0)})
        elif isinstance(param, int):
            args.append({"type": "integer", "value": str(param)})
        elif isinstance(param, float):
            args.append({"type": "float", "value": str(param)})
        elif isinstance(param, bytes):
            import base64
            args.append({"type": "blob", "base64": base64.b64encode(param).decode()})
        else:
            # 文字列として扱う（Turso側でサニタイズ）
            args.append({"type": "text", "value": str(param)})

    return args


def _turso_http_query(sql: str, params: list = None, max_retries: int = 2) -> tuple:
    """Turso HTTP APIクエリ実行（httpx同期版、リトライ付き）

    改善点（2025-12-22）:
    - requests → httpx（非同期対応の基盤）
    - タイムアウト: 60秒 → 10秒（短縮でブロッキング軽減）
    - HTTPステータスコードチェック追加
    - リトライロジック追加（最大2回）
    - パラメータバインディング実装（SQLインジェクション対策）
    """
    # HTTPSのURLを構築
    http_url = TURSO_DATABASE_URL
    if http_url.startswith('libsql://'):
        http_url = http_url.replace('libsql://', 'https://')

    headers = {
        'Authorization': f'Bearer {TURSO_AUTH_TOKEN}',
        'Content-Type': 'application/json'
    }

    # パラメータバインディング（SQLインジェクション対策）
    # Turso HTTP API v2はargs配列でパラメータを渡す
    stmt = {'sql': sql}
    if params:
        stmt['args'] = _build_turso_args(params)

    payload = {
        'requests': [
            {'type': 'execute', 'stmt': stmt}
        ]
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f'{http_url}/v2/pipeline',
                    headers=headers,
                    json=payload
                )

                # HTTPステータスコードチェック
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response
                    )

                data = response.json()

            if not data.get('results'):
                return [], []

            result = data['results'][0]
            if result.get('type') == 'error':
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Turso query error: {error_msg}")

            resp = result['response']['result']
            columns = [c['name'] for c in resp['cols']]

            rows = []
            for row in resp['rows']:
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    if isinstance(val, dict):
                        row_dict[col] = val.get('value')
                    else:
                        row_dict[col] = val
                rows.append(row_dict)

            return rows, columns

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if attempt < max_retries:
                print(f"[Turso] Retry {attempt + 1}/{max_retries} after error: {e}")
                continue
            raise
        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < max_retries and e.response.status_code >= 500:
                print(f"[Turso] Retry {attempt + 1}/{max_retries} after HTTP {e.response.status_code}")
                continue
            raise

    raise last_error if last_error else Exception("Turso query failed")


async def _turso_async_query(sql: str, params: list = None, max_retries: int = 2) -> tuple:
    """Turso非同期クエリ実行（httpx.AsyncClient使用）

    改善点（2025-12-22）:
    - 真の非同期実装（asyncio対応）
    - タイムアウト: 10秒
    - リトライロジック追加
    - パラメータバインディング実装（SQLインジェクション対策）
    """
    http_url = TURSO_DATABASE_URL
    if http_url.startswith('libsql://'):
        http_url = http_url.replace('libsql://', 'https://')

    headers = {
        'Authorization': f'Bearer {TURSO_AUTH_TOKEN}',
        'Content-Type': 'application/json'
    }

    # パラメータバインディング（SQLインジェクション対策）
    stmt = {'sql': sql}
    if params:
        stmt['args'] = _build_turso_args(params)

    payload = {
        'requests': [
            {'type': 'execute', 'stmt': stmt}
        ]
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f'{http_url}/v2/pipeline',
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response
                    )

                data = response.json()

            if not data.get('results'):
                return [], []

            result = data['results'][0]
            if result.get('type') == 'error':
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Turso query error: {error_msg}")

            resp = result['response']['result']
            columns = [c['name'] for c in resp['cols']]

            rows = []
            for row in resp['rows']:
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    if isinstance(val, dict):
                        row_dict[col] = val.get('value')
                    else:
                        row_dict[col] = val
                rows.append(row_dict)

            return rows, columns

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if attempt < max_retries:
                print(f"[Turso] Async retry {attempt + 1}/{max_retries} after error: {e}")
                await asyncio.sleep(0.5)  # 短い待機
                continue
            raise
        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < max_retries and e.response.status_code >= 500:
                print(f"[Turso] Async retry {attempt + 1}/{max_retries} after HTTP {e.response.status_code}")
                await asyncio.sleep(0.5)
                continue
            raise

    raise last_error if last_error else Exception("Turso async query failed")


def _convert_sql_placeholders(sql: str, db_type: str) -> str:
    """SQLプレースホルダーをDB種別に応じて変換"""
    if db_type == "postgresql":
        return sql.replace("?", "%s")
    return sql


def query_df(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """SQLクエリを実行してDataFrameとして取得（全DB対応）

    改善点（2025-12-22）:
    - ThreadPoolExecutorの複雑なロジックを削除
    - 同期版httpxを直接使用（シンプルで確実）
    - 例外処理の改善（具体的なエラーメッセージ）
    """
    db_type = get_db_type()

    if db_type == "turso":
        # Turso用: 同期版を直接使用（シンプルで確実）
        try:
            params_list = list(params) if params else None
            rows, columns = _turso_http_query(sql, params_list)

            if not rows:
                return pd.DataFrame()

            return pd.DataFrame(rows, columns=columns)

        except httpx.TimeoutException as e:
            print(f"[ERROR] Turso timeout (10s): {e}")
            return pd.DataFrame()
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Turso HTTP error: {e.response.status_code}")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] Turso query failed: {type(e).__name__}: {e}")
            return pd.DataFrame()

    else:
        # SQLite/PostgreSQL用
        conn = get_connection()
        try:
            converted_sql = _convert_sql_placeholders(sql, db_type)
            if params:
                df = pd.read_sql_query(converted_sql, conn, params=params)
            else:
                df = pd.read_sql_query(converted_sql, conn)
            return df
        finally:
            conn.close()


def _get_cached(key: str, ttl_minutes: int = None):
    """キャッシュからデータを取得

    Args:
        key: キャッシュキー
        ttl_minutes: TTL（分）。Noneの場合はデフォルト（2時間）
    """
    if key not in _cache:
        return None
    elapsed = datetime.now() - _cache_time[key]
    effective_ttl = ttl_minutes if ttl_minutes is not None else _ttl_minutes
    if elapsed > timedelta(minutes=effective_ttl):
        del _cache[key]
        del _cache_time[key]
        return None
    return _cache[key]


def _set_cache(key: str, data):
    """キャッシュにデータを保存"""
    if len(_cache) >= _max_cache_items:
        oldest = min(_cache_time, key=_cache_time.get)
        del _cache[oldest]
        del _cache_time[oldest]
    _cache[key] = data
    _cache_time[key] = datetime.now()


def clear_cache():
    """全キャッシュをクリア（永続キャッシュ含む）+ ガベージコレクション"""
    import gc
    global _cache, _cache_time, _static_cache, _cache_initialized
    _cache = {}
    _cache_time = {}
    _static_cache = {
        "prefectures": None,
        "municipalities": {},
        "filtered_data": {},
    }
    _cache_initialized = False
    gc.collect()  # メモリ解放
    print("[CACHE] All cache cleared + gc.collect()")


def refresh_all_cache():
    """全キャッシュをクリアして再読み込み

    データベース更新後に呼び出すことで、全ユーザーに最新データを反映。
    使用例:
        from db_helper import refresh_all_cache
        refresh_all_cache()  # 全キャッシュクリア&都道府県リスト再読み込み
    """
    print("[CACHE] Refreshing all cache...")
    clear_cache()

    # 都道府県リストを事前読み込み（よく使うため）
    prefectures = get_prefectures()
    print(f"[CACHE] Reloaded {len(prefectures)} prefectures")

    return {"status": "success", "prefectures_count": len(prefectures)}


def get_cache_stats() -> dict:
    """キャッシュ統計情報を取得"""
    return {
        "prefectures_cached": _static_cache["prefectures"] is not None,
        "municipalities_cached": len(_static_cache["municipalities"]),
        "filtered_data_cached": len(_static_cache.get("filtered_data", {})),
        "legacy_cache_items": len(_cache),
    }


def get_table(table_name: str) -> pd.DataFrame:
    """テーブル全体をDataFrameとして取得"""
    return query_df(f"SELECT * FROM {table_name}")


def get_all_data() -> pd.DataFrame:
    """Turso job_seeker_data テーブルから全データを取得（キャッシュ対応）"""
    cache_key = "ALL_DATA"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    if _HAS_TURSO:
        df = query_df("SELECT * FROM job_seeker_data")
    else:
        df = query_df("SELECT * FROM mapcomplete_raw")

    _set_cache(cache_key, df)
    return df


def get_prefectures() -> list:
    """都道府県一覧を取得（北から南の標準順序）

    最適化: 静的キャッシュを使用。アプリ起動後最初のアクセスで1回だけDBクエリ。
    以降は全ユーザーでキャッシュを共有（DBアクセスゼロ）。

    フォールバック機構:
    - Turso接続失敗時は自動的にCSVから読み込み
    - 最大3回リトライ（指数バックオフ）
    """
    global _static_cache
    import time

    # 静的キャッシュにあれば即座に返す（DBアクセスなし）
    if _static_cache["prefectures"] is not None:
        return _static_cache["prefectures"]

    print("[DB] Fetching prefectures (first time only)...")
    result = []

    if USE_CSV_MODE:
        # CSVモード: 直接CSV読み込み
        try:
            df = _load_csv_data()
            prefectures = df['prefecture'].dropna().unique().tolist()
            result = _sort_prefectures(prefectures)
            print(f"[CSV] Loaded {len(result)} prefectures from CSV")
        except Exception as e:
            print(f"[ERROR] CSV load failed: {e}")
            result = []
    elif _HAS_TURSO:
        # Tursoモード: リトライ付きでDBクエリ、失敗時CSVフォールバック
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = query_df("SELECT DISTINCT prefecture FROM job_seeker_data")
                if not df.empty:
                    prefectures = df['prefecture'].tolist()
                    result = _sort_prefectures(prefectures)
                    print(f"[TURSO] Loaded {len(result)} prefectures (attempt {attempt + 1})")
                    break
            except Exception as e:
                print(f"[TURSO] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 0.5秒, 1秒, 2秒
                    time.sleep(wait_time)

        # Turso失敗時: CSVフォールバック
        if not result:
            print("[FALLBACK] Turso failed, trying CSV fallback...")
            try:
                csv_path, is_gzip = _find_csv_path()
                if csv_path:
                    if is_gzip:
                        df = pd.read_csv(csv_path, encoding='utf-8-sig', compression='gzip')
                    else:
                        df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    prefectures = df['prefecture'].dropna().unique().tolist()
                    result = _sort_prefectures(prefectures)
                    print(f"[FALLBACK] Loaded {len(result)} prefectures from CSV")
            except Exception as e2:
                print(f"[FALLBACK] CSV fallback also failed: {e2}")
                result = []
    else:
        # SQLite/PostgreSQLモード
        try:
            prefectures = get_unique_values("applicants", "residence_prefecture")
            result = _sort_prefectures(prefectures)
        except Exception as e:
            print(f"[ERROR] DB query failed: {e}")
            result = []

    # 静的キャッシュに保存（空でも保存して繰り返しクエリを防止）
    if result:
        _static_cache["prefectures"] = result
        print(f"[DB] Cached {len(result)} prefectures")
    else:
        print("[WARNING] No prefectures loaded - dropdown will be empty")

    return result


def get_municipalities(prefecture: str) -> list:
    """指定都道府県の市区町村一覧を取得

    最適化: 都道府県ごとに静的キャッシュ。
    同じ都道府県へのアクセスは全ユーザーでキャッシュ共有（DBアクセスゼロ）。

    フォールバック機構:
    - Turso接続失敗時は自動的にCSVから読み込み
    - リトライ付き（最大2回）
    """
    global _static_cache
    import time

    # 静的キャッシュにあれば即座に返す（DBアクセスなし）
    if prefecture in _static_cache["municipalities"]:
        return _static_cache["municipalities"][prefecture]

    print(f"[DB] Fetching municipalities for {prefecture} (first time only)...")
    result = []

    if USE_CSV_MODE:
        # CSVモード: 直接CSV読み込み
        try:
            df = _load_csv_data()
            filtered = df[df['prefecture'] == prefecture]
            municipalities = filtered['municipality'].dropna().unique().tolist()
            result = sorted(municipalities)
            print(f"[CSV] Loaded {len(result)} municipalities for {prefecture}")
        except Exception as e:
            print(f"[ERROR] CSV load failed: {e}")
            result = []
    elif _HAS_TURSO:
        # Tursoモード: リトライ付きクエリ、失敗時CSVフォールバック
        max_retries = 2
        for attempt in range(max_retries):
            try:
                df = query_df(
                    "SELECT DISTINCT municipality FROM job_seeker_data WHERE prefecture = ? AND municipality IS NOT NULL ORDER BY municipality",
                    (prefecture,)
                )
                if not df.empty:
                    result = df['municipality'].tolist()
                    print(f"[TURSO] Loaded {len(result)} municipalities for {prefecture} (attempt {attempt + 1})")
                    break
                else:
                    result = []
                    break
            except Exception as e:
                print(f"[TURSO] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))

        # Turso失敗時: CSVフォールバック
        if not result:
            print(f"[FALLBACK] Turso failed for {prefecture}, trying CSV fallback...")
            try:
                csv_path, is_gzip = _find_csv_path()
                if csv_path:
                    if is_gzip:
                        df = pd.read_csv(csv_path, encoding='utf-8-sig', compression='gzip')
                    else:
                        df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    filtered = df[df['prefecture'] == prefecture]
                    municipalities = filtered['municipality'].dropna().unique().tolist()
                    result = sorted(municipalities)
                    print(f"[FALLBACK] Loaded {len(result)} municipalities for {prefecture} from CSV")
            except Exception as e2:
                print(f"[FALLBACK] CSV fallback also failed: {e2}")
                result = []
    else:
        # SQLite/PostgreSQLモード
        try:
            sql = """
                SELECT DISTINCT residence_municipality
                FROM applicants
                WHERE residence_prefecture = ?
                ORDER BY residence_municipality
            """
            df = query_df(sql, (prefecture,))
            result = df["residence_municipality"].tolist()
        except Exception as e:
            print(f"[ERROR] DB query failed: {e}")
            result = []

    # 静的キャッシュに保存
    _static_cache["municipalities"][prefecture] = result
    if result:
        print(f"[DB] Cached {len(result)} municipalities for {prefecture}")
    else:
        print(f"[WARNING] No municipalities loaded for {prefecture}")

    return result


def query_municipality(prefecture: str, municipality: str = None) -> pd.DataFrame:
    """市区町村単位でデータを取得（永続キャッシュ対応、Turso専用）

    最適化: prefecture+municipality単位で永続キャッシュ。
    同じ地域へのアクセスは全ユーザーでキャッシュ共有（TTLなし）。
    データ更新時は refresh_all_cache() を呼び出してキャッシュをクリア。
    """
    global _static_cache

    cache_key = f"{prefecture}_{municipality or 'ALL'}"

    # 永続キャッシュから取得
    if cache_key in _static_cache.get("filtered_data", {}):
        cached = _static_cache["filtered_data"][cache_key]
        # キャッシュヒットログは抑制（ノイズ削減）
        return cached

    print(f"[DB] Fetching data for {prefecture}/{municipality or 'ALL'} (first time only)...")

    if municipality:
        sql = "SELECT * FROM job_seeker_data WHERE prefecture = ? AND municipality = ?"
        df = query_df(sql, (prefecture, municipality))
    else:
        sql = "SELECT * FROM job_seeker_data WHERE prefecture = ?"
        df = query_df(sql, (prefecture,))

    # 永続キャッシュに保存
    if "filtered_data" not in _static_cache:
        _static_cache["filtered_data"] = {}
    _static_cache["filtered_data"][cache_key] = df
    print(f"[DB] Cached {len(df)} rows for {cache_key} (persistent)")
    return df


def get_filtered_data(prefecture: str, municipality: str = None) -> pd.DataFrame:
    """サーバーサイドフィルタリング: 指定地域のデータのみ取得

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名（Noneの場合は都道府県全体）

    Returns:
        フィルタ済みDataFrame（数十〜数百行）

    最適化: 永続キャッシュを使用。
    同じ地域へのアクセスは全ユーザーでキャッシュ共有（TTLなし）。
    データ更新時は refresh_all_cache() を呼び出してキャッシュをクリア。
    """
    global _static_cache

    # 永続キャッシュキーを生成
    cache_key = f"{prefecture}_{municipality or 'ALL'}"

    # 永続キャッシュから取得
    if cache_key in _static_cache.get("filtered_data", {}):
        cached = _static_cache["filtered_data"][cache_key]
        # キャッシュヒットログは抑制（ノイズ削減）
        return cached

    if USE_CSV_MODE:
        print(f"[CSV] Filtering data for {prefecture}/{municipality or 'ALL'}...")
        df = _load_csv_data()
        if prefecture:
            df = df[df['prefecture'] == prefecture]
        if municipality:
            df = df[df['municipality'] == municipality]
        result = df.copy()
        # 永続キャッシュに保存
        if "filtered_data" not in _static_cache:
            _static_cache["filtered_data"] = {}
        _static_cache["filtered_data"][cache_key] = result
        print(f"[CSV] Cached {len(result)} rows for {cache_key} (persistent)")
        return result
    elif _HAS_TURSO:
        return query_municipality(prefecture, municipality)
    else:
        # SQLite/PostgreSQL用
        sql = "SELECT * FROM mapcomplete_raw WHERE 1=1"
        params = []

        if prefecture:
            sql += " AND prefecture = ?"
            params.append(prefecture)

        if municipality:
            sql += " AND municipality = ?"
            params.append(municipality)

        return query_df(sql, tuple(params)) if params else query_df(sql)


def get_row_count_by_location(prefecture: str, municipality: str = None) -> int:
    """指定地域のデータ行数を取得（軽量クエリ）"""
    if _HAS_TURSO:
        if municipality:
            sql = "SELECT COUNT(*) as cnt FROM job_seeker_data WHERE prefecture = ? AND municipality = ?"
            df = query_df(sql, (prefecture, municipality))
        else:
            sql = "SELECT COUNT(*) as cnt FROM job_seeker_data WHERE prefecture = ?"
            df = query_df(sql, (prefecture,))
    else:
        sql = "SELECT COUNT(*) as cnt FROM mapcomplete_raw WHERE 1=1"
        params = []

        if prefecture:
            sql += " AND prefecture = ?"
            params.append(prefecture)

        if municipality:
            sql += " AND municipality = ?"
            params.append(municipality)

        df = query_df(sql, tuple(params)) if params else query_df(sql)

    if not df.empty and 'cnt' in df.columns:
        return int(df['cnt'].iloc[0])
    return 0


def get_applicants(
    prefecture: Optional[str] = None, municipality: Optional[str] = None
) -> pd.DataFrame:
    """申請者データを取得（フィルタ可能）"""
    sql = "SELECT * FROM applicants WHERE 1=1"
    params = []

    if prefecture:
        sql += " AND residence_prefecture = ?"
        params.append(prefecture)

    if municipality:
        sql += " AND residence_municipality = ?"
        params.append(municipality)

    return query_df(sql, tuple(params)) if params else query_df(sql)


def get_persona_summary(
    age_group: Optional[str] = None,
    gender: Optional[str] = None,
    has_national_license: Optional[bool] = None,
) -> pd.DataFrame:
    """ペルソナサマリーを取得（フィルタ可能）"""
    sql = "SELECT * FROM persona_summary WHERE 1=1"
    params = []

    if age_group:
        sql += " AND age_group = ?"
        params.append(age_group)

    if gender:
        sql += " AND gender = ?"
        params.append(gender)

    if has_national_license is not None:
        sql += " AND has_national_license = ?"
        params.append(1 if has_national_license else 0)

    sql += " ORDER BY count DESC"

    return query_df(sql, tuple(params)) if params else query_df(sql)


def get_supply_density_map(location: Optional[str] = None) -> pd.DataFrame:
    """人材供給密度マップデータを取得"""
    sql = "SELECT * FROM supply_density_map WHERE 1=1"
    params = []

    if location:
        sql += " AND location = ?"
        params.append(location)

    sql += " ORDER BY supply_count DESC"

    return query_df(sql, tuple(params)) if params else query_df(sql)


def get_municipality_flow_edges(
    from_prefecture: Optional[str] = None,
    from_municipality: Optional[str] = None,
    to_prefecture: Optional[str] = None,
    to_municipality: Optional[str] = None,
) -> pd.DataFrame:
    """自治体間フローエッジを取得"""
    sql = "SELECT * FROM municipality_flow_edges WHERE 1=1"
    params = []

    if from_prefecture:
        sql += " AND from_prefecture = ?"
        params.append(from_prefecture)

    if from_municipality:
        sql += " AND from_municipality = ?"
        params.append(from_municipality)

    if to_prefecture:
        sql += " AND to_prefecture = ?"
        params.append(to_prefecture)

    if to_municipality:
        sql += " AND to_municipality = ?"
        params.append(to_municipality)

    sql += " ORDER BY flow_count DESC"

    return query_df(sql, tuple(params)) if params else query_df(sql)


def get_unique_values(table_name: str, column_name: str) -> list:
    """テーブルの指定カラムのユニーク値を取得"""
    sql = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL ORDER BY {column_name}"
    df = query_df(sql)
    return df[column_name].tolist()


def execute_custom_query(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """カスタムSQLクエリを実行"""
    return query_df(sql, params)


# =============================================================================
# 3層比較用統計取得関数（Turso最適化）
# =============================================================================

def _batch_stats_query(prefecture: str = None, municipality: str = None) -> dict:
    """統計取得用バッチクエリ（1回のHTTP通信で全row_type取得）

    従来: row_typeごとに3-4回クエリ
    改善後: 1回のクエリで全row_type取得 → Python内でフィルタ

    Args:
        prefecture: 都道府県名（Noneで全国）
        municipality: 市区町村名

    Returns:
        dict: {
            "SUMMARY": DataFrame,
            "RESIDENCE_FLOW": DataFrame,
            "AGE_GENDER": DataFrame
        }
    """
    print(f"[DEBUG] _batch_stats_query called: _HAS_TURSO={_HAS_TURSO}", flush=True)
    if not _HAS_TURSO:
        print("[DEBUG] _HAS_TURSO is False, returning empty dict", flush=True)
        return {}

    # キャッシュキー生成
    cache_key = f"batch_stats_{prefecture or 'ALL'}_{municipality or 'ALL'}"

    # 永続キャッシュから取得
    if cache_key in _static_cache:
        # キャッシュヒットログは抑制（ノイズ削減）
        return _static_cache[cache_key]

    print(f"[DB] Batch stats query for {prefecture or 'ALL'}/{municipality or 'ALL'}...")

    try:
        # 必要なrow_typeを1回のクエリで全取得
        conditions = ["row_type IN ('SUMMARY', 'RESIDENCE_FLOW', 'AGE_GENDER')"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality = ?")
            params.append(municipality)

        where_clause = " AND ".join(conditions)

        # 1回のHTTP通信で全データ取得
        df_all = query_df(
            f"SELECT * FROM job_seeker_data WHERE {where_clause}",
            tuple(params) if params else None
        )

        if df_all.empty:
            result = {"SUMMARY": pd.DataFrame(), "RESIDENCE_FLOW": pd.DataFrame(), "AGE_GENDER": pd.DataFrame()}
        else:
            # Python内でrow_typeごとにフィルタ（HTTP通信なし）
            result = {
                "SUMMARY": df_all[df_all["row_type"] == "SUMMARY"].copy(),
                "RESIDENCE_FLOW": df_all[df_all["row_type"] == "RESIDENCE_FLOW"].copy(),
                "AGE_GENDER": df_all[df_all["row_type"] == "AGE_GENDER"].copy()
            }

        # 永続キャッシュに保存
        _static_cache[cache_key] = result
        print(f"[DB] Batch stats cached: {cache_key} (SUMMARY:{len(result['SUMMARY'])}, FLOW:{len(result['RESIDENCE_FLOW'])}, AGE:{len(result['AGE_GENDER'])})")

        return result

    except Exception as e:
        print(f"[ERROR] Batch stats query failed: {e}")
        return {"SUMMARY": pd.DataFrame(), "RESIDENCE_FLOW": pd.DataFrame(), "AGE_GENDER": pd.DataFrame()}


def _batch_flow_query(municipality: str) -> dict:
    """人材フロー取得用バッチクエリ（1回のHTTP通信で流入元・流出先を同時取得）

    従来: get_flow_sources + get_flow_destinations で2回クエリ
    改善後: 1回のクエリで全DESIRED_AREA_PATTERN取得 → Python内で集計

    Args:
        municipality: 市区町村名（必須）

    Returns:
        dict: {
            "sources": list[dict],  # 流入元リスト [{"name": "本庄市", "count": 144}, ...]
            "destinations": list[dict]  # 流出先リスト [{"name": "前橋市", "count": 406}, ...]
        }
    """
    if not _HAS_TURSO or not municipality:
        return {"sources": [], "destinations": []}

    # キャッシュキー生成
    cache_key = f"batch_flow_{municipality}"

    # 永続キャッシュから取得
    if cache_key in _static_cache:
        # キャッシュヒットログは抑制（ノイズ削減）
        return _static_cache[cache_key]

    print(f"[DB] Batch flow query for {municipality}...")

    try:
        # 1回のHTTP通信で関連データを全取得
        # - 流入元: co_desired_municipality = municipality のレコード
        # - 流出先: municipality = municipality のレコード
        sql = """
            SELECT municipality, co_desired_municipality, count
            FROM job_seeker_data
            WHERE row_type = 'DESIRED_AREA_PATTERN'
            AND (municipality = ? OR co_desired_municipality = ?)
        """
        df_all = query_df(sql, (municipality, municipality))

        sources = []
        destinations = []

        if not df_all.empty:
            # 流入元集計: co_desired_municipality = municipality のレコードから
            # → 居住地(municipality)ごとに集計
            df_inbound = df_all[
                (df_all["co_desired_municipality"] == municipality) &
                (df_all["municipality"] != municipality) &
                (df_all["municipality"].notna())
            ]
            if not df_inbound.empty:
                inbound_agg = df_inbound.groupby("municipality")["count"].sum().sort_values(ascending=False).head(5)
                sources = [{"name": str(k), "count": int(v)} for k, v in inbound_agg.items()]

            # 流出先集計: municipality = municipality のレコードから
            # → 希望勤務地(co_desired_municipality)ごとに集計
            df_outbound = df_all[
                (df_all["municipality"] == municipality) &
                (df_all["co_desired_municipality"] != municipality) &
                (df_all["co_desired_municipality"].notna())
            ]
            if not df_outbound.empty:
                outbound_agg = df_outbound.groupby("co_desired_municipality")["count"].sum().sort_values(ascending=False).head(5)
                destinations = [{"name": str(k), "count": int(v)} for k, v in outbound_agg.items()]

        result = {"sources": sources, "destinations": destinations}

        # 永続キャッシュに保存
        _static_cache[cache_key] = result
        print(f"[DB] Batch flow cached: {cache_key} (sources:{len(sources)}, destinations:{len(destinations)})")

        return result

    except Exception as e:
        print(f"[ERROR] Batch flow query failed: {e}")
        return {"sources": [], "destinations": []}


def _batch_persona_query(prefecture: str = None, municipality: str = None) -> dict:
    """人材属性取得用バッチクエリ（1回のHTTP通信で複数row_type取得）

    従来: get_persona_market_share + get_qualification_retention_rates + get_qualification_options で3回クエリ
    改善後: 1回のクエリで AGE_GENDER_RESIDENCE + QUALIFICATION_DETAIL + QUALIFICATION_PERSONA を取得 → Python内で処理

    Args:
        prefecture: 都道府県名（Noneで全国）
        municipality: 市区町村名

    Returns:
        dict: {
            "AGE_GENDER_RESIDENCE": DataFrame,
            "QUALIFICATION_DETAIL": DataFrame,
            "QUALIFICATION_PERSONA": DataFrame
        }
    """
    if not _HAS_TURSO:
        return {}

    # キャッシュキー生成
    cache_key = f"batch_persona_{prefecture or 'ALL'}_{municipality or 'ALL'}"

    # 永続キャッシュから取得
    if cache_key in _static_cache:
        # キャッシュヒットログは抑制（ノイズ削減）
        return _static_cache[cache_key]

    print(f"[DB] Batch persona query for {prefecture or 'ALL'}/{municipality or 'ALL'}...")

    try:
        # 必要なrow_typeを1回のクエリで全取得（QUALIFICATION_PERSONA追加）
        conditions = ["row_type IN ('AGE_GENDER_RESIDENCE', 'QUALIFICATION_DETAIL', 'QUALIFICATION_PERSONA')"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality LIKE ?")
            params.append(f"{municipality}%")

        where_clause = " AND ".join(conditions)

        # 1回のHTTP通信で全データ取得
        df_all = query_df(
            f"SELECT * FROM job_seeker_data WHERE {where_clause}",
            tuple(params) if params else None
        )

        if df_all.empty:
            result = {"AGE_GENDER_RESIDENCE": pd.DataFrame(), "QUALIFICATION_DETAIL": pd.DataFrame(), "QUALIFICATION_PERSONA": pd.DataFrame()}
        else:
            # Python内でrow_typeごとにフィルタ（HTTP通信なし）
            result = {
                "AGE_GENDER_RESIDENCE": df_all[df_all["row_type"] == "AGE_GENDER_RESIDENCE"].copy(),
                "QUALIFICATION_DETAIL": df_all[df_all["row_type"] == "QUALIFICATION_DETAIL"].copy(),
                "QUALIFICATION_PERSONA": df_all[df_all["row_type"] == "QUALIFICATION_PERSONA"].copy()
            }

        # 永続キャッシュに保存
        _static_cache[cache_key] = result
        print(f"[DB] Batch persona cached: {cache_key} (AGE_GENDER_RESIDENCE:{len(result['AGE_GENDER_RESIDENCE'])}, QUALIFICATION_DETAIL:{len(result['QUALIFICATION_DETAIL'])}, QUALIFICATION_PERSONA:{len(result['QUALIFICATION_PERSONA'])})")

        return result

    except Exception as e:
        print(f"[ERROR] Batch persona query failed: {e}")
        return {"AGE_GENDER_RESIDENCE": pd.DataFrame(), "QUALIFICATION_DETAIL": pd.DataFrame(), "QUALIFICATION_PERSONA": pd.DataFrame()}


def get_national_stats() -> dict:
    """全国統計をバッチクエリで効率的に計算（Turso用）

    最適化: 従来3回のHTTP通信 → 1回のバッチクエリで取得

    Returns:
        dict: {
            "desired_areas": float,  # 平均希望勤務地数
            "distance_km": float,    # 平均移動距離
            "qualifications": float, # 平均資格保有数
            "male_count": int,       # 男性数
            "female_count": int,     # 女性数
            "age_distribution": dict # 年齢層別分布
        }
    """
    print("[DEBUG] get_national_stats() called", flush=True)
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    turso_init_result = _lazy_init_turso()
    print(f"[DEBUG] _lazy_init_turso() returned: {turso_init_result}", flush=True)
    if not turso_init_result:
        print("[DEBUG] Turso not initialized, returning empty dict")
        return {}

    result = {
        "desired_areas": 0.0,
        "distance_km": 0.0,
        "qualifications": 0.0,
        "male_count": 0,
        "female_count": 0,
        "age_distribution": {}
    }

    try:
        # バッチクエリで全row_typeを1回で取得
        batch_data = _batch_stats_query()

        # SUMMARYから基本統計を計算（Python内で集計）
        df_summary = batch_data.get("SUMMARY", pd.DataFrame())
        print(f"[DEBUG] get_national_stats: SUMMARY rows={len(df_summary)}", flush=True)

        # フォールバック: バッチクエリが空の場合、SUMMARYのみの小さいクエリを実行
        # （全国バッチクエリは502エラーになることがあるため）
        if df_summary.empty:
            print("[DEBUG] Batch query returned empty SUMMARY, trying fallback query...", flush=True)
            try:
                df_summary = query_df("SELECT * FROM job_seeker_data WHERE row_type = 'SUMMARY'")
                print(f"[DEBUG] Fallback SUMMARY query: {len(df_summary)} rows", flush=True)
            except Exception as fallback_err:
                print(f"[ERROR] Fallback SUMMARY query failed: {fallback_err}", flush=True)

        if not df_summary.empty:
            print(f"[DEBUG] SUMMARY columns: {list(df_summary.columns)[:15]}...", flush=True)
            # avg_desired_areasカラムの存在確認
            if 'avg_desired_areas' in df_summary.columns:
                valid_desired = df_summary['avg_desired_areas'].dropna()
                print(f"[DEBUG] avg_desired_areas: count={len(valid_desired)}, sample={valid_desired.head(3).tolist()}", flush=True)
                result["desired_areas"] = round(float(valid_desired.mean() or 0), 2)
            else:
                print(f"[DEBUG] avg_desired_areas column NOT FOUND in SUMMARY", flush=True)
            if 'avg_qualifications' in df_summary.columns:
                result["qualifications"] = round(float(df_summary['avg_qualifications'].mean() or 0), 2)
            if 'male_count' in df_summary.columns:
                result["male_count"] = int(df_summary['male_count'].sum() or 0)
            if 'female_count' in df_summary.columns:
                result["female_count"] = int(df_summary['female_count'].sum() or 0)
            print(f"[DEBUG] get_national_stats result: desired_areas={result['desired_areas']}, qualifications={result['qualifications']}", flush=True)

        # RESIDENCE_FLOWから平均移動距離を計算
        df_flow = batch_data.get("RESIDENCE_FLOW", pd.DataFrame())
        # フォールバック: RESIDENCE_FLOWも空の場合、個別クエリ
        if df_flow.empty:
            try:
                df_flow = query_df("SELECT * FROM job_seeker_data WHERE row_type = 'RESIDENCE_FLOW' LIMIT 5000")
                print(f"[DEBUG] Fallback RESIDENCE_FLOW query: {len(df_flow)} rows", flush=True)
            except Exception as flow_err:
                print(f"[DEBUG] Fallback RESIDENCE_FLOW query failed: {flow_err}", flush=True)

        if not df_flow.empty and 'avg_reference_distance_km' in df_flow.columns:
            valid_distances = df_flow['avg_reference_distance_km'].dropna()
            if len(valid_distances) > 0:
                result["distance_km"] = round(float(valid_distances.mean()), 2)

        # AGE_GENDERから年齢層別分布を計算
        df_age = batch_data.get("AGE_GENDER", pd.DataFrame())
        # フォールバック: AGE_GENDERも空の場合、個別クエリ
        if df_age.empty:
            try:
                df_age = query_df("SELECT * FROM job_seeker_data WHERE row_type = 'AGE_GENDER'")
                print(f"[DEBUG] Fallback AGE_GENDER query: {len(df_age)} rows", flush=True)
            except Exception as age_err:
                print(f"[DEBUG] Fallback AGE_GENDER query failed: {age_err}", flush=True)

        print(f"[DEBUG] df_age check: empty={df_age.empty}, rows={len(df_age)}", flush=True)
        if not df_age.empty:
            print(f"[DEBUG] AGE_GENDER columns: {list(df_age.columns)}", flush=True)
            print(f"[DEBUG] AGE_GENDER sample (first 3 rows): {df_age.head(3).to_dict('records')}", flush=True)
            if 'category1' in df_age.columns and 'count' in df_age.columns:
                # count列を数値に変換（文字列で格納されている場合の対策）
                df_age['count'] = pd.to_numeric(df_age['count'], errors='coerce').fillna(0)
                age_dist = df_age.groupby('category1')['count'].sum().to_dict()
                # 事前定義されたキーにマッピング（正規化）
                normalized_dist = {"20代": 0, "30代": 0, "40代": 0, "50代": 0, "60代": 0, "70歳以上": 0}
                for k, v in age_dist.items():
                    if k in normalized_dist:
                        normalized_dist[k] = int(v)
                result["age_distribution"] = normalized_dist
                print(f"[DEBUG] age_distribution: {result['age_distribution']}", flush=True)
            else:
                # category1がない場合、別のカラムを探す
                print(f"[DEBUG] AGE_GENDER missing category1 or count column, trying alternatives...", flush=True)
                # age_groupカラムがあるか確認
                if 'age_group' in df_age.columns:
                    print(f"[DEBUG] Found age_group column", flush=True)
                    if 'count' in df_age.columns:
                        age_dist = df_age.groupby('age_group')['count'].sum().to_dict()
                    elif 'applicant_count' in df_age.columns:
                        age_dist = df_age.groupby('age_group')['applicant_count'].sum().to_dict()
                    else:
                        age_dist = df_age['age_group'].value_counts().to_dict()
                    result["age_distribution"] = {str(k): int(v) for k, v in age_dist.items() if k}
                    print(f"[DEBUG] age_distribution from age_group: {result['age_distribution']}", flush=True)

        print(f"[DB] National stats FINAL: male={result['male_count']}, female={result['female_count']}, desired_areas={result['desired_areas']}, age_dist_keys={list(result.get('age_distribution', {}).keys())}", flush=True)

    except Exception as e:
        print(f"[ERROR] get_national_stats failed: {e}")

    return result


def get_prefecture_stats(prefecture: str) -> dict:
    """都道府県統計をバッチクエリで効率的に計算（Turso用）

    最適化: 従来3回のHTTP通信 → 1回のバッチクエリで取得

    Args:
        prefecture: 都道府県名

    Returns:
        dict: get_national_stats()と同じ形式
    """
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso() or not prefecture:
        return {}

    result = {
        "desired_areas": 0.0,
        "distance_km": 0.0,
        "qualifications": 0.0,
        "male_count": 0,
        "female_count": 0,
        "age_distribution": {}
    }

    try:
        # バッチクエリで全row_typeを1回で取得（都道府県フィルタ付き）
        batch_data = _batch_stats_query(prefecture=prefecture)

        # SUMMARYから基本統計を計算（Python内で集計）
        df_summary = batch_data.get("SUMMARY", pd.DataFrame())
        if not df_summary.empty:
            result["desired_areas"] = round(float(df_summary['avg_desired_areas'].mean() or 0), 2)
            result["qualifications"] = round(float(df_summary['avg_qualifications'].mean() or 0), 2)
            result["male_count"] = int(df_summary['male_count'].sum() or 0)
            result["female_count"] = int(df_summary['female_count'].sum() or 0)

        # RESIDENCE_FLOWから平均移動距離を計算
        df_flow = batch_data.get("RESIDENCE_FLOW", pd.DataFrame())
        if not df_flow.empty and 'avg_reference_distance_km' in df_flow.columns:
            valid_distances = df_flow['avg_reference_distance_km'].dropna()
            if len(valid_distances) > 0:
                result["distance_km"] = round(float(valid_distances.mean()), 2)

        # AGE_GENDERから年齢層別分布を計算
        df_age = batch_data.get("AGE_GENDER", pd.DataFrame())
        if not df_age.empty and 'category1' in df_age.columns and 'count' in df_age.columns:
            # count列を数値に変換（文字列で格納されている場合の対策）
            df_age['count'] = pd.to_numeric(df_age['count'], errors='coerce').fillna(0)
            age_dist = df_age.groupby('category1')['count'].sum().to_dict()
            # 事前定義されたキーにマッピング（正規化）
            normalized_dist = {"20代": 0, "30代": 0, "40代": 0, "50代": 0, "60代": 0, "70歳以上": 0}
            for k, v in age_dist.items():
                if k in normalized_dist:
                    normalized_dist[k] = int(v)
            result["age_distribution"] = normalized_dist

        print(f"[DB] Prefecture stats (batch) for {prefecture}: male={result['male_count']}, female={result['female_count']}, age_dist={result.get('age_distribution', {})}")

    except Exception as e:
        print(f"[ERROR] get_prefecture_stats failed: {e}")

    return result


def get_all_prefectures_stats() -> dict:
    """全都道府県の統計を一括取得（Turso用・キャッシュ効率化）

    Returns:
        dict: {prefecture_name: stats_dict, ...}
    """
    global _static_cache

    cache_key = "all_prefecture_stats"
    if cache_key in _static_cache:
        # キャッシュヒットログは抑制（ノイズ削減）
        return _static_cache[cache_key]

    if not _HAS_TURSO:
        return {}

    result = {}
    prefectures = get_prefectures()

    for pref in prefectures:
        result[pref] = get_prefecture_stats(pref)

    _static_cache[cache_key] = result
    print(f"[DB] Cached stats for {len(result)} prefectures")
    return result


def get_municipality_stats(prefecture: str, municipality: str) -> dict:
    """市区町村統計をバッチクエリで取得（Turso用3層比較）

    最適化: 従来3回のHTTP通信 → 1回のバッチクエリで取得
    遅延初期化対応（NiceGUI移行 2025-12-24）

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名

    Returns:
        dict: {
            "desired_areas": float,  # 平均希望勤務地数
            "distance_km": float,    # 平均移動距離
            "qualifications": float, # 平均資格保有数
            "male_count": int,
            "female_count": int,
            "female_ratio": float,
            "age_distribution": dict  # 年代別分布
        }
    """
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso():
        return {}

    try:
        # バッチクエリで全row_typeを1回で取得（市区町村フィルタ付き）
        batch_data = _batch_stats_query(prefecture=prefecture, municipality=municipality)

        desired_areas = 0.0
        qualifications = 0.0
        male_count = 0
        female_count = 0

        # SUMMARYから基本統計を計算（Python内で集計）
        df_summary = batch_data.get("SUMMARY", pd.DataFrame())
        if not df_summary.empty:
            row = df_summary.iloc[0]
            desired_areas = float(row.get('avg_desired_areas', 0) or 0)
            qualifications = float(row.get('avg_qualifications', 0) or 0)
            male_count = int(row.get('male_count', 0) or 0)
            female_count = int(row.get('female_count', 0) or 0)

        # RESIDENCE_FLOWから平均移動距離を計算
        distance_km = 0.0
        df_flow = batch_data.get("RESIDENCE_FLOW", pd.DataFrame())
        if not df_flow.empty and 'avg_reference_distance_km' in df_flow.columns:
            valid_distances = df_flow['avg_reference_distance_km'].dropna()
            if len(valid_distances) > 0:
                distance_km = float(valid_distances.mean())

        # AGE_GENDERから年代別分布を計算
        age_distribution = {"20代": 0, "30代": 0, "40代": 0, "50代": 0, "60代": 0, "70歳以上": 0}
        df_age = batch_data.get("AGE_GENDER", pd.DataFrame())
        print(f"[DEBUG] AGE_GENDER rows for {prefecture}/{municipality}: {len(df_age)}", flush=True)
        if not df_age.empty:
            print(f"[DEBUG] AGE_GENDER columns: {list(df_age.columns)}", flush=True)
            print(f"[DEBUG] AGE_GENDER sample: {df_age[['category1', 'count']].head().to_dict()}", flush=True)
        if not df_age.empty and 'category1' in df_age.columns and 'count' in df_age.columns:
            # count列を数値に変換
            df_age['count'] = pd.to_numeric(df_age['count'], errors='coerce').fillna(0)
            age_dist = df_age.groupby('category1')['count'].sum().to_dict()
            print(f"[DEBUG] age_dist after groupby: {age_dist}", flush=True)
            for age_group, cnt in age_dist.items():
                if age_group in age_distribution:
                    age_distribution[age_group] = int(cnt)
        print(f"[DEBUG] Final age_distribution: {age_distribution}", flush=True)

        # 女性比率計算
        total = male_count + female_count
        female_ratio = round(female_count / total * 100, 1) if total > 0 else 0.0

        return {
            "desired_areas": round(desired_areas, 2),
            "distance_km": round(distance_km, 2),
            "qualifications": round(qualifications, 2),
            "male_count": male_count,
            "female_count": female_count,
            "female_ratio": female_ratio,
            "age_distribution": age_distribution
        }

    except Exception as e:
        print(f"[DB] Municipality stats error for {prefecture}/{municipality}: {e}")
        return {}


def get_persona_market_share(prefecture: str = None, municipality: str = None) -> list:
    """ペルソナシェア（年齢×性別）をSQLで取得（Turso用）

    最適化: _batch_persona_query()を使用して他の人材属性クエリと1回のHTTP通信で取得

    Args:
        prefecture: 都道府県名（Noneで全国）
        municipality: 市区町村名

    Returns:
        list: [{"label": "30代×女性", "count": 156, "share_pct": "12.6%"}, ...]
    """
    if not _HAS_TURSO:
        return []

    try:
        # バッチクエリからAGE_GENDER_RESIDENCEデータを取得（キャッシュ済みなら即座に返却）
        batch_data = _batch_persona_query(prefecture, municipality)
        df = batch_data.get("AGE_GENDER_RESIDENCE", pd.DataFrame())

        if df.empty:
            return []

        # Python内で集計（HTTP通信なし）
        agg_df = df.groupby(["category1", "category2"])["count"].sum().reset_index()
        agg_df = agg_df.sort_values("count", ascending=False).head(12)

        total_all = agg_df["count"].sum()
        results = []
        for _, row in agg_df.iterrows():
            count = int(row["count"])
            share = (count / total_all * 100) if total_all > 0 else 0
            label = f"{row['category1']}×{row['category2']}"
            results.append({
                "label": label,
                "count": count,
                "share_pct": f"{share:.1f}%"
            })

        return results

    except Exception as e:
        print(f"[DB] Persona market share error: {e}")
        return []


def get_qualification_retention_rates(prefecture: str = None, municipality: str = None) -> list:
    """資格別定着率をSQLで取得（Turso用）

    最適化: _batch_persona_query()を使用して他の人材属性クエリと1回のHTTP通信で取得

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名

    Returns:
        list: [{"qualification": "介護福祉士", "retention_rate": "1.09", "interpretation": "地元志向"}, ...]
    """
    print(f"[DB] get_qualification_retention_rates called: pref={prefecture}, muni={municipality}")
    if not _HAS_TURSO:
        return []

    try:
        # バッチクエリからQUALIFICATION_DETAILデータを取得（キャッシュ済みなら即座に返却）
        batch_data = _batch_persona_query(prefecture, municipality)
        df = batch_data.get("QUALIFICATION_DETAIL", pd.DataFrame())

        if df.empty:
            return []

        # retention_rateがあるレコードのみ抽出
        df = df[df["retention_rate"].notna()].copy()
        if df.empty:
            return []

        # カウント用カラムを特定（applicant_count優先、なければcount、なければ1をデフォルト）
        if "applicant_count" in df.columns and df["applicant_count"].notna().any():
            count_col = "applicant_count"
        elif "count" in df.columns and df["count"].notna().any():
            count_col = "count"
        else:
            # カラムがない場合は1を仮定（行数をカウント）
            df["_temp_count"] = 1
            count_col = "_temp_count"

        print(f"[DB] get_qualification_retention_rates: using count_col={count_col}, sample values={df[count_col].head(3).tolist()}")

        # Python内で集計（HTTP通信なし）
        agg_df = df.groupby("category1").agg({
            "retention_rate": "mean",
            count_col: "sum"
        }).reset_index()
        agg_df.columns = ["qualification", "avg_retention", "total_count"]
        agg_df = agg_df.sort_values("total_count", ascending=False).head(10)

        results = []
        for _, row in agg_df.iterrows():
            rate = float(row["avg_retention"]) if pd.notna(row["avg_retention"]) else 1.0
            interpretation = "地元志向" if rate >= 1.0 else "流出傾向"
            results.append({
                "qualification": row["qualification"],
                "retention_rate": f"{rate:.2f}",
                "interpretation": interpretation,
                "count": int(row["total_count"]) if pd.notna(row["total_count"]) else 0
            })

        return results

    except Exception as e:
        print(f"[DB] Qualification retention rates error: {e}")
        return []


def get_rarity_analysis(prefecture: str = None, municipality: str = None,
                        ages: list = None, genders: list = None,
                        qualifications: list = None) -> list:
    """RARITY分析（年齢×性別×資格）をSQLで取得（Turso用）

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名
        ages: 選択された年齢層リスト
        genders: 選択された性別リスト
        qualifications: 選択された資格リスト

    Returns:
        list: [{"qualification": "介護福祉士", "age": "30代", "gender": "女性", "count": 156, "share_pct": "12.6%"}, ...]
    """
    if not _HAS_TURSO:
        return []

    try:
        # QUALIFICATION_PERSONA または RARITYを使用
        conditions = ["row_type IN ('QUALIFICATION_PERSONA', 'RARITY')"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality LIKE ?")
            params.append(f"{municipality}%")

        where_clause = " AND ".join(conditions)

        # QUALIFICATION_PERSONA/RARITYデータを取得
        df = query_df(
            f"""SELECT category1 as qualification, category2 as age, category3 as gender,
                       SUM(count) as total
               FROM job_seeker_data
               WHERE {where_clause}
               GROUP BY category1, category2, category3
               ORDER BY total DESC""",
            tuple(params)
        )

        if df.empty:
            return []

        # フィルタ適用
        if ages:
            df = df[df["age"].isin(ages)]
        if genders:
            df = df[df["gender"].isin(genders)]
        if qualifications:
            df = df[df["qualification"].isin(qualifications)]

        if df.empty:
            return []

        total_all = df["total"].sum()
        results = []
        for _, row in df.head(20).iterrows():
            count = int(row["total"])
            share = (count / total_all * 100) if total_all > 0 else 0
            results.append({
                "qualification": row["qualification"],
                "age": row["age"],
                "gender": row["gender"],
                "count": count,
                "share_pct": f"{share:.1f}%"
            })

        return results

    except Exception as e:
        print(f"[DB] Rarity analysis error: {e}")
        return []


def get_qualification_options(prefecture: str = None, municipality: str = None) -> list:
    """選択可能な資格リストを取得（Turso用）- 取得者数順

    最適化: _batch_persona_query()を使用して他の人材属性クエリと1回のHTTP通信で取得

    Returns:
        list: [(資格名, 取得者数), ...] 取得者数の多い順
    """
    if not _HAS_TURSO:
        return []

    try:
        # バッチクエリを使用（キャッシュ活用）
        batch_data = _batch_persona_query(prefecture, municipality)
        df = batch_data.get("QUALIFICATION_DETAIL", pd.DataFrame())

        if df.empty:
            return []

        # Python内で資格ごとの取得者数を集計（HTTP通信なし）
        if "category1" in df.columns and "count" in df.columns:
            # 資格ごとにcountを合計
            grouped = df.groupby("category1")["count"].sum().reset_index()
            grouped.columns = ["qualification", "total_count"]
            # 取得者数の多い順にソート
            grouped = grouped.sort_values("total_count", ascending=False)
            # (資格名, 取得者数) のタプルリストを返す
            return [(row["qualification"], int(row["total_count"])) for _, row in grouped.iterrows()]
        elif "category1" in df.columns:
            # countがない場合は出現回数でソート
            counts = df["category1"].value_counts()
            return [(qual, int(cnt)) for qual, cnt in counts.items()]

        return []

    except Exception as e:
        print(f"[DB] Get qualification options error: {e}")
        return []


def get_age_gender_stats(prefecture: str = None, municipality: str = None) -> list:
    """年齢×性別ごとの平均希望勤務地数・平均資格保有数を取得（Turso用）

    最適化: _batch_persona_query()を使用して他の人材属性クエリと1回のHTTP通信で取得

    Returns:
        list: [
            {"label": "20代男性", "desired_areas": "2.8", "qualifications": "0.8"},
            {"label": "20代女性", "desired_areas": "3.1", "qualifications": "1.2"},
            ...
        ]
    """
    if not _HAS_TURSO:
        return []

    try:
        # バッチクエリを使用（キャッシュ活用）
        batch_data = _batch_persona_query(prefecture, municipality)
        df = batch_data.get("AGE_GENDER_RESIDENCE", pd.DataFrame())

        if df.empty:
            return []

        # 年齢×性別で集計
        age_order = ['20代', '30代', '40代', '50代', '60代', '70歳以上']
        gender_order = ['男性', '女性']

        results = []
        for age in age_order:
            for gender in gender_order:
                subset = df[(df['category1'] == age) & (df['category2'] == gender)]
                if len(subset) > 0:
                    desired = subset['avg_desired_areas'].mean() if 'avg_desired_areas' in subset.columns else 0
                    quals = subset['avg_qualifications'].mean() if 'avg_qualifications' in subset.columns else 0

                    results.append({
                        "label": f"{age}{gender}",
                        "desired_areas": f"{desired:.1f}" if pd.notna(desired) else "-",
                        "qualifications": f"{quals:.1f}" if pd.notna(quals) else "-"
                    })

        return results

    except Exception as e:
        print(f"[DB] Get age gender stats error: {e}")
        return []


def get_persona_employment_breakdown(prefecture: str = None, municipality: str = None) -> list:
    """就業状態別ペルソナ分析データを取得（Turso用）

    Reflex版と同じロジック: PERSONA_MUNI の category1 を分解して
    年齢・性別×就業状態の積み上げデータを生成

    Returns:
        list: [
            {"age_gender": "50代・女性", "就業中": 256, "離職中": 80, "在学中": 10},
            {"age_gender": "30代・女性", "就業中": 200, "離職中": 50, "在学中": 5},
            ...
        ]
    """
    if not _HAS_TURSO:
        return []

    try:
        conditions = ["row_type = 'PERSONA_MUNI'"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality LIKE ?")
            params.append(f"{municipality}%")

        where_clause = " AND ".join(conditions)

        df = query_df(
            f"""SELECT category1, SUM(count) as total
               FROM job_seeker_data
               WHERE {where_clause}
               GROUP BY category1
               ORDER BY total DESC""",
            tuple(params)
        )

        if df.empty:
            return []

        # ペルソナ名を分解して就業状態別に集計
        breakdown_data = {}
        for _, row in df.iterrows():
            persona_name = str(row.get('category1', ''))
            count = int(row.get('total', 0))

            # ペルソナ名を「・」で分割（例: "50代・女性・就業中"）
            parts = persona_name.split('・')
            if len(parts) >= 3:
                age_gender = f"{parts[0]}・{parts[1]}"  # "50代・女性"
                employment = parts[2]  # "就業中"

                if age_gender not in breakdown_data:
                    breakdown_data[age_gender] = {"age_gender": age_gender, "就業中": 0, "離職中": 0, "在学中": 0}

                if employment in ["就業中", "離職中", "在学中"]:
                    breakdown_data[age_gender][employment] += count

        # リストに変換してソート（合計人数降順）
        result = list(breakdown_data.values())
        result.sort(key=lambda x: x["就業中"] + x["離職中"] + x["在学中"], reverse=True)

        # 上位10件のみ返す
        return result[:10]

    except Exception as e:
        print(f"[DB] Get persona employment breakdown error: {e}")
        return []


def get_qualification_by_gender(prefecture: str = None, municipality: str = None) -> list:
    """資格別男女保有者数を取得（Turso用）

    QUALIFICATION_PERSONAを使用:
    - category1 = 資格名
    - category2 = 年齢帯（50代など）
    - category3 = 性別（男性/女性）
    - count = 件数

    Returns:
        list: [
            {"qualification": "介護福祉士", "male": 1200, "female": 3500, "total": 4700},
            ...
        ]
    """
    if not _HAS_TURSO:
        return []

    try:
        # QUALIFICATION_PERSONAからデータ取得（性別情報はcategory3にある）
        batch_data = _batch_persona_query(prefecture, municipality)
        df = batch_data.get("QUALIFICATION_PERSONA", pd.DataFrame())

        if df.empty:
            print("[DB] get_qualification_by_gender: QUALIFICATION_PERSONA is empty")
            return []

        # 資格×性別で集計（category1=資格名, category3=性別）
        results = []
        if "category1" in df.columns and "category3" in df.columns and "count" in df.columns:
            for qual in df["category1"].dropna().unique():
                qual_df = df[df["category1"] == qual]
                male_count = qual_df[qual_df["category3"] == "男性"]["count"].sum()
                female_count = qual_df[qual_df["category3"] == "女性"]["count"].sum()

                if male_count > 0 or female_count > 0:
                    results.append({
                        "qualification": qual,
                        "male": int(male_count),
                        "female": int(female_count),
                        "total": int(male_count + female_count)
                    })
        else:
            print(f"[DB] get_qualification_by_gender: Missing columns. Available: {list(df.columns)}")

        # 合計降順でソート
        results.sort(key=lambda x: x["total"], reverse=True)
        return results[:10]

    except Exception as e:
        print(f"[DB] Get qualification by gender error: {e}")
        return []


def get_distance_stats(prefecture: str = None, municipality: str = None) -> dict:
    """距離統計をSQLで取得（Turso用）

    Returns:
        dict: {"mean": "42.5", "min": "0.0", "max": "150.3", "q25": "10.2", "median": "35.0", "q75": "65.8", "unit": "km"}
    """
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso():
        print("[DB] get_distance_stats: Turso not initialized")
        return {"mean": "-", "min": "-", "max": "-", "q25": "-", "median": "-", "q75": "-", "unit": "km"}

    try:
        conditions = ["row_type = 'RESIDENCE_FLOW'", "avg_reference_distance_km IS NOT NULL"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality LIKE ?")
            params.append(f"{municipality}%")

        where_clause = " AND ".join(conditions)

        # 全国レベルでも同じクエリ（集計はPython側で行う）
        # SQLで基本統計を計算（メモリ効率化）
        df = query_df(
            f"""SELECT
                   AVG(avg_reference_distance_km) as mean,
                   MIN(avg_reference_distance_km) as min,
                   MAX(avg_reference_distance_km) as max
               FROM job_seeker_data
               WHERE {where_clause}""",
            tuple(params)
        )

        if df.empty or df["mean"].isna().all():
            print(f"[DB] get_distance_stats: No aggregated data found")
            return {"mean": "-", "min": "-", "max": "-", "q25": "-", "median": "-", "q75": "-", "unit": "km"}

        row = df.iloc[0]
        mean_val = row["mean"] if pd.notna(row["mean"]) else 0
        min_val = row["min"] if pd.notna(row["min"]) else 0
        max_val = row["max"] if pd.notna(row["max"]) else 0

        # パーセンタイルはSQLでは計算困難なのでサンプリングで推定
        sample_df = query_df(
            f"""SELECT avg_reference_distance_km as distance
               FROM job_seeker_data
               WHERE {where_clause}
               ORDER BY RANDOM()
               LIMIT 10000""",  # サンプリング
            tuple(params)
        )

        # サンプルからパーセンタイルを計算
        if sample_df.empty or sample_df["distance"].isna().all():
            # サンプルが空なら集計結果のみ返す
            print(f"[DB] get_distance_stats: Using aggregated only (pref={prefecture}, muni={municipality})")
            return {
                "mean": f"{mean_val:.1f}",
                "min": f"{min_val:.1f}",
                "max": f"{max_val:.1f}",
                "q25": "-",
                "median": "-",
                "q75": "-",
                "unit": "km"
            }

        distances = sample_df["distance"].dropna()
        print(f"[DB] get_distance_stats: Sampled {len(distances)} distance records, mean={mean_val:.1f}")
        return {
            "mean": f"{mean_val:.1f}",  # SQLで計算した正確な平均
            "min": f"{min_val:.1f}",    # SQLで計算した正確な最小値
            "max": f"{max_val:.1f}",    # SQLで計算した正確な最大値
            "q25": f"{distances.quantile(0.25):.1f}" if len(distances) > 0 else "-",
            "median": f"{distances.median():.1f}" if len(distances) > 0 else "-",
            "q75": f"{distances.quantile(0.75):.1f}" if len(distances) > 0 else "-",
            "unit": "km"
        }

    except Exception as e:
        print(f"[DB] Distance stats error: {e}")
        import traceback
        traceback.print_exc()
        return {"mean": "-", "min": "-", "max": "-", "q25": "-", "median": "-", "q75": "-", "unit": "km"}


def get_mobility_type_distribution(prefecture: str = None, municipality: str = None,
                                     mode: str = "residence") -> list:
    """移動タイプ分布をSQLで取得（Turso用）

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名
        mode: "residence"（居住地ベース）または"destination"（希望勤務地ベース）

    Returns:
        list: [{"type": "地元希望", "count": 280, "pct": "25.5%"}, ...]
    """
    if not _HAS_TURSO:
        return []

    try:
        conditions = ["row_type = 'RESIDENCE_FLOW'", "mobility_type IS NOT NULL"]
        params = []

        # モードによってフィルタ対象列を変更
        if mode == "residence":
            if prefecture:
                conditions.append("prefecture = ?")
                params.append(prefecture)
            if municipality:
                conditions.append("municipality LIKE ?")
                params.append(f"{municipality}%")
        else:
            # destination mode
            if prefecture:
                conditions.append("desired_prefecture = ?")
                params.append(prefecture)
            if municipality:
                conditions.append("desired_municipality LIKE ?")
                params.append(f"{municipality}%")

        where_clause = " AND ".join(conditions)

        df = query_df(
            f"""SELECT mobility_type, SUM(count) as total
               FROM job_seeker_data
               WHERE {where_clause}
               GROUP BY mobility_type""",
            tuple(params)
        )

        if df.empty:
            return []

        total_all = df["total"].sum()
        type_order = ['地元希望', '近隣移動', '中距離移動', '遠距離移動']

        results = []
        for t in type_order:
            row = df[df["mobility_type"] == t]
            count = int(row["total"].iloc[0]) if len(row) > 0 else 0
            pct = (count / total_all * 100) if total_all > 0 else 0
            results.append({
                "type": t,
                "count": count,
                "pct": f"{pct:.1f}%"
            })

        return results

    except Exception as e:
        print(f"[DB] Mobility type distribution error: {e}")
        return []


def get_competition_overview(prefecture: str = None, municipality: str = None) -> dict:
    """競争度概要をSQLで取得（Turso用）

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名

    Returns:
        dict: {
            "total_applicants": int,
            "female_ratio": str,
            "male_ratio": str,
            ...
        }
    """
    if not _HAS_TURSO:
        return {}

    try:
        conditions = ["row_type = 'COMPETITION'"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality LIKE ?")
            params.append(f"{municipality}%")

        where_clause = " AND ".join(conditions)

        df = query_df(
            f"""SELECT total_applicants, female_ratio, male_ratio, category1,
                       top_age_ratio, category2, top_employment_ratio, avg_qualification_count
               FROM job_seeker_data
               WHERE {where_clause}
               LIMIT 1""",
            tuple(params)
        )

        # 市区町村指定時にデータがない場合は、都道府県レベルにフォールバック
        if df.empty and municipality and prefecture:
            conditions_pref = ["row_type = 'COMPETITION'", "prefecture = ?"]
            df = query_df(
                f"""SELECT total_applicants, female_ratio, male_ratio, category1,
                           top_age_ratio, category2, top_employment_ratio, avg_qualification_count
                   FROM job_seeker_data
                   WHERE {' AND '.join(conditions_pref)}
                   LIMIT 1""",
                (prefecture,)
            )

        if df.empty:
            return {}

        row = df.iloc[0]

        def safe_float(val, default=0.0):
            try:
                return float(val) if pd.notna(val) else default
            except:
                return default

        def safe_int(val, default=0):
            try:
                return int(val) if pd.notna(val) else default
            except:
                return default

        return {
            "total_applicants": safe_int(row.get('total_applicants')),
            "female_ratio": f"{safe_float(row.get('female_ratio')) * 100:.1f}%",
            "male_ratio": f"{safe_float(row.get('male_ratio')) * 100:.1f}%",
            "top_age": str(row.get('category1', '-')) if pd.notna(row.get('category1')) else '-',
            "top_age_ratio": f"{safe_float(row.get('top_age_ratio')) * 100:.1f}%",
            "top_employment": str(row.get('category2', '-')) if pd.notna(row.get('category2')) else '-',
            "top_employment_ratio": f"{safe_float(row.get('top_employment_ratio')) * 100:.1f}%",
            "avg_qualification_count": f"{safe_float(row.get('avg_qualification_count')):.1f}"
        }

    except Exception as e:
        print(f"[DB] Competition overview error: {e}")
        return {}


def get_talent_flow(prefecture: str = None, municipality: str = None) -> dict:
    """人材フロー（流入/流出/純流）をSQLで取得（Turso用）

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名

    Returns:
        dict: {
            "inflow": int,      # 流入数
            "outflow": int,     # 流出数
            "net_flow": int,    # 純流入（inflow - outflow）
            "applicant_count": int  # 求職者数
        }
    """
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso():
        return {}

    try:
        conditions = ["row_type = 'FLOW'"]
        params = []

        if prefecture:
            conditions.append("prefecture = ?")
            params.append(prefecture)
        if municipality:
            conditions.append("municipality = ?")
            params.append(municipality)

        where_clause = " AND ".join(conditions)

        # 全国レベル（フィルタなし）の場合は集計、それ以外はLIMIT 1
        if not prefecture and not municipality:
            # 全国合計を取得
            df = query_df(
                f"""SELECT
                       SUM(CAST(COALESCE(applicant_count, 0) AS REAL)) as applicant_count,
                       SUM(CAST(COALESCE(inflow, 0) AS REAL)) as inflow,
                       SUM(CAST(COALESCE(outflow, 0) AS REAL)) as outflow,
                       SUM(CAST(COALESCE(net_flow, 0) AS REAL)) as net_flow
                   FROM job_seeker_data
                   WHERE {where_clause}""",
                tuple(params)
            )
        else:
            df = query_df(
                f"""SELECT municipality, applicant_count, inflow, outflow, net_flow
                   FROM job_seeker_data
                   WHERE {where_clause}
                   LIMIT 1""",
                tuple(params)
            )

        if df.empty:
            return {}

        row = df.iloc[0]

        def safe_int(val, default=0):
            try:
                return int(val) if pd.notna(val) else default
            except:
                return default

        return {
            "inflow": safe_int(row.get('inflow')),
            "outflow": safe_int(row.get('outflow')),
            "net_flow": safe_int(row.get('net_flow')),
            "applicant_count": safe_int(row.get('applicant_count')),
            "municipality": str(row.get('municipality', '')) if pd.notna(row.get('municipality')) else ''
        }

    except Exception as e:
        print(f"[DB] Talent flow error for {prefecture}/{municipality}: {e}")
        return {}


def get_flow_sources(prefecture: str = None, municipality: str = None, limit: int = 5) -> list:
    """流入元（どこから来るか）を取得（Turso用）

    最適化: _batch_flow_query()を使用して1回のHTTP通信で取得

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名（必須）
        limit: 取得件数上限

    Returns:
        list: [{"name": "本庄市", "count": 144}, ...]
    """
    if not _HAS_TURSO or not municipality:
        return []

    # バッチクエリから取得（キャッシュ済みなら即座に返却）
    batch_data = _batch_flow_query(municipality)
    return batch_data.get("sources", [])[:limit]


def get_flow_destinations(prefecture: str = None, municipality: str = None, limit: int = 5) -> list:
    """流出先（どこへ流れるか）を取得（Turso用）

    最適化: _batch_flow_query()を使用して1回のHTTP通信で取得

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名（必須）
        limit: 取得件数上限

    Returns:
        list: [{"name": "前橋市", "count": 406}, ...]
    """
    if not _HAS_TURSO or not municipality:
        return []

    # バッチクエリから取得（キャッシュ済みなら即座に返却）
    batch_data = _batch_flow_query(municipality)
    return batch_data.get("destinations", [])[:limit]


# 旧実装（参考用・削除可能）
def _get_flow_sources_legacy(prefecture: str = None, municipality: str = None, limit: int = 5) -> list:
    """【旧実装】流入元取得（個別HTTP通信版）"""
    if not _HAS_TURSO or not municipality:
        return []

    try:
        sql = """
            SELECT municipality as source_muni, SUM(count) as total_count
            FROM job_seeker_data
            WHERE row_type = 'DESIRED_AREA_PATTERN'
            AND co_desired_municipality = ?
            AND municipality IS NOT NULL
            AND municipality != ?
            GROUP BY municipality
            ORDER BY total_count DESC
            LIMIT ?
        """
        params = [municipality, municipality, limit]

        df = query_df(sql, tuple(params))

        if df.empty:
            return []

        result = []
        for _, row in df.iterrows():
            source = row.get('source_muni', '')
            count = row.get('total_count', 0)
            if source and pd.notna(source):
                result.append({
                    "name": str(source),
                    "count": int(count) if pd.notna(count) else 0
                })

        return result

    except Exception as e:
        print(f"[DB] Flow sources error for {municipality}: {e}")
        return []


def get_residence_flow_data(prefecture: str = None, municipality: str = None) -> pd.DataFrame:
    """RESIDENCE_FLOWデータを取得（都道府県/市区町村間フロー表示用）

    Args:
        prefecture: 都道府県名（Noneで全国）
        municipality: 市区町村名

    Returns:
        DataFrame: RESIDENCE_FLOWデータ（residence_pref, desired_prefecture, count等）
    """
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso():
        return pd.DataFrame()

    try:
        # バッチクエリからRESIDENCE_FLOWデータを取得
        batch_data = _batch_stats_query(prefecture, municipality)
        df = batch_data.get("RESIDENCE_FLOW", pd.DataFrame())

        if df.empty:
            print(f"[DB] get_residence_flow_data: No RESIDENCE_FLOW data for {prefecture}/{municipality}")
            return pd.DataFrame()

        print(f"[DB] get_residence_flow_data: {len(df)} rows for {prefecture}/{municipality}")
        print(f"[DB] RESIDENCE_FLOW columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"[DB] get_residence_flow_data error: {e}")
        return pd.DataFrame()


def get_pref_flow_top10(prefecture: str = None) -> list:
    """都道府県間フローTop10を取得

    Args:
        prefecture: 都道府県名（フィルタ用、Noneで全国）

    Returns:
        list: [{"origin": "大阪府", "destination": "東京都", "count": 1234}, ...]
    """
    print(f"[DB] get_pref_flow_top10 called: prefecture={prefecture}")
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso():
        print("[DB] get_pref_flow_top10: _lazy_init_turso returned False")
        return []

    try:
        df = get_residence_flow_data(prefecture, None)
        print(f"[DB] get_pref_flow_top10: got {len(df)} rows from get_residence_flow_data")

        if df.empty:
            print("[DB] get_pref_flow_top10: df is empty, returning []")
            return []

        # RESIDENCE_FLOWのカラム構造:
        # prefecture = 現住所都道府県, desired_prefecture = 希望勤務地都道府県
        # count = 件数
        origin_col = "prefecture"
        dest_col = "desired_prefecture"
        count_col = "count"

        if origin_col not in df.columns or dest_col not in df.columns:
            print(f"[DB] get_pref_flow_top10: Required columns not found. Available: {list(df.columns)}")
            return []

        # 異なる都道府県間のフローのみ（同じ都道府県は除外）
        df_flow = df[(df[origin_col].notna()) & (df[dest_col].notna())].copy()
        df_flow = df_flow[df_flow[origin_col] != df_flow[dest_col]]

        if df_flow.empty:
            return []

        # 集計
        if count_col:
            agg = df_flow.groupby([origin_col, dest_col])[count_col].sum().reset_index()
            agg = agg.sort_values(count_col, ascending=False).head(10)

            result = []
            for _, row in agg.iterrows():
                result.append({
                    "origin": str(row[origin_col]),
                    "destination": str(row[dest_col]),
                    "count": int(row[count_col])
                })
            return result
        else:
            # count列がない場合は件数カウント
            agg = df_flow.groupby([origin_col, dest_col]).size().reset_index(name="count")
            agg = agg.sort_values("count", ascending=False).head(10)

            result = []
            for _, row in agg.iterrows():
                result.append({
                    "origin": str(row[origin_col]),
                    "destination": str(row[dest_col]),
                    "count": int(row["count"])
                })
            return result

    except Exception as e:
        print(f"[DB] get_pref_flow_top10 error: {e}")
        return []


def get_muni_flow_top10(prefecture: str = None, municipality: str = None) -> list:
    """市区町村間フローTop10を取得

    Args:
        prefecture: 都道府県名
        municipality: 市区町村名（フィルタ用）

    Returns:
        list: [{"origin": "渋谷区", "destination": "新宿区", "count": 567}, ...]
    """
    print(f"[DB] get_muni_flow_top10 called: prefecture={prefecture}, municipality={municipality}")
    # 遅延初期化を呼び出し（NiceGUI移行対応 2025-12-24）
    if not _lazy_init_turso():
        print("[DB] get_muni_flow_top10: _lazy_init_turso returned False")
        return []

    try:
        df = get_residence_flow_data(prefecture, municipality)
        print(f"[DB] get_muni_flow_top10: got {len(df)} rows from get_residence_flow_data")

        if df.empty:
            print("[DB] get_muni_flow_top10: df is empty, returning []")
            return []

        # RESIDENCE_FLOWのカラム構造:
        # municipality = 現住所市区町村, desired_municipality = 希望勤務地市区町村
        # count = 件数
        origin_col = "municipality"
        dest_col = "desired_municipality"
        count_col = "count"

        if origin_col not in df.columns or dest_col not in df.columns:
            print(f"[DB] get_muni_flow_top10: Required columns not found. Available: {list(df.columns)}")
            return []

        # 異なる市区町村間のフローのみ（同じ市区町村は除外）
        df_flow = df[(df[origin_col].notna()) & (df[dest_col].notna())].copy()
        print(f"[DB] get_muni_flow_top10: after notna filter: {len(df_flow)} rows")
        if len(df_flow) > 0:
            print(f"[DB] get_muni_flow_top10: sample origin={df_flow[origin_col].iloc[0]}, dest={df_flow[dest_col].iloc[0]}")
        df_flow = df_flow[df_flow[origin_col] != df_flow[dest_col]]
        print(f"[DB] get_muni_flow_top10: after diff filter: {len(df_flow)} rows")

        if df_flow.empty:
            print("[DB] get_muni_flow_top10: df_flow is empty after filtering")
            return []

        # 集計
        if count_col:
            agg = df_flow.groupby([origin_col, dest_col])[count_col].sum().reset_index()
            agg = agg.sort_values(count_col, ascending=False).head(10)

            result = []
            for _, row in agg.iterrows():
                result.append({
                    "origin": str(row[origin_col]),
                    "destination": str(row[dest_col]),
                    "count": int(row[count_col])
                })
            return result
        else:
            agg = df_flow.groupby([origin_col, dest_col]).size().reset_index(name="count")
            agg = agg.sort_values("count", ascending=False).head(10)

            result = []
            for _, row in agg.iterrows():
                result.append({
                    "origin": str(row[origin_col]),
                    "destination": str(row[dest_col]),
                    "count": int(row["count"])
                })
            return result

    except Exception as e:
        print(f"[DB] get_muni_flow_top10 error: {e}")
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("データベースヘルパーテスト")
    print("=" * 60)
    print(f"Database Type: {get_db_type()}")

    if _HAS_TURSO:
        print(f"Turso URL: {TURSO_DATABASE_URL}")

        # 都道府県一覧取得
        print("\n都道府県一覧:")
        prefectures = get_prefectures()
        print(f"  {len(prefectures)}都道府県")
        if prefectures:
            print(f"  例: {prefectures[:3]}")

            # 最初の都道府県の市区町村
            first_pref = prefectures[0]
            municipalities = get_municipalities(first_pref)
            print(f"\n{first_pref}の市区町村: {len(municipalities)}")

            # データ取得テスト
            df = query_municipality(first_pref)
            print(f"{first_pref}のデータ: {len(df)}行")
    else:
        print("Turso未設定。SQLite/PostgreSQLモード。")
