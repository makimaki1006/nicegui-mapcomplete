# -*- coding: utf-8 -*-
"""
NiceGUI版 MapComplete Dashboard PoC
Reflex版からの移行テスト

起動方法:
    cd nicegui_app
    pip install -r requirements.txt
    python main.py
"""

import os
from pathlib import Path
from nicegui import ui, app
import pandas as pd
import httpx

# =====================================
# 環境変数読み込み
# =====================================
try:
    from dotenv import load_dotenv
    # ローカル開発用: nicegui_app/.env または reflex_app/.env.production
    local_env = Path(__file__).parent / ".env"
    reflex_env = Path(__file__).parent.parent / "reflex_app" / ".env.production"

    if local_env.exists():
        load_dotenv(local_env)
        print(f"[STARTUP] Loaded: {local_env}")
    elif reflex_env.exists():
        load_dotenv(reflex_env)
        print(f"[STARTUP] Loaded: {reflex_env}")
    else:
        print("[STARTUP] No .env file found, using system environment variables")
except Exception as e:
    print(f"[STARTUP] dotenv error: {e}")

# Turso接続設定
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
print(f"[TURSO] URL set: {bool(TURSO_DATABASE_URL)}, TOKEN set: {bool(TURSO_AUTH_TOKEN)}")

# 認証設定
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "cyzen_2025")
ALLOWED_DOMAINS = [d.strip() for d in os.getenv("ALLOWED_DOMAINS", "f-a-c.co.jp,cyxen.co.jp").split(",")]
print(f"[AUTH] Allowed domains: {ALLOWED_DOMAINS}")

# =====================================
# データ読み込み（Turso優先、CSVフォールバック）
# =====================================
CSV_PATH = Path(__file__).parent.parent / "reflex_app" / "MapComplete_Complete_All_FIXED.csv"
CSV_PATH_GZ = Path(__file__).parent.parent / "reflex_app" / "MapComplete_Complete_All_FIXED.csv.gz"
CSV_PATH_ALT = Path(__file__).parent.parent / "python_scripts" / "data" / "output_v2" / "mapcomplete_complete_sheets" / "MapComplete_Complete_All_FIXED.csv"

_dataframe = None
_data_source = "未読み込み"  # データソース表示用

def query_turso(sql: str) -> pd.DataFrame:
    """Turso HTTP APIでクエリ実行"""
    http_url = TURSO_DATABASE_URL
    if http_url.startswith('libsql://'):
        http_url = http_url.replace('libsql://', 'https://')

    headers = {
        'Authorization': f'Bearer {TURSO_AUTH_TOKEN}',
        'Content-Type': 'application/json'
    }

    payload = {
        'requests': [
            {'type': 'execute', 'stmt': {'sql': sql}}
        ]
    }

    try:
        response = httpx.post(
            f'{http_url}/v2/pipeline',
            headers=headers,
            json=payload,
            timeout=30.0
        )

        if response.status_code != 200:
            raise Exception(f"Turso HTTP {response.status_code}: {response.text}")

        data = response.json()

        if not data.get('results'):
            return pd.DataFrame()

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

        return pd.DataFrame(rows, columns=columns)

    except Exception as e:
        print(f"[TURSO] Query error: {e}")
        raise

def load_data() -> pd.DataFrame:
    """Turso優先、CSVフォールバックでデータを読み込み（キャッシュ付き）"""
    global _dataframe, _data_source
    if _dataframe is not None:
        return _dataframe

    # Tursoからデータ取得を試みる（SUMMARYデータのみ - 高速化のため）
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            print("[DATA] Loading from Turso (SUMMARY only)...")
            print(f"[DATA] Turso URL: {TURSO_DATABASE_URL[:50]}...")
            # SUMMARYのみ取得（全データは大きすぎてHTTP APIでタイムアウト）
            _dataframe = query_turso("SELECT * FROM job_seeker_data WHERE row_type = 'SUMMARY'")
            _data_source = "Turso DB"
            print(f"[DATA] SUCCESS: Loaded {len(_dataframe):,} rows from Turso")
            return _dataframe
        except Exception as e:
            print(f"[DATA] Turso failed: {type(e).__name__}: {e}")
            print(f"[DATA] Falling back to CSV...")

    # CSVファイルを探す（フォールバック）
    for path in [CSV_PATH_GZ, CSV_PATH, CSV_PATH_ALT]:
        if path.exists():
            print(f"[DATA] Loading from CSV: {path}")
            if path.suffix == '.gz':
                _dataframe = pd.read_csv(path, encoding='utf-8-sig', compression='gzip', low_memory=False)
            else:
                _dataframe = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
            _data_source = f"CSV ({path.name})"
            print(f"[DATA] FALLBACK: Loaded {len(_dataframe):,} rows from CSV")
            return _dataframe

    # 見つからない場合はエラー
    _data_source = "エラー"
    raise FileNotFoundError("No data source available (Turso and CSV both failed)")

# =====================================
# 認証チェック
# =====================================
def is_authenticated() -> bool:
    """認証済みかチェック"""
    return app.storage.user.get('authenticated', False)

def get_user_email() -> str:
    """認証済みユーザーのメールアドレス"""
    return app.storage.user.get('email', '')

def verify_login(email: str, password: str) -> tuple[bool, str]:
    """ログイン検証"""
    if not email or not password:
        return False, "メールアドレスとパスワードを入力してください"

    if "@" not in email:
        return False, "有効なメールアドレスを入力してください"

    domain = email.split("@")[1].lower()
    if domain not in [d.lower() for d in ALLOWED_DOMAINS]:
        return False, f"このドメイン（@{domain}）は許可されていません"

    if password != AUTH_PASSWORD:
        return False, "パスワードが間違っています"

    return True, ""

# =====================================
# ログインページ
# =====================================
@ui.page('/login')
def login_page():
    """ログインページ"""

    # 既に認証済みならダッシュボードへ
    if is_authenticated():
        ui.navigate.to('/')
        return

    with ui.card().classes('absolute-center w-96'):
        ui.label('MapComplete Dashboard').classes('text-2xl font-bold text-center w-full mb-4')
        ui.label('ログイン').classes('text-lg text-center w-full mb-4')

        email_input = ui.input('メールアドレス', placeholder='example@f-a-c.co.jp').classes('w-full')
        password_input = ui.input('パスワード', password=True, password_toggle_button=True).classes('w-full')
        error_label = ui.label('').classes('text-red-500 text-sm')

        def handle_login():
            email = email_input.value
            password = password_input.value

            success, message = verify_login(email, password)

            if success:
                app.storage.user['authenticated'] = True
                app.storage.user['email'] = email
                print(f"[AUTH] Login success: {email}")
                ui.navigate.to('/')
            else:
                error_label.text = message
                print(f"[AUTH] Login failed: {message}")

        ui.button('ログイン', on_click=handle_login).classes('w-full mt-4')

        ui.label('許可ドメイン: ' + ', '.join([f'@{d}' for d in ALLOWED_DOMAINS])).classes('text-xs text-gray-500 mt-4 text-center w-full')

# =====================================
# ダッシュボードページ
# =====================================
@ui.page('/')
def dashboard_page():
    """メインダッシュボード"""

    # 認証チェック
    if not is_authenticated():
        ui.navigate.to('/login')
        return

    # データ読み込み
    df = load_data()

    # 都道府県リスト
    prefectures = ['全国'] + sorted(df['prefecture'].dropna().unique().tolist()) if 'prefecture' in df.columns else ['全国']

    # 状態管理
    state = {
        'prefecture': '全国',
        'municipality': '全て',
        'tab': '市場概況'
    }

    # =====================================
    # ヘッダー
    # =====================================
    with ui.header().classes('bg-blue-600 text-white'):
        ui.label('MapComplete Dashboard').classes('text-xl font-bold')
        # データソース表示（Turso/CSV判定）
        if _data_source == "Turso DB":
            ui.label(f'[DB] {_data_source}').classes('text-sm bg-green-700 px-2 py-1 rounded')
        else:
            ui.label(f'[!] {_data_source}').classes('text-sm bg-yellow-700 px-2 py-1 rounded')
        ui.space()
        ui.label(f'Login: {get_user_email()}').classes('text-sm')

        def handle_logout():
            app.storage.user['authenticated'] = False
            app.storage.user['email'] = ''
            ui.navigate.to('/login')

        ui.button('ログアウト', on_click=handle_logout, color='white').props('flat')

    # =====================================
    # フィルタリング関数
    # =====================================
    def get_filtered_data():
        """現在のフィルター条件でデータをフィルタリング"""
        filtered = df.copy()
        if state['prefecture'] != '全国' and 'prefecture' in df.columns:
            filtered = filtered[filtered['prefecture'] == state['prefecture']]
        if state['municipality'] != '全て' and 'municipality' in df.columns:
            filtered = filtered[filtered['municipality'] == state['municipality']]
        return filtered

    # =====================================
    # コンテンツ表示関数（refreshable）
    # =====================================
    @ui.refreshable
    def show_content():
        """現在のタブに応じたコンテンツを表示"""
        filtered_df = get_filtered_data()
        tab = state['tab']

        with ui.column().classes('w-full p-4'):
            if tab == '市場概況':
                # 市場概況タブ
                ui.label(f'対象データ: {len(filtered_df):,} 件').classes('text-lg font-bold mb-4')

                # 集計値を計算
                total_applicants = int(filtered_df['applicant_count'].sum()) if 'applicant_count' in filtered_df.columns else len(filtered_df)
                male_total = int(filtered_df['male_count'].sum()) if 'male_count' in filtered_df.columns else 0
                female_total = int(filtered_df['female_count'].sum()) if 'female_count' in filtered_df.columns else 0

                with ui.row().classes('w-full gap-4 flex-wrap'):
                    with ui.card().classes('p-4'):
                        ui.label('総求職者数').classes('text-sm text-gray-500')
                        ui.label(f'{total_applicants:,}').classes('text-2xl font-bold text-blue-600')

                    if male_total > 0 or female_total > 0:
                        with ui.card().classes('p-4'):
                            ui.label('男性').classes('text-sm text-gray-500')
                            ui.label(f'{male_total:,}').classes('text-2xl font-bold text-blue-600')

                        with ui.card().classes('p-4'):
                            ui.label('女性').classes('text-sm text-gray-500')
                            ui.label(f'{female_total:,}').classes('text-2xl font-bold text-pink-600')

                # 都道府県別グラフ（ECharts）
                if 'prefecture' in filtered_df.columns and state['prefecture'] == '全国':
                    # 都道府県ごとにapplicant_countを集計
                    pref_data = filtered_df.groupby('prefecture')['applicant_count'].sum().sort_values(ascending=False).head(10)
                    ui.echart({
                        'title': {'text': '都道府県別求職者数（TOP 10）'},
                        'tooltip': {'trigger': 'axis'},
                        'xAxis': {'type': 'category', 'data': pref_data.index.tolist(), 'axisLabel': {'rotate': 45}},
                        'yAxis': {'type': 'value', 'name': '人数'},
                        'series': [{'data': [int(v) for v in pref_data.values], 'type': 'bar', 'name': '人数', 'itemStyle': {'color': '#3B82F6'}}]
                    }).classes('w-full h-96')

            elif tab == '人材属性':
                # 人材属性タブ
                ui.label('人材属性分析').classes('text-lg font-bold mb-4')

                # 平均年齢分布（ヒストグラム形式）
                if 'avg_age' in filtered_df.columns:
                    age_data = filtered_df['avg_age'].dropna()
                    # 年齢帯別に分類
                    bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, 100]
                    labels = ['~25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61~']
                    age_groups = pd.cut(age_data, bins=bins, labels=labels)
                    age_counts = age_groups.value_counts().sort_index()
                    ui.echart({
                        'title': {'text': '平均年齢分布'},
                        'tooltip': {'trigger': 'axis'},
                        'xAxis': {'type': 'category', 'data': age_counts.index.tolist(), 'name': '年齢帯'},
                        'yAxis': {'type': 'value', 'name': '件数'},
                        'series': [{'data': [int(v) for v in age_counts.values], 'type': 'bar', 'name': '件数', 'itemStyle': {'color': '#10B981'}}]
                    }).classes('w-full h-64')

                # 男女比率（パイチャート）
                if 'male_count' in filtered_df.columns and 'female_count' in filtered_df.columns:
                    male_total = int(filtered_df['male_count'].sum())
                    female_total = int(filtered_df['female_count'].sum())
                    ui.echart({
                        'title': {'text': '男女比率'},
                        'tooltip': {'trigger': 'item', 'formatter': '{b}: {c} ({d}%)'},
                        'legend': {'orient': 'vertical', 'left': 'left'},
                        'series': [{
                            'type': 'pie',
                            'radius': ['40%', '70%'],
                            'data': [
                                {'value': male_total, 'name': '男性', 'itemStyle': {'color': '#3B82F6'}},
                                {'value': female_total, 'name': '女性', 'itemStyle': {'color': '#EC4899'}}
                            ]
                        }]
                    }).classes('w-full h-96')

            elif tab == '地域・移動パターン':
                # 地域・移動パターンタブ
                ui.label('地域・移動パターン分析').classes('text-lg font-bold mb-4')
                ui.label('（実装予定）').classes('text-gray-500')

            elif tab == '需給バランス':
                # 需給バランスタブ
                ui.label('需給バランス分析').classes('text-lg font-bold mb-4')
                ui.label('（実装予定）').classes('text-gray-500')

    # =====================================
    # フィルター
    # =====================================
    with ui.row().classes('w-full p-4 bg-gray-100'):
        def on_prefecture_change(e):
            state['prefecture'] = e.value
            # 市区町村リストを更新
            if e.value == '全国' or 'municipality' not in df.columns:
                municipalities = ['全て']
            else:
                filtered = df[df['prefecture'] == e.value]
                municipalities = ['全て'] + sorted(filtered['municipality'].dropna().unique().tolist())
            municipality_select.options = municipalities
            municipality_select.value = '全て'
            state['municipality'] = '全て'
            show_content.refresh()

        prefecture_select = ui.select(
            prefectures,
            value='全国',
            label='都道府県',
            on_change=on_prefecture_change
        ).classes('w-48')

        def on_municipality_change(e):
            state['municipality'] = e.value
            show_content.refresh()

        municipality_select = ui.select(
            ['全て'],
            value='全て',
            label='市区町村',
            on_change=on_municipality_change
        ).classes('w-48')

    # =====================================
    # タブボタン
    # =====================================
    tab_names = ['市場概況', '人材属性', '地域・移動パターン', '需給バランス']

    with ui.row().classes('w-full justify-center gap-2 mb-4'):
        def create_tab_click_handler(name):
            def handler():
                state['tab'] = name
                # すべてのボタンのスタイルを更新
                for btn, btn_name in tab_buttons:
                    if btn_name == name:
                        btn.classes(remove='bg-gray-200', add='bg-blue-600 text-white')
                    else:
                        btn.classes(remove='bg-blue-600 text-white', add='bg-gray-200')
                show_content.refresh()
            return handler

        tab_buttons = []
        for name in tab_names:
            if name == '市場概況':
                btn = ui.button(name, on_click=create_tab_click_handler(name)).classes('bg-blue-600 text-white px-4 py-2')
            else:
                btn = ui.button(name, on_click=create_tab_click_handler(name)).classes('bg-gray-200 px-4 py-2')
            tab_buttons.append((btn, name))

    # =====================================
    # メインコンテンツ
    # =====================================
    with ui.card().classes('w-full'):
        show_content()


# =====================================
# アプリ起動
# =====================================
if __name__ in {"__main__", "__mp_main__"}:
    # Render対応: 環境変数PORTを使用
    port = int(os.getenv("PORT", 9090))  # 8080が使用中のため一時的に9090
    is_production = os.getenv("RENDER") is not None or os.getenv("PORT") is not None

    # セッション暗号化キー（本番は環境変数から）
    storage_secret = os.getenv("NICEGUI_STORAGE_SECRET", "nicegui_mapcomplete_secret_key_2025")

    print(f"[STARTUP] Starting NiceGUI app on port {port}...")
    print(f"[STARTUP] Production mode: {is_production}")

    ui.run(
        title='MapComplete Dashboard',
        host='0.0.0.0',  # 外部アクセス許可（Render必須）
        port=port,
        reload=not is_production,  # 本番では自動リロード無効
        storage_secret=storage_secret,
        show=False,  # 本番ではブラウザ自動起動しない
    )
