# NiceGUI版 MapComplete Dashboard

Reflex版からの移行テスト用PoC

## セットアップ

```bash
cd nicegui_app
pip install -r requirements.txt
```

## 起動

```bash
python main.py
```

ブラウザで http://localhost:8080 にアクセス

## ログイン情報

- **メールアドレス**: `xxx@f-a-c.co.jp` または `xxx@cyxen.co.jp`
- **パスワード**: `cyzen_2025`

## 機能

- [x] ログイン認証（ドメイン制限）
- [x] ダッシュボードUI（タブ構造）
- [x] 都道府県・市区町村フィルター
- [x] Plotlyグラフ表示
- [ ] Turso DB接続（現在はCSV読み込み）
- [ ] 地域・移動パターン分析
- [ ] 需給バランス分析

## ディレクトリ構成

```
nicegui_app/
├── main.py           # メインアプリケーション
├── requirements.txt  # 依存パッケージ
├── .env              # 環境変数
└── README.md         # このファイル
```

## Reflex版との比較

| 機能 | Reflex | NiceGUI |
|-----|--------|---------|
| 状態管理 | rx.State | app.storage |
| ルーティング | @rx.page | @ui.page |
| コンポーネント | rx.* | ui.* |
| グラフ | rx.plotly | ui.plotly |

## 本番デプロイ（Render）

```bash
# Dockerfile作成後
docker build -t nicegui-dashboard .
```

または render.yaml で設定
