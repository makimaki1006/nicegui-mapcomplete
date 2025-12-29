# -*- coding: utf-8 -*-
"""
コロプレスマップヘルパーモジュール（47都道府県対応版）

test_choropleth_v3.pyのロジックを拡張し、47都道府県のGeoJSONに対応。
NiceGUI Leafletウィジェットで使用。
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# GeoJSONディレクトリ
GEOJSON_DIR = Path(__file__).parent / "static" / "geojson"

# ============================================================
# 色定義（単一ソース - Python/JavaScript両方で使用）
# ============================================================
# 形式: [(threshold, color), ...] - thresholdを超えたら該当colorを使用
# 最後の要素はデフォルト色（threshold=0）
COLOR_CONFIG = {
    "count": [
        # 基本表示: 緑（少）→ 赤（多）
        (0.8, "#dc2626"),  # 赤（多い）
        (0.6, "#f97316"),  # オレンジ
        (0.4, "#eab308"),  # 黄
        (0.2, "#84cc16"),  # 黄緑
        (0.0, "#22c55e"),  # 緑（少ない）
    ],
    "balance": [
        # 流出入バランス: 赤（流出超過）〜 青（流入超過）
        (0.6, "#3b82f6"),  # 青（流入超過）
        (0.4, "#93c5fd"),  # 薄青
        (0.3, "#f1f5f9"),  # 白（均衡）
        (0.2, "#fca5a5"),  # 薄赤
        (0.0, "#dc2626"),  # 赤（流出超過）
    ],
    "inflow": [
        # 流入元: 緑のグラデーション
        (0.8, "#00ff88"),  # 明るい緑
        (0.6, "#22c55e"),  # 緑
        (0.4, "#4ade80"),
        (0.2, "#86efac"),
        (0.0, "#dcfce7"),  # 薄緑
    ],
    "competition": [
        # 競合地域: マゼンタのグラデーション
        (0.8, "#ff00ff"),  # マゼンタ
        (0.6, "#d946ef"),
        (0.4, "#e879f9"),
        (0.2, "#f0abfc"),
        (0.0, "#fae8ff"),  # 薄いピンク
    ],
}

# 特殊色定義
SPECIAL_COLORS = {
    "default": "#9ca3af",    # グレー（データなし）
    "selected": "#00d4ff",   # シアン（選択中）
    "inflow_highlight": "#00ff88",   # 緑（流入元ハイライト）
    "competition_highlight": "#ff00ff",  # マゼンタ（競合ハイライト）
}


def _get_color_from_config(ratio: float, mode: str) -> str:
    """COLOR_CONFIGから色を取得する共通関数"""
    config = COLOR_CONFIG.get(mode, COLOR_CONFIG["count"])
    for threshold, color in config:
        if ratio >= threshold:
            return color
    return config[-1][1]  # デフォルト（最後の色）


def _generate_js_color_function() -> str:
    """COLOR_CONFIGからJavaScriptのgetColor関数を自動生成"""
    js_lines = []
    js_lines.append("function getColor(value, maxV, m) {")
    js_lines.append(f'    if (maxV == 0) return "{SPECIAL_COLORS["default"]}";')
    js_lines.append("    var ratio = Math.min(value / maxV, 1.0);")

    for i, (mode, thresholds) in enumerate(COLOR_CONFIG.items()):
        condition = "if" if i == 0 else "} else if"
        js_lines.append(f'    {condition} (m === "{mode}") {{')
        for threshold, color in thresholds[:-1]:  # 最後以外
            js_lines.append(f'        if (ratio >= {threshold}) return "{color}";')
        # デフォルト色（最後）
        js_lines.append(f'        return "{thresholds[-1][1]}";')

    js_lines.append("    } else {")
    # count がデフォルト
    for threshold, color in COLOR_CONFIG["count"][:-1]:
        js_lines.append(f'        if (ratio >= {threshold}) return "{color}";')
    js_lines.append(f'        return "{COLOR_CONFIG["count"][-1][1]}";')
    js_lines.append("    }")
    js_lines.append("}")

    return "\n        ".join(js_lines)

# 都道府県コードマッピング
PREF_NAME_TO_CODE = {
    "北海道": "01", "青森県": "02", "岩手県": "03", "宮城県": "04",
    "秋田県": "05", "山形県": "06", "福島県": "07", "茨城県": "08",
    "栃木県": "09", "群馬県": "10", "埼玉県": "11", "千葉県": "12",
    "東京都": "13", "神奈川県": "14", "新潟県": "15", "富山県": "16",
    "石川県": "17", "福井県": "18", "山梨県": "19", "長野県": "20",
    "岐阜県": "21", "静岡県": "22", "愛知県": "23", "三重県": "24",
    "滋賀県": "25", "京都府": "26", "大阪府": "27", "兵庫県": "28",
    "奈良県": "29", "和歌山県": "30", "鳥取県": "31", "島根県": "32",
    "岡山県": "33", "広島県": "34", "山口県": "35", "徳島県": "36",
    "香川県": "37", "愛媛県": "38", "高知県": "39", "福岡県": "40",
    "佐賀県": "41", "長崎県": "42", "熊本県": "43", "大分県": "44",
    "宮崎県": "45", "鹿児島県": "46", "沖縄県": "47"
}

PREF_CODE_TO_NAME_EN = {
    "01": "hokkaido", "02": "aomori", "03": "iwate", "04": "miyagi",
    "05": "akita", "06": "yamagata", "07": "fukushima", "08": "ibaraki",
    "09": "tochigi", "10": "gunma", "11": "saitama", "12": "chiba",
    "13": "tokyo", "14": "kanagawa", "15": "niigata", "16": "toyama",
    "17": "ishikawa", "18": "fukui", "19": "yamanashi", "20": "nagano",
    "21": "gifu", "22": "shizuoka", "23": "aichi", "24": "mie",
    "25": "shiga", "26": "kyoto", "27": "osaka", "28": "hyogo",
    "29": "nara", "30": "wakayama", "31": "tottori", "32": "shimane",
    "33": "okayama", "34": "hiroshima", "35": "yamaguchi", "36": "tokushima",
    "37": "kagawa", "38": "ehime", "39": "kochi", "40": "fukuoka",
    "41": "saga", "42": "nagasaki", "43": "kumamoto", "44": "oita",
    "45": "miyazaki", "46": "kagoshima", "47": "okinawa"
}

# 都道府県の中心座標
PREF_CENTERS = {
    "北海道": (43.0642, 141.3469), "青森県": (40.8243, 140.7400),
    "岩手県": (39.7036, 141.1527), "宮城県": (38.2688, 140.8721),
    "秋田県": (39.7186, 140.1024), "山形県": (38.2404, 140.3633),
    "福島県": (37.7500, 140.4678), "茨城県": (36.3418, 140.4468),
    "栃木県": (36.5657, 139.8836), "群馬県": (36.3911, 139.0608),
    "埼玉県": (35.8569, 139.6489), "千葉県": (35.6050, 140.1233),
    "東京都": (35.6895, 139.6917), "神奈川県": (35.4478, 139.6425),
    "新潟県": (37.9026, 139.0236), "富山県": (36.6953, 137.2113),
    "石川県": (36.5947, 136.6256), "福井県": (36.0652, 136.2216),
    "山梨県": (35.6642, 138.5683), "長野県": (36.6513, 138.1810),
    "岐阜県": (35.3912, 136.7223), "静岡県": (34.9769, 138.3831),
    "愛知県": (35.1802, 136.9066), "三重県": (34.7303, 136.5086),
    "滋賀県": (35.0045, 135.8686), "京都府": (35.0116, 135.7681),
    "大阪府": (34.6863, 135.5200), "兵庫県": (34.6913, 135.1830),
    "奈良県": (34.6851, 135.8049), "和歌山県": (34.2260, 135.1675),
    "鳥取県": (35.5039, 134.2378), "島根県": (35.4723, 133.0505),
    "岡山県": (34.6618, 133.9344), "広島県": (34.3966, 132.4596),
    "山口県": (34.1859, 131.4706), "徳島県": (34.0658, 134.5593),
    "香川県": (34.3401, 134.0434), "愛媛県": (33.8416, 132.7656),
    "高知県": (33.5597, 133.5311), "福岡県": (33.6064, 130.4183),
    "佐賀県": (33.2494, 130.2988), "長崎県": (32.7448, 129.8737),
    "熊本県": (32.7898, 130.7417), "大分県": (33.2382, 131.6126),
    "宮崎県": (31.9111, 131.4239), "鹿児島県": (31.5602, 130.5581),
    "沖縄県": (26.2124, 127.6809),
}

# GeoJSONキャッシュ（LRU方式、最大3都道府県 - Render 512MB対応）
_geojson_cache: Dict[str, dict] = {}
_cache_order: List[str] = []  # LRU順序追跡
_CACHE_MAX_SIZE = 3  # Renderメモリ制限対応（512MB）


def get_geojson_path(prefecture: str) -> Optional[Path]:
    """都道府県名からGeoJSONファイルパスを取得"""
    code = PREF_NAME_TO_CODE.get(prefecture)
    if not code:
        return None
    name_en = PREF_CODE_TO_NAME_EN.get(code)
    if not name_en:
        return None
    path = GEOJSON_DIR / f"{code}_{name_en}.json"
    return path if path.exists() else None


def load_geojson(prefecture: str) -> Optional[dict]:
    """GeoJSONを読み込む（LRUキャッシュ付き、最大10件）"""
    global _cache_order

    # キャッシュヒット
    if prefecture in _geojson_cache:
        # LRU更新（最近使用したものを末尾に移動）
        if prefecture in _cache_order:
            _cache_order.remove(prefecture)
        _cache_order.append(prefecture)
        return _geojson_cache[prefecture]

    path = get_geojson_path(prefecture)
    if not path:
        print(f"[CHOROPLETH] GeoJSON not found for {prefecture}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # キャッシュサイズ制限（LRU削除）
        while len(_geojson_cache) >= _CACHE_MAX_SIZE and _cache_order:
            oldest = _cache_order.pop(0)
            if oldest in _geojson_cache:
                del _geojson_cache[oldest]
                print(f"[CHOROPLETH] Evicted from cache: {oldest}")

        _geojson_cache[prefecture] = data
        _cache_order.append(prefecture)
        print(f"[CHOROPLETH] Loaded GeoJSON for {prefecture}: {len(data.get('features', []))} features (cache: {len(_geojson_cache)}/{_CACHE_MAX_SIZE})")
        return data
    except Exception as e:
        print(f"[CHOROPLETH] Error loading GeoJSON for {prefecture}: {e}")
        return None


def clear_geojson_cache():
    """キャッシュをクリア"""
    global _cache_order
    _geojson_cache.clear()
    _cache_order = []
    print("[CHOROPLETH] Cache cleared")


def preload_geojson(prefectures: List[str]):
    """複数の都道府県を事前ロード"""
    for pref in prefectures:
        load_geojson(pref)


def get_pref_center(prefecture: str) -> tuple:
    """都道府県の中心座標を取得"""
    return PREF_CENTERS.get(prefecture, (36.5, 138.0))


def get_color_by_value(value: float, max_value: float, mode: str = "count") -> str:
    """値に応じて色を返す（モード別）

    COLOR_CONFIGの単一ソースから色を取得。
    JavaScript側も同じCOLOR_CONFIGから自動生成されるため、
    色の変更はCOLOR_CONFIGを修正するだけで両方に反映される。
    """
    if max_value == 0:
        return SPECIAL_COLORS["default"]

    ratio = min(value / max_value, 1.0)
    return _get_color_from_config(ratio, mode)


def style_geojson_feature(
    feature: dict,
    municipality_data: Dict[str, dict],
    mode: str = "count",
    selected_muni: Optional[str] = None,
    inflow_sources: Optional[List[str]] = None,
    competing_areas: Optional[List[str]] = None
) -> dict:
    """GeoJSON featureにスタイルを適用

    色はSPECIAL_COLORSおよびCOLOR_CONFIGから取得（単一ソース）。
    """
    props = feature.get("properties", {})
    muni_name = props.get("N03_004", "")  # 市区町村名

    # デフォルトスタイル（SPECIAL_COLORSから取得）
    style = {
        "color": "#ffffff",      # 境界線の色
        "weight": 1,
        "fillColor": SPECIAL_COLORS["default"],
        "fillOpacity": 0.6,
    }

    if not muni_name:
        return style

    # 選択中の市区町村（SPECIAL_COLORSから取得）
    if selected_muni and muni_name == selected_muni:
        style["fillColor"] = SPECIAL_COLORS["selected"]
        style["weight"] = 3
        style["fillOpacity"] = 0.8
        return style

    # 流入元（SPECIAL_COLORSから取得）
    if inflow_sources and muni_name in inflow_sources:
        style["fillColor"] = SPECIAL_COLORS["inflow_highlight"]
        style["weight"] = 2
        style["fillOpacity"] = 0.7
        return style

    # 競合地域（SPECIAL_COLORSから取得）
    if competing_areas and muni_name in competing_areas:
        style["fillColor"] = SPECIAL_COLORS["competition_highlight"]
        style["weight"] = 2
        style["fillOpacity"] = 0.7
        return style

    # 通常の色分け
    data = municipality_data.get(muni_name, {})
    if mode == "count":
        value = data.get("count", 0)
    elif mode == "inflow":
        value = data.get("inflow", 0)
    elif mode == "balance":
        inflow = data.get("inflow", 0)
        outflow = data.get("outflow", 0)
        value = inflow - outflow + 50  # 0-100スケールに正規化
    elif mode == "competition":
        value = data.get("competition", 0)
    else:
        value = data.get("count", 0)

    # 最大値を取得
    max_val = max(
        (d.get("count", 0) if mode == "count" else
         d.get("inflow", 0) if mode == "inflow" else
         d.get("competition", 0) if mode == "competition" else
         abs(d.get("inflow", 0) - d.get("outflow", 0)) + 50)
        for d in municipality_data.values()
    ) if municipality_data else 1

    style["fillColor"] = get_color_by_value(value, max_val, mode)
    return style


def prepare_styled_geojson(
    prefecture: str,
    municipality_data: Dict[str, dict],
    mode: str = "count",
    selected_muni: Optional[str] = None,
    inflow_sources: Optional[List[str]] = None,
    competing_areas: Optional[List[str]] = None
) -> Optional[dict]:
    """スタイル付きGeoJSONを準備"""
    geojson = load_geojson(prefecture)
    if not geojson:
        return None

    styled_features = []
    for feature in geojson.get("features", []):
        styled_feature = feature.copy()
        style = style_geojson_feature(
            feature,
            municipality_data,
            mode,
            selected_muni,
            inflow_sources,
            competing_areas
        )
        # プロパティにスタイルを追加
        if "properties" not in styled_feature:
            styled_feature["properties"] = {}
        styled_feature["properties"]["_style"] = style
        styled_features.append(styled_feature)

    return {
        "type": "FeatureCollection",
        "features": styled_features
    }


def get_municipality_name_from_feature(feature: dict) -> str:
    """GeoJSON featureから市区町村名を取得"""
    props = feature.get("properties", {})
    return props.get("N03_004", "")


def point_in_polygon(lat: float, lng: float, polygon_coords: list) -> bool:
    """点が多角形内にあるかをレイキャスティング法で判定

    Args:
        lat: 緯度
        lng: 経度
        polygon_coords: [[lat, lng], ...] 形式の座標リスト
    """
    n = len(polygon_coords)
    inside = False

    j = n - 1
    for i in range(n):
        yi, xi = polygon_coords[i][0], polygon_coords[i][1]
        yj, xj = polygon_coords[j][0], polygon_coords[j][1]

        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def find_municipality_at_point(lat: float, lng: float, geojson_data: dict) -> Optional[str]:
    """指定座標にある市区町村を見つける

    Args:
        lat: 緯度
        lng: 経度
        geojson_data: GeoJSONデータ

    Returns:
        市区町村名、見つからない場合はNone
    """
    if not geojson_data:
        return None

    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        muni_name = props.get("N03_004", "")
        if not muni_name:
            continue

        geometry = feature.get("geometry", {})

        if geometry.get("type") == "Polygon":
            coords = geometry["coordinates"][0]
            # GeoJSONは [lng, lat] 形式なので [lat, lng] に変換
            latlngs = [[c[1], c[0]] for c in coords]
            if point_in_polygon(lat, lng, latlngs):
                return muni_name

        elif geometry.get("type") == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                coords = polygon[0]
                latlngs = [[c[1], c[0]] for c in coords]
                if point_in_polygon(lat, lng, latlngs):
                    return muni_name

    return None


def create_geojson_style_function(
    municipality_data: Dict[str, dict],
    mode: str = "count",
    selected_muni: Optional[str] = None,
    inflow_sources: Optional[List[str]] = None,
    competing_areas: Optional[List[str]] = None
) -> str:
    """NiceGUI Leaflet用のスタイル関数（JavaScript文字列）を生成"""
    # municipality_dataをJSON文字列に変換
    data_json = json.dumps(municipality_data, ensure_ascii=False)
    selected_json = json.dumps(selected_muni) if selected_muni else "null"
    inflow_json = json.dumps(inflow_sources or [])
    competing_json = json.dumps(competing_areas or [])

    # 最大値計算
    if mode == "count":
        max_val = max((d.get("count", 0) for d in municipality_data.values()), default=1)
    elif mode == "inflow":
        max_val = max((d.get("inflow", 0) for d in municipality_data.values()), default=1)
    elif mode == "competition":
        max_val = max((d.get("competition", 0) for d in municipality_data.values()), default=1)
    else:
        max_val = 100

    # COLOR_CONFIGから自動生成されたgetColor関数を使用（単一ソース）
    js_get_color = _generate_js_color_function()

    # SPECIAL_COLORSから色を取得（単一ソース）
    default_color = SPECIAL_COLORS["default"]
    selected_color = SPECIAL_COLORS["selected"]
    inflow_highlight = SPECIAL_COLORS["inflow_highlight"]
    competition_highlight = SPECIAL_COLORS["competition_highlight"]

    return f"""
    (function() {{
        var data = {data_json};
        var selected = {selected_json};
        var inflow = {inflow_json};
        var competing = {competing_json};
        var maxVal = {max_val};
        var mode = "{mode}";

        // COLOR_CONFIGから自動生成された色関数
        {js_get_color}

        return function(feature) {{
            var name = feature.properties.N03_004 || "";
            var style = {{ color: "#fff", weight: 1, fillOpacity: 0.6, fillColor: "{default_color}" }};

            if (selected && name === selected) {{
                style.fillColor = "{selected_color}";
                style.weight = 3;
                style.fillOpacity = 0.8;
                return style;
            }}
            if (inflow.indexOf(name) >= 0) {{
                style.fillColor = "{inflow_highlight}";
                style.weight = 2;
                style.fillOpacity = 0.7;
                return style;
            }}
            if (competing.indexOf(name) >= 0) {{
                style.fillColor = "{competition_highlight}";
                style.weight = 2;
                style.fillOpacity = 0.7;
                return style;
            }}

            var d = data[name] || {{}};
            var value = 0;
            if (mode === "count") value = d.count || 0;
            else if (mode === "inflow") value = d.inflow || 0;
            else if (mode === "balance") value = (d.inflow || 0) - (d.outflow || 0) + 50;
            else if (mode === "competition") value = d.competition || 0;

            style.fillColor = getColor(value, maxVal, mode);
            return style;
        }};
    }})()
    """
