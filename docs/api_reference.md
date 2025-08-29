# Stock Analyzer API リファレンス

## 目次

1. [CLI コマンドリファレンス](#cliコマンドリファレンス)
2. [内部API仕様](#内部api仕様)
3. [データ構造](#データ構造)
4. [設定ファイル仕様](#設定ファイル仕様)
5. [エラーコード](#エラーコード)
6. [使用例](#使用例)

---

## CLIコマンドリファレンス

### 基本構文

```bash
uv run python -m stock_analyzer.cli.main [COMMAND] [OPTIONS] [ARGS]
```

---

### `get-data` - 株価データ取得

基本的な株価データを取得して表示します。

```bash
uv run python -m stock_analyzer.cli.main get-data SYMBOL [OPTIONS]
```

#### パラメータ

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|-----------|----|----|-----------|------|
| `SYMBOL` | str | ✅ | - | 株式シンボル（例: AAPL, MSFT） |

#### オプション

| オプション | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `--period` | choice | `1y` | データ取得期間 |
| `--info` | flag | `False` | 会社情報も表示する |

#### 期間の選択肢

- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

#### 使用例

```bash
# 基本的な使用
uv run python -m stock_analyzer.cli.main get-data AAPL

# 6ヶ月間のデータを取得
uv run python -m stock_analyzer.cli.main get-data AAPL --period 6mo

# 会社情報も含めて表示
uv run python -m stock_analyzer.cli.main get-data AAPL --info
```

#### 出力フォーマット

```
📊 AAPL 株価データ:
期間: 2024-02-28 ～ 2024-08-28 (182日分)
最新価格: $230.51
期間最高値: $237.23
期間最安値: $164.08
平均出来高: 52,441,832

📈 最新5日間の終値:
2024-08-26: $227.52 (-0.50, -0.2%) 📉
2024-08-27: $228.03 (+0.51, +0.2%) 📈
2024-08-28: $230.51 (+2.48, +1.1%) 📈
```

---

### `compare` - 銘柄価格比較

複数銘柄の現在価格を比較表示します。

```bash
uv run python -m stock_analyzer.cli.main compare SYMBOL1 SYMBOL2 [SYMBOL...]
```

#### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|----|----|------|
| `SYMBOLS` | str... | ✅ | 比較する銘柄リスト |

#### 使用例

```bash
# 複数銘柄の価格比較
uv run python -m stock_analyzer.cli.main compare AAPL MSFT GOOGL AMZN TSLA
```

---

### `analyze` - テクニカル分析

指定銘柄のテクニカル分析を実行します。

```bash
uv run python -m stock_analyzer.cli.main analyze SYMBOL [OPTIONS]
```

#### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|----|----|------|
| `SYMBOL` | str | ✅ | 分析する株式シンボル |

#### オプション

| オプション | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `--period` | choice | `6mo` | データ取得期間 |
| `--signals` | flag | `False` | 売買シグナル分析も表示 |

#### 使用例

```bash
# 基本的なテクニカル分析
uv run python -m stock_analyzer.cli.main analyze AAPL

# 売買シグナル分析も含める
uv run python -m stock_analyzer.cli.main analyze AAPL --signals

# 1年間のデータで分析
uv run python -m stock_analyzer.cli.main analyze AAPL --period 1y --signals
```

---

### `predict` - 未来予測

現在データを使って未来の価格動向を予測します。

```bash
uv run python -m stock_analyzer.cli.main predict SYMBOL [OPTIONS]
```

#### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|----|----|------|
| `SYMBOL` | str | ✅ | 予測する株式シンボル |

#### オプション

| オプション | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `--prediction-days` | int | `30` | 予測日数 |
| `--training-months` | int | `24` | 訓練期間（月数） |
| `--prediction-type` | choice | `direction` | 予測タイプ |

#### 予測タイプ

- `direction`: 上昇/下降の方向性予測
- `return`: リターン率の数値予測

#### 使用例

```bash
# 30日後の方向性を予測
uv run python -m stock_analyzer.cli.main predict AAPL

# 60日後のリターン率を予測
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 60 --prediction-type return

# より長い訓練期間で予測
uv run python -m stock_analyzer.cli.main predict AAPL --training-months 36
```

#### 出力フォーマット

```
🔮 60日後の予測結果:
============================================================
予測方向: 上昇
信頼度: 22.3%
予測スコア: 0.611
推定リターン: +3.1%
目標価格: $237.69

💡 投資判断:
➡️ 中立・様子見

🔍 2025-10-27以降に予測精度を検証するコマンド:
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2025-08-28 --validation-date 2025-10-27
```

---

### `backtest` - 過去データ検証

過去の特定時点での予測精度を検証します。

```bash
uv run python -m stock_analyzer.cli.main backtest SYMBOL [OPTIONS]
```

#### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|----|----|------|
| `SYMBOL` | str | ✅ | 検証する株式シンボル |

#### オプション

| オプション | 型 | 必須 | デフォルト | 説明 |
|------------|----|----|-----------|------|
| `--investment-date` | str | ✅ | - | 投資日（YYYY-MM-DD） |
| `--validation-date` | str | ✅ | - | 検証日（YYYY-MM-DD） |
| `--training-months` | int | | `24` | 訓練期間（月数） |
| `--prediction-type` | choice | | `direction` | 予測タイプ |

#### 使用例

```bash
# 特定期間の予測精度を検証
uv run python -m stock_analyzer.cli.main backtest AAPL \
  --investment-date 2024-07-01 \
  --validation-date 2024-08-25

# リターン予測の検証
uv run python -m stock_analyzer.cli.main backtest AAPL \
  --investment-date 2024-06-01 \
  --validation-date 2024-07-31 \
  --prediction-type return
```

---

### `portfolio` - ポートフォリオ分析

複数銘柄の最適ポートフォリオを構築・分析します。

```bash
uv run python -m stock_analyzer.cli.main portfolio [SYMBOLS] [OPTIONS]
```

#### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|----|----|------|
| `SYMBOLS` | str... | | 分析する銘柄リスト（オプション使用時は不要） |

#### オプション

| オプション | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `--period` | choice | `1y` | データ取得期間 |
| `--investment-amount` | float | `100000.0` | 総投資金額（USD） |
| `--max-stocks` | int | `10` | 最大銘柄数 |
| `--risk-tolerance` | float | `0.3` | リスク許容度（0-1） |
| `--export-csv` | flag | `False` | CSV出力 |
| `--preset` | str | - | プリセット銘柄グループ |
| `--watchlist` | str | - | ウォッチリスト名 |
| `--symbols-file` | str | - | 銘柄リストファイル |
| `--list-presets` | flag | `False` | プリセット一覧表示 |
| `--list-watchlists` | flag | `False` | ウォッチリスト一覧表示 |

#### 使用例

```bash
# 基本的なポートフォリオ分析
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL AMZN TSLA

# 投資金額を指定
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL \
  --investment-amount 50000 --max-stocks 3

# プリセットを使用
uv run python -m stock_analyzer.cli.main portfolio --preset tech-giants

# CSV出力付き
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL --export-csv
```

---

### `compare-advanced` - 詳細銘柄比較

複数銘柄の詳細比較分析を実行します。

```bash
uv run python -m stock_analyzer.cli.main compare-advanced [SYMBOLS] [OPTIONS]
```

#### パラメータ・オプション

`portfolio`コマンドと同様ですが、以下が追加されます：

| オプション | 型 | デフォルト | 説明 |
|------------|----|-----------|----|
| `--sort-by` | str | `investment_score` | ソート基準 |

#### ソート基準

- `investment_score`: 投資スコア
- `risk_score`: リスクスコア
- `current_price`: 現在価格
- `return_5d`: 5日後リターン予測
- `return_30d`: 30日後リターン予測

#### 使用例

```bash
# 詳細比較分析
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL

# リスクスコアでソート
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL --sort-by risk_score

# プリセット使用でCSV出力
uv run python -m stock_analyzer.cli.main compare-advanced --preset tech-giants --export-csv
```

---

### `config` - 設定管理

アプリケーションの設定を管理します。

```bash
uv run python -m stock_analyzer.cli.main config [OPTIONS]
```

#### オプション

| オプション | 型 | 説明 |
|------------|----|----|
| `--show` | flag | 現在の設定を表示 |
| `--init` | flag | デフォルト設定ファイルを作成 |
| `--set` | str | 設定キー（--valueと組み合わせ） |
| `--value` | str | 設定値 |
| `--add-watchlist` | str | ウォッチリスト名 |
| `--watchlist-symbols` | str | ウォッチリスト銘柄（カンマ区切り） |

#### 使用例

```bash
# 現在の設定を表示
uv run python -m stock_analyzer.cli.main config --show

# デフォルト設定を作成
uv run python -m stock_analyzer.cli.main config --init

# 設定値を変更
uv run python -m stock_analyzer.cli.main config --set general.default_period --value 1y

# ウォッチリストを追加
uv run python -m stock_analyzer.cli.main config \
  --add-watchlist "tech-stocks" \
  --watchlist-symbols "AAPL,MSFT,GOOGL,META,NVDA"
```

---

## 内部API仕様

### データ取得API

#### `get_stock_data()`

```python
def get_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Yahoo Finance APIから株価データを取得

    Args:
        symbol: 株式シンボル（例: "AAPL"）
        period: 取得期間（例: "1y", "6mo"）

    Returns:
        pd.DataFrame: OHLCV データフレーム
        Columns: ["Open", "High", "Low", "Close", "Volume"]
        Index: DatetimeIndex

    Raises:
        DataFetchError: データ取得に失敗した場合
        ValueError: 無効なパラメータの場合
    """
```

#### `get_company_info()`

```python
def get_company_info(symbol: str) -> Dict[str, Any]:
    """
    企業情報を取得

    Args:
        symbol: 株式シンボル

    Returns:
        Dict[str, Any]: 企業情報
        - company_name: str
        - sector: str
        - industry: str
        - market_cap: int
        - employees: int
        - website: str
    """
```

### テクニカル分析API

#### `calculate_all_indicators()`

```python
def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """
    全テクニカル指標を計算

    Args:
        data: OHLCV データフレーム

    Returns:
        Dict[str, float]: 指標辞書
        - sma_5, sma_10, sma_20, sma_50: 移動平均
        - ema_12, ema_26: 指数移動平均
        - rsi: RSI (14日)
        - macd: MACD
        - macd_signal: MACDシグナル
        - macd_histogram: MACDヒストグラム
        - bb_upper, bb_lower: ボリンジャーバンド
        - bb_position: BB内での位置（0-1）
        - volume_ratio: 出来高比率
        - volatility: ボラティリティ
    """
```

#### 個別指標計算関数

```python
def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """単純移動平均を計算"""

def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """指数移動平均を計算"""

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSIを計算"""

def calculate_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Dict[str, pd.Series]:
    """MACDを計算"""

def calculate_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """ボリンジャーバンドを計算"""

def calculate_volume_indicators(
    data: pd.DataFrame,
    period: int = 20
) -> Dict[str, pd.Series]:
    """出来高指標を計算"""
```

### 機械学習API

#### `LightGBMStockPredictor`

```python
class LightGBMStockPredictor:
    """LightGBMベースの株価予測モデル"""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: モデル識別名
        """

    def train_model(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_columns: List[str],
        n_splits: int = 5,
        hyperparams: Dict[str, Any] | None = None
    ) -> None:
        """
        モデルを訓練

        Args:
            features: 特徴量データ
            targets: 目的変数データ
            target_columns: 予測対象カラム
            n_splits: 交差検証分割数
            hyperparams: ハイパーパラメータ
        """

    def predict(
        self,
        features: pd.DataFrame,
        target_columns: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        予測を実行

        Args:
            features: 特徴量データ
            target_columns: 予測対象カラム

        Returns:
            Dict[str, np.ndarray]: 予測結果
        """

    def evaluate(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_columns: List[str]
    ) -> Dict[str, float]:
        """
        モデルを評価

        Returns:
            Dict[str, float]: 評価指標
            - accuracy: 精度（分類問題）
            - rmse: RMSE（回帰問題）
            - mae: MAE（回帰問題）
            - r2: 決定係数（回帰問題）
        """
```

### バックテストAPI

#### `BacktestSimulator`

```python
class BacktestSimulator:
    """時点指定型予測シミュレーター"""

    def run_point_in_time_simulation(
        self,
        symbol: str,
        investment_date: str,
        validation_date: str,
        training_period_months: int = 24,
        prediction_type: str = "direction"
    ) -> Dict[str, Any]:
        """
        時点指定シミュレーションを実行

        Args:
            symbol: 株式シンボル
            investment_date: 投資日（"YYYY-MM-DD"）
            validation_date: 検証日（"YYYY-MM-DD"）
            training_period_months: 訓練期間（月数）
            prediction_type: "direction" or "return"

        Returns:
            Dict[str, Any]: シミュレーション結果
            - symbol: str
            - investment_date: str
            - validation_date: str
            - prediction_days: int
            - prediction_type: str
            - prediction: Dict (予測結果)
            - actual: Dict (実際の結果)
            - accuracy: bool (正解/不正解)
            - confidence_score: float (信頼性スコア 0-100)
            - prediction_summary: str (結果サマリー)
        """
```

---

## データ構造

### 株価データ (OHLCV)

```python
# pd.DataFrame 形式
columns = ["Open", "High", "Low", "Close", "Volume"]
index = pd.DatetimeIndex  # 日付インデックス

# 例
data = pd.DataFrame({
    "Open": [230.00, 231.50, ...],
    "High": [232.00, 233.00, ...],
    "Low": [229.50, 230.00, ...],
    "Close": [231.00, 232.50, ...],
    "Volume": [50000000, 52000000, ...]
}, index=pd.to_datetime(["2024-08-26", "2024-08-27", ...]))
```

### テクニカル指標結果

```python
indicators = {
    # 移動平均
    "sma_5": 230.45,
    "sma_10": 229.80,
    "sma_20": 228.15,
    "sma_50": 225.60,

    # 指数移動平均
    "ema_12": 230.20,
    "ema_26": 227.90,

    # オシレーター
    "rsi": 58.2,
    "macd": 0.825,
    "macd_signal": 0.642,
    "macd_histogram": 0.183,

    # ボリンジャーバンド
    "bb_upper": 235.50,
    "bb_lower": 220.80,
    "bb_position": 0.653,  # 0-1の相対位置

    # 出来高
    "volume_ratio": 1.15,  # 平均出来高との比率
    "volume_sma": 48500000,

    # ボラティリティ
    "volatility": 0.024,  # 20日標準偏差
}
```

### 予測結果

```python
# 方向性予測の場合
prediction_result = {
    "prediction_type": "direction",
    "prediction_days": 30,
    "predicted_value": 0.725,  # 0-1のスコア
    "predicted_direction": "上昇",
    "confidence": 0.450,  # 信頼度 0-1
    "prediction_date": "2024-08-28"
}

# リターン予測の場合
prediction_result = {
    "prediction_type": "return",
    "prediction_days": 30,
    "predicted_value": 3.25,  # 予測リターン%
    "prediction_date": "2024-08-28"
}
```

### バックテスト結果

```python
backtest_result = {
    "symbol": "AAPL",
    "investment_date": "2024-07-01",
    "validation_date": "2024-08-25",
    "prediction_days": 55,
    "prediction_type": "direction",

    # 予測結果
    "prediction": {
        "predicted_direction": "上昇",
        "confidence": 0.213,
        "predicted_value": 0.607
    },

    # 実際の結果
    "actual": {
        "investment_price": 215.50,
        "validation_price": 225.79,
        "actual_return": 4.78,
        "actual_direction": "上昇",
        "actual_direction_binary": 1
    },

    # 評価結果
    "accuracy": True,  # 正解/不正解
    "confidence_score": 84.3,  # 信頼性スコア 0-100
    "direction_accuracy": True,
    "prediction_summary": "予測: 上昇 (信頼度: 21.3%) | 実際: 上昇 | 正解: ○"
}
```

### ポートフォリオ結果

```python
portfolio_result = {
    # 分析サマリー
    "analysis_summary": {
        "総分析銘柄数": 5,
        "選択銘柄数": 3,
        "総投資金額": "$100,000",
        "ポートフォリオリスク": "中程度",
        "期待リターン": "8.5%"
    },

    # 推奨構成
    "portfolio_stocks": [
        {
            "symbol": "AAPL",
            "weight": 0.40,
            "allocation_amount": 40000,
            "investment_score": 78.5,
            "risk_score": 35.2,
            "recommendation": "強い買い"
        },
        {
            "symbol": "MSFT",
            "weight": 0.35,
            "allocation_amount": 35000,
            "investment_score": 72.1,
            "risk_score": 28.9,
            "recommendation": "買い"
        },
        # ...
    ],

    # 推奨アクション
    "recommendations": {
        "action": "積極的投資",
        "risk_assessment": "適度なリスクで良好なリターン期待",
        "reasoning": [
            "高い投資スコアの銘柄が中心",
            "リスクの分散が適切",
            "成長期待の高いテック株中心"
        ]
    }
}
```

---

## 設定ファイル仕様

### 設定ファイル場所

```bash
# 設定ファイルパス
~/.config/stock_analyzer/config.yaml

# キャッシュディレクトリ
~/.cache/stock_analyzer/
```

### 設定ファイル形式

```yaml
# ~/.config/stock_analyzer/config.yaml

general:
  default_period: "1y"                    # デフォルト取得期間
  default_investment_amount: 100000.0     # デフォルト投資金額
  default_max_stocks: 10                  # デフォルト最大銘柄数
  default_risk_tolerance: 0.3             # デフォルトリスク許容度
  auto_export_csv: false                  # 自動CSV出力

analysis:
  # テクニカル指標のパラメータ
  sma_periods: [5, 10, 20, 50]           # SMA期間
  ema_periods: [12, 26]                  # EMA期間
  rsi_period: 14                         # RSI期間
  macd_fast: 12                          # MACD高速期間
  macd_slow: 26                          # MACD低速期間
  macd_signal: 9                         # MACDシグナル期間
  bb_period: 20                          # ボリンジャーバンド期間
  bb_std_dev: 2.0                        # BB標準偏差倍率
  volume_period: 20                      # 出来高平均期間

portfolio:
  min_allocation: 0.05                   # 最小配分比率
  max_allocation: 0.4                    # 最大配分比率
  rebalance_threshold: 0.05              # リバランス閾値
  risk_free_rate: 0.02                   # リスクフリーレート

ml:
  # 機械学習パラメータ
  default_training_months: 24            # デフォルト訓練期間
  cross_validation_splits: 5             # 交差検証分割数
  feature_selection_threshold: 0.01     # 特徴量選択閾値
  lightgbm_params:
    objective: "binary"
    metric: "binary_logloss"
    boosting_type: "gbdt"
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.9

output:
  log_level: "INFO"                      # ログレベル
  log_format: "console"                  # ログフォーマット
  csv_encoding: "utf-8-sig"             # CSV文字エンコーディング
  date_format: "%Y-%m-%d"               # 日付フォーマット
  decimal_places: 2                     # 小数点桁数

# デフォルト銘柄リスト
default_symbols:
  - "AAPL"
  - "MSFT"
  - "GOOGL"
  - "AMZN"
  - "TSLA"

# ウォッチリスト
watchlists:
  tech-giants:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
    - "META"
    - "NVDA"
    - "AMZN"
    - "TSLA"

  dividend-stocks:
    - "JNJ"
    - "PG"
    - "KO"
    - "PEP"
    - "MCD"

  sp500-top10:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
    - "AMZN"
    - "NVDA"
    - "META"
    - "TSLA"
    - "BRK.B"
    - "UNH"
    - "JNJ"

# プリセット設定
presets:
  conservative:
    risk_tolerance: 0.2
    max_stocks: 8
    min_allocation: 0.1

  aggressive:
    risk_tolerance: 0.6
    max_stocks: 15
    min_allocation: 0.03
```

### 設定値の型

| 設定キー | 型 | 説明 | デフォルト |
|---------|----|----|-----------|
| `general.default_period` | str | 期間 | `"1y"` |
| `general.default_investment_amount` | float | 投資金額 | `100000.0` |
| `general.default_max_stocks` | int | 最大銘柄数 | `10` |
| `general.default_risk_tolerance` | float | リスク許容度 | `0.3` |
| `analysis.rsi_period` | int | RSI期間 | `14` |
| `ml.default_training_months` | int | 訓練期間 | `24` |
| `output.log_level` | str | ログレベル | `"INFO"` |

---

## エラーコード

### データ取得エラー

| エラーコード | メッセージ | 原因 | 解決策 |
|-------------|-----------|------|-------|
| `DATA_FETCH_001` | `Failed to fetch data for {symbol}` | 無効なシンボル | シンボルの確認 |
| `DATA_FETCH_002` | `No data available for period {period}` | データなし | 期間の調整 |
| `DATA_FETCH_003` | `API rate limit exceeded` | レート制限 | 時間をおいて再試行 |
| `DATA_FETCH_004` | `Network connection error` | ネットワークエラー | 接続の確認 |

### 分析エラー

| エラーコード | メッセージ | 原因 | 解決策 |
|-------------|-----------|------|-------|
| `ANALYSIS_001` | `Insufficient data for analysis` | データ不足 | 期間の延長 |
| `ANALYSIS_002` | `Invalid parameter: {param}` | 無効パラメータ | パラメータの確認 |
| `ANALYSIS_003` | `Calculation failed: {indicator}` | 計算エラー | データの確認 |

### 予測エラー

| エラーコード | メッセージ | 原因 | 解決策 |
|-------------|-----------|------|-------|
| `PREDICT_001` | `Model training failed` | 訓練失敗 | データの確認 |
| `PREDICT_002` | `Prediction failed: {error}` | 予測失敗 | モデルの確認 |
| `PREDICT_003` | `Invalid date range` | 日付エラー | 日付の確認 |

### バックテストエラー

| エラーコード | メッセージ | 原因 | 解決策 |
|-------------|-----------|------|-------|
| `BACKTEST_001` | `投資日または検証日の価格データが見つかりません` | データなし | 営業日の確認 |
| `BACKTEST_002` | `検証日は投資日より後である必要があります` | 日付順序エラー | 日付の修正 |
| `BACKTEST_003` | `Training data insufficient` | 訓練データ不足 | 期間の延長 |

---

## 使用例

### 基本的なワークフロー

```bash
# 1. 設定初期化
uv run python -m stock_analyzer.cli.main config --init

# 2. 銘柄の基本情報確認
uv run python -m stock_analyzer.cli.main get-data AAPL --info

# 3. テクニカル分析
uv run python -m stock_analyzer.cli.main analyze AAPL --signals

# 4. 未来予測
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 30

# 5. 過去精度検証
uv run python -m stock_analyzer.cli.main backtest AAPL \
  --investment-date 2024-07-01 --validation-date 2024-08-25
```

### 複数銘柄分析

```bash
# 1. ウォッチリスト作成
uv run python -m stock_analyzer.cli.main config \
  --add-watchlist "my-stocks" \
  --watchlist-symbols "AAPL,MSFT,GOOGL,AMZN,NVDA"

# 2. 比較分析
uv run python -m stock_analyzer.cli.main compare-advanced --watchlist my-stocks

# 3. ポートフォリオ最適化
uv run python -m stock_analyzer.cli.main portfolio --watchlist my-stocks \
  --investment-amount 100000 --export-csv
```

### 自動化スクリプト例

```bash
#!/bin/bash
# daily_analysis.sh - 日次分析スクリプト

SYMBOLS="AAPL MSFT GOOGL AMZN NVDA"
DATE=$(date +%Y%m%d)
OUTPUT_DIR="reports/$DATE"

mkdir -p "$OUTPUT_DIR"

# 各銘柄の予測を実行
for symbol in $SYMBOLS; do
    echo "Analyzing $symbol..."

    uv run python -m stock_analyzer.cli.main predict "$symbol" \
      --prediction-days 30 > "$OUTPUT_DIR/${symbol}_prediction.txt"
done

# ポートフォリオ分析
uv run python -m stock_analyzer.cli.main portfolio $SYMBOLS \
  --export-csv > "$OUTPUT_DIR/portfolio_analysis.txt"

echo "Analysis complete. Results saved to $OUTPUT_DIR"
```

---

このAPIリファレンスは、Stock Analyzerの全機能を網羅的に説明しています。各コマンドの詳細な使用方法やパラメータ、内部APIの仕様、データ構造などを参考に、効果的にツールをご活用ください。
