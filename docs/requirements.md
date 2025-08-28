# Stock Analyzer 要件定義書

## 1. プロジェクト概要

### 1.1 目的
LightGBM機械学習を活用した米国株式市場の予測分析CLIツールを開発し、分散投資における個人投資家の投資判断をサポートする。

### 1.2 対象ユーザー
- 分散投資を行う個人投資家（初級〜中級）
- **Python初心者〜中級者**（学習しながら実装）
- 機械学習による株価予測に興味のあるプログラマー
- データ分析学習者

### 1.3 開発方針
- **段階的学習アプローチ**：Python初心者が無理なく学べる構成
- **実用的なプロジェクト**：実際に使える株価予測ツールの開発
- LightGBMによる機械学習予測
- TDD（テスト駆動開発）によるエビデンスベース開発
- 無料APIの活用で誰でも利用可能
- CLIによるシンプルな操作性
- 複数銘柄対応による分散投資支援
- 拡張性を考慮した設計
- **豊富なコメントと説明**：コードの理解促進

## 2. 機能要件

### 2.1 コア機能

#### 2.1.1 株価データ取得機能
**優先度**: 高

**概要**: 指定した銘柄の株価データを取得・キャッシュする

**詳細要件**:
- Yahoo Finance APIから OHLCV（始値・高値・安値・終値・出来高）データを取得
- 取得期間：1日〜5年（デフォルト1年）
- データキャッシュ機能（24時間有効）
- API制限対応（リトライ & レート制限）
- 複数銘柄の一括取得

**入力**:
- 銘柄シンボル（例：AAPL, MSFT）
- 期間指定（1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y）

**出力**:
- OHLCV データ（JSON/CSV形式）
- 取得ステータス情報

#### 2.1.2 機械学習予測機能
**優先度**: 最高

**概要**: LightGBMを使用した株価予測による投資判断支援

**予測対象**:
- 翌日の株価（1日後）
- 短期予測（1週間後）
- 中期予測（1ヶ月後、3ヶ月後）
- 長期予測（6ヶ月後）

**特徴量（自動選択）**:
- **基本価格データ**: OHLCV、価格変動率
- **テクニカル指標**: SMA(5,20,50,200日)、EMA(12,26日)、RSI(14日)、MACD(12,26,9日)、ボリンジャーバンド(20日,2σ)
- **ボラティリティ指標**: ATR、ボラティリティレート
- **出来高指標**: 出来高変動率、出来高移動平均比
- **価格パターン**: 高値/安値更新、レジスタンス/サポート

**予測モデル**: LightGBM（勾配ブースティング）
**投資判断基準**: 70%以上の上昇率予測時に投資推奨
**信頼度算出**: 予測精度に基づくスコア（0-1）

**入力**: 銘柄シンボル + 予測期間
**出力**: 予測価格 + 期待リターン + 投資スコア + 信頼度

#### 2.1.3 複数銘柄分析・比較機能
**優先度**: 高

**概要**: 分散投資のための複数銘柄の一括分析・相対評価

**分析機能**:
- 複数銘柄の同時予測分析
- 期待リターンによるランキング
- 投資スコアによる推奨度判定
- リスク・リターン分析

**銘柄指定方法**:
- 手動指定: `--symbols AAPL,MSFT,GOOGL`
- 設定ファイル: `--watchlist stocks.txt`
- 同時分析上限: 設定ファイルで指定可能（デフォルト50銘柄）

**出力**: CSV形式での比較レポート（日本語カラム名）

#### 2.1.4 投資ポートフォリオ最適化機能（将来拡張）
**優先度**: 低

**概要**: 複数銘柄の最適な投資配分を提案

**最適化ロジック**:
- リスク・リターン最適化
- 相関を考慮した分散投資提案
- 投資額配分アドバイス

### 2.2 レポート機能

#### 2.2.1 予測レポート生成
**優先度**: 最高

**概要**: 機械学習予測結果を投資判断に適した形式で出力

**出力形式**:
- CSV形式（メイン、日本語カラム名）
- JSON形式（API連携用）
- コンソール表示（概要表示）

**CSV出力項目**:
- 銘柄コード、銘柄名、現在価格、予測価格(各期間)
- 期待リターン(%)、投資スコア、信頼度
- 投資推奨度、リスク評価、最終更新日時

**レポート内容**:
- 基本情報（銘柄名、現在価格、変動率）
- 予測結果（各期間の予測価格・リターン）
- 機械学習スコア（信頼度、投資推奨度）
- リスク指標（ボラティリティ、最大ドローダウン予測）

#### 2.2.2 予測精度検証・バックテスト機能
**優先度**: 中

**概要**: LightGBM予測モデルの精度検証と投資戦略の過去データでの検証

**予測精度検証**:
- 過去データでの予測精度測定（RMSE, MAE, 方向性適中率）
- 期間別精度分析（短期vs長期）
- 銘柄別予測精度ランキング

**投資戦略バックテスト**:
- ML予測ベース投資戦略
- 70%リターン基準での投資判断検証

**検証指標**:
- 総リターン、勝率、最大ドローダウン、シャープレシオ
- 予測精度指標（RMSE, MAE, 方向性適中率）

## 3. 非機能要件

### 3.1 性能要件
- 単一銘柄予測: 10秒以内（初回学習込み）
- 10銘柄一括分析: 60秒以内
- 50銘柄大量分析: 300秒以内
- API制限遵守（Yahoo Finance: 2000リクエスト/時間）
- モデル学習: バックグラウンド実行対応
- キャッシュ活用で2回目以降高速化

### 3.2 可用性要件
- APIエラー時のフォールバック機能
- 学習済みモデルの永続化とキャッシュ
- オフライン予測（学習済みモデル使用）
- エラーの適切な処理と表示
- 部分的失敗時の継続処理

### 3.3 操作性要件
- 直感的なCLIコマンド体系
- 進捗表示（学習・予測プロセス）
- 日本語での結果表示
- 設定ファイルによるカスタマイズ
- ヘルプ機能の充実

### 3.4 拡張性要件
- 新しい特徴量の追加容易性
- 他の機械学習モデル対応（XGBoost等）
- 複数データソース対応の基盤
- 予測期間の動的設定
- プラグイン機構（将来）

## 4. システム構成

### 4.1 アーキテクチャ
```
CLI Layer (click)
    ↓
Business Logic Layer
    ├── Data Fetcher (yfinance)
    ├── Feature Engineer (ta + 自社実装)
    ├── ML Predictor (LightGBM)
    ├── Portfolio Analyzer
    └── Report Generator
    ↓
Data Layer
    ├── Cache Storage (requests-cache)
    ├── Model Storage (joblib)
    └── Configuration (pydantic)
```

### 4.2 主要コンポーネント

| コンポーネント | 責務 | 実装 |
|---|---|---|
| CLI | ユーザーインターフェース | click |
| DataFetcher | 株価データ取得 | yfinance + httpx |
| FeatureEngineer | 特徴量生成 | ta + pandas + 自社実装 |
| MLPredictor | 機械学習予測 | LightGBM + scikit-learn |
| PortfolioAnalyzer | 複数銘柄分析 | pandas + 自社実装 |
| ReportGenerator | レポート生成 | pandas + 自社実装 |
| Cache | データ・モデルキャッシュ | requests-cache + joblib |
| Config | 設定管理 | pydantic + YAML |

## 5. CLIコマンド設計

### 5.1 基本コマンド

```bash
# 単一銘柄予測
stock-analyzer predict AAPL
stock-analyzer predict AAPL --days 30,90,180 --format csv

# 複数銘柄分析（手動指定）
stock-analyzer predict AAPL,MSFT,GOOGL --days 30 --output results.csv

# ウォッチリスト分析（設定ファイル）
stock-analyzer predict --watchlist my_stocks.txt --days 90
stock-analyzer predict --watchlist sp500_top50.txt --format csv

# 投資推奨銘柄抽出
stock-analyzer recommend --min-return 70 --max-stocks 10 --days 90

# モデル管理
stock-analyzer train --retrain  # モデル再学習
stock-analyzer model --status   # モデル情報表示

# キャッシュ管理
stock-analyzer cache --clear
stock-analyzer cache --status
```

### 5.2 共通オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--days` | 予測期間（1,7,30,90,180） | 30 |
| `--format` | 出力形式（console/json/csv） | console |
| `--output` | 出力ファイル | stdout |
| `--watchlist` | 銘柄リストファイル | - |
| `--cache` | キャッシュ利用 | True |
| `--min-return` | 最小期待リターン(%) | 50 |
| `--max-stocks` | 最大分析銘柄数 | 設定ファイル |
| `--confidence` | 最小信頼度 | 0.6 |
| `--verbose` | 詳細出力 | False |

## 6. データモデル

### 6.1 株価データ
```python
class StockData(TypedDict):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float  # 調整後終値
```

### 6.2 特徴量データ
```python
class FeatureData(TypedDict):
    # 基本価格指標
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volatility_20d: float

    # テクニカル指標
    sma_5: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    rsi_14: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float

    # 出来高指標
    volume_sma_20: float
    volume_ratio: float

    # 価格パターン
    is_near_high: bool
    is_near_low: bool
    trend_strength: float
```

### 6.3 予測結果
```python
class PredictionResult(TypedDict):
    symbol: str
    company_name: str
    current_price: float
    predicted_prices: dict[int, float]  # {days: price}
    expected_returns: dict[int, float]  # {days: return_pct}
    investment_score: float  # 0-1
    confidence_score: float  # 0-1
    risk_level: Literal["低", "中", "高"]
    recommendation: Literal["強く推奨", "推奨", "中立", "非推奨"]
    last_updated: str
```

### 6.4 CSV出力形式
```csv
銘柄コード,銘柄名,現在価格,予測価格_1日,予測価格_7日,予測価格_30日,予測価格_90日,予測価格_180日,期待リターン_30日,期待リターン_90日,期待リターン_180日,投資スコア,信頼度,推奨度,リスクレベル,最終更新
AAPL,Apple Inc,150.25,152.10,158.75,195.30,210.50,225.80,30.0%,40.1%,50.3%,0.85,0.78,強く推奨,中,2024-01-15 10:30:00
```

## 7. 開発フェーズ

### Phase 1: 学習フェーズ (v0.1.0) - Python基礎 + データ取得
**学習目標**: Python基本構文、ライブラリ使用、API連携
- [ ] プロジェクト構造の理解・作成
- [ ] 仮想環境・依存関係管理（uv使用）
- [ ] 基本的なデータ取得機能（yfinance）
- [ ] pandasでのデータ操作基礎
- [ ] 簡単なCLI作成（click）
- [ ] エラーハンドリング基礎
- [ ] **学習メモ**: 各段階で学んだことを記録

### Phase 2: 分析基礎 (v0.2.0) - データ分析・可視化
**学習目標**: データ分析、テクニカル指標、機械学習基礎
- [ ] テクニカル指標計算（SMA、RSI等）
- [ ] データの前処理・特徴量エンジニアリング
- [ ] 基本的な可視化（matplotlib/seaborn）
- [ ] scikit-learnでの機械学習入門
- [ ] LightGBM基本使用法
- [ ] モデル評価指標の理解
- [ ] **実装チェックリスト**: 各機能の動作確認

### Phase 3: 予測モデル (v0.3.0) - 本格的な機械学習実装
**学習目標**: モデル構築、予測、評価
- [ ] 複数期間予測モデルの実装
- [ ] ハイパーパラメータチューニング
- [ ] クロスバリデーション
- [ ] 予測精度の評価・改善
- [ ] モデルの保存・読み込み
- [ ] **コードレビュー**: 実装品質の確認

### Phase 4: 高度機能 (v1.0.0) - 実用化
**学習目標**: プロダクト品質、保守性、拡張性
- [ ] 複数銘柄対応・ポートフォリオ分析
- [ ] 設定ファイル・ウォッチリスト機能
- [ ] バックテスト・検証機能
- [ ] エラーハンドリング強化
- [ ] ドキュメント整備
- [ ] **デプロイ準備**: パッケージング、配布

## 7.1 学習支援要素

### 7.1.1 段階的実装アプローチ
- **Baby Steps**: 小さな機能から段階的に実装
- **動作確認**: 各段階で必ず動くものを作成
- **リファクタリング**: 動いてから改善

### 7.1.2 実装時の学習サポート
- **詳細コメント**: コードの意図と動作を説明
- **型ヒント**: Python3.12+の型システム活用
- **テスト**: 動作確認とデバッグ手法
- **ログ**: 処理の可視化とトラブルシューティング

### 7.1.3 困った時のサポート
- **実装代行**: 複雑な部分は支援可能
- **コードレビュー**: 実装品質の確認
- **デバッグ支援**: エラー解決のサポート
- **学習リソース**: 参考資料の提供

## 8. リスク・制約

### 8.1 技術リスク
- Yahoo Finance API の利用制限・データ品質
- 機械学習モデルの予測精度限界
- 大量銘柄処理時のパフォーマンス
- モデル学習データの過学習リスク

### 8.2 ビジネスリスク
- 金融データの免責事項（投資は自己責任）
- 予測結果による損失への責任範囲
- 市場急変時の予測精度低下

### 8.3 制約事項
- 無料API利用による制限（リクエスト数・頻度）
- リアルタイムデータ非対応（15-20分遅延）
- 米国株のみ対応（日本株・他国株除外）
- ファンダメンタル分析データの限界

## 9. 成功指標

### 9.1 技術指標
- テストカバレッジ > 90%
- 型チェックエラー 0件
- 単一銘柄予測 < 10秒（初回学習込み）
- 10銘柄一括予測 < 60秒

### 9.2 品質指標
- API エラー率 < 1%
- 予測方向性適中率 > 60%（バックテスト）
- ドキュメント完成度 > 95%

### 9.3 ユーザビリティ指標
- CLI操作の直感性（設定なしで基本機能動作）
- エラーメッセージの分かりやすさ
- CSV出力の投資判断有用性

---

**作成日**: 2025-01-15
**更新日**: 2025-08-28（LightGBM機械学習対応）
**承認者**: プロジェクトオーナー

## 10. 技術仕様追加事項

### 10.1 必要なライブラリ
```toml
[dependencies]
# 機械学習
lightgbm = "^4.1.0"
scikit-learn = "^1.3.0"
pandas = "^2.1.0"
numpy = "^1.25.0"

# データ取得
yfinance = "^0.2.24"
httpx = "^0.25.0"
requests-cache = "^1.1.1"

# テクニカル分析
ta = "^0.10.2"

# CLI・設定
click = "^8.1.7"
pydantic = "^2.4.0"
pydantic-settings = "^2.0.0"

# ユーティリティ
joblib = "^1.3.0"  # モデル保存
tqdm = "^4.66.0"   # 進捗表示
```

### 10.2 設定ファイル例
```yaml
# config.yaml
analyzer:
  max_stocks: 50
  cache_days: 7
  model_retrain_days: 30

prediction:
  default_days: [30, 90, 180]
  min_confidence: 0.6
  min_return_threshold: 50.0

data:
  yahoo_finance:
    timeout: 30
    retries: 3

output:
  csv_encoding: "utf-8"
  date_format: "%Y-%m-%d %H:%M:%S"
```

## 11. 実装例とサンプルコード

### 11.1 Phase 1 実装例：基本的なデータ取得

```python
# src/stock_analyzer/data/fetchers.py
"""株価データを取得するモジュール（初心者向け実装例）"""
import yfinance as yf
import pandas as pd
from typing import Dict, Any
import logging

# ロガーの設定（ログで処理の流れを確認）
logger = logging.getLogger(__name__)

def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    株価データを取得する関数

    Args:
        symbol (str): 株式シンボル（例：'AAPL'）
        period (str): 取得期間（例：'1y', '6m', '1m'）

    Returns:
        pd.DataFrame: 株価データ（日付、OHLCV）

    Example:
        >>> data = get_stock_data('AAPL', '6m')
        >>> print(data.head())
    """
    try:
        logger.info(f"株価データを取得開始: {symbol}, 期間: {period}")

        # yfinanceを使ってデータ取得
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)

        # データが空でないかチェック
        if data.empty:
            logger.error(f"データが取得できませんでした: {symbol}")
            raise ValueError(f"No data found for symbol: {symbol}")

        logger.info(f"データ取得成功: {len(data)}件のデータ")
        return data

    except Exception as e:
        logger.error(f"データ取得エラー: {symbol}, エラー: {e}")
        raise

# 使用例
if __name__ == "__main__":
    # 基本的な使い方
    apple_data = get_stock_data("AAPL", "6m")
    print("Apple株価データ:")
    print(apple_data.head())

    print(f"\nデータ件数: {len(apple_data)}")
    print(f"最新価格: {apple_data['Close'].iloc[-1]:.2f}")
```

### 11.2 Phase 2 実装例：テクニカル指標計算

```python
# src/stock_analyzer/analysis/indicators.py
"""テクニカル指標を計算するモジュール（学習用）"""
import pandas as pd
import ta  # テクニカル分析ライブラリ
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    単純移動平均（SMA: Simple Moving Average）を計算

    Args:
        data (pd.DataFrame): 株価データ
        window (int): 期間（例：20日移動平均なら20）

    Returns:
        pd.Series: SMA値
    """
    logger.debug(f"SMA計算開始: 期間={window}日")
    sma = data['Close'].rolling(window=window).mean()
    logger.debug(f"SMA計算完了: {len(sma)}件")
    return sma

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    RSI（Relative Strength Index）を計算
    値が70以上で買われすぎ、30以下で売られすぎとされる

    Args:
        data (pd.DataFrame): 株価データ
        window (int): 計算期間（通常14日）

    Returns:
        pd.Series: RSI値（0-100の範囲）
    """
    logger.debug(f"RSI計算開始: 期間={window}日")
    rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
    logger.debug(f"RSI計算完了: {len(rsi)}件")
    return rsi

def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """
    すべてのテクニカル指標をまとめて計算

    Args:
        data (pd.DataFrame): 株価データ

    Returns:
        Dict[str, Any]: 各指標の値を含む辞書
    """
    logger.info("全テクニカル指標の計算開始")

    indicators = {}

    # 移動平均線
    indicators['sma_5'] = calculate_sma(data, 5)
    indicators['sma_20'] = calculate_sma(data, 20)
    indicators['sma_50'] = calculate_sma(data, 50)

    # RSI
    indicators['rsi'] = calculate_rsi(data, 14)

    # 最新値を取得（予測で使用）
    latest_values = {}
    for name, series in indicators.items():
        latest_values[name] = series.iloc[-1]
        logger.debug(f"{name}の最新値: {latest_values[name]:.2f}")

    logger.info("全テクニカル指標の計算完了")
    return latest_values

# 使用例
if __name__ == "__main__":
    from ..data.fetchers import get_stock_data

    # データ取得
    data = get_stock_data("AAPL", "6m")

    # 指標計算
    indicators = calculate_all_indicators(data)

    print("テクニカル指標:")
    for name, value in indicators.items():
        print(f"{name}: {value:.2f}")
```

### 11.3 Phase 3 実装例：機械学習モデル基礎

```python
# src/stock_analyzer/ml/predictor.py
"""機械学習による株価予測（初心者向け実装）"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StockPredictor:
    """株価予測モデルクラス（初心者でも理解しやすい構成）"""

    def __init__(self, model_path: str = "models/stock_predictor.joblib"):
        """
        予測モデルを初期化

        Args:
            model_path (str): モデル保存パス
        """
        self.model = None
        self.model_path = Path(model_path)
        self.features = ['sma_5', 'sma_20', 'rsi', 'volume_ratio', 'price_change']

        logger.info(f"予測モデル初期化: {model_path}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        機械学習用の特徴量を準備

        Args:
            data (pd.DataFrame): 株価データ

        Returns:
            pd.DataFrame: 特徴量データ
        """
        logger.debug("特徴量準備開始")

        # 特徴量を計算
        features_df = pd.DataFrame(index=data.index)

        # 移動平均
        features_df['sma_5'] = data['Close'].rolling(5).mean()
        features_df['sma_20'] = data['Close'].rolling(20).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))

        # 出来高比率
        features_df['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

        # 価格変動率
        features_df['price_change'] = data['Close'].pct_change()

        # 欠損値を削除
        features_df = features_df.dropna()

        logger.debug(f"特徴量準備完了: {len(features_df)}件, {len(features_df.columns)}特徴量")
        return features_df

    def train(self, data: pd.DataFrame, target_days: int = 5) -> Dict[str, float]:
        """
        モデルを学習

        Args:
            data (pd.DataFrame): 学習用株価データ
            target_days (int): 何日後の価格を予測するか

        Returns:
            Dict[str, float]: 学習結果の評価指標
        """
        logger.info(f"モデル学習開始: {target_days}日後予測")

        # 特徴量準備
        features = self.prepare_features(data)

        # 目的変数（target_days日後の価格）
        target = data['Close'].shift(-target_days).dropna()

        # データの長さを合わせる
        min_length = min(len(features), len(target))
        X = features.iloc[:min_length]
        y = target.iloc[:min_length]

        # 訓練・テストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        logger.info(f"学習データ: {len(X_train)}件, テストデータ: {len(X_test)}件")

        # LightGBMモデルの学習
        self.model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=-1  # ログを抑制
        )

        self.model.fit(X_train, y_train)

        # 予測・評価
        y_pred = self.model.predict(X_test)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred),
        }

        logger.info(f"学習完了 - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2_score']:.3f}")

        # モデル保存
        self.save_model()

        return metrics

    def predict(self, data: pd.DataFrame) -> float:
        """
        株価を予測

        Args:
            data (pd.DataFrame): 予測用データ

        Returns:
            float: 予測価格
        """
        if self.model is None:
            self.load_model()

        features = self.prepare_features(data)
        latest_features = features.iloc[-1:][self.features]

        prediction = self.model.predict(latest_features)[0]

        logger.info(f"予測価格: {prediction:.2f}")
        return prediction

    def save_model(self):
        """モデルを保存"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"モデル保存: {self.model_path}")

    def load_model(self):
        """モデルを読み込み"""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"モデル読み込み: {self.model_path}")
        else:
            logger.warning(f"モデルファイルが見つかりません: {self.model_path}")

# 使用例
if __name__ == "__main__":
    from ..data.fetchers import get_stock_data

    # データ取得
    data = get_stock_data("AAPL", "2y")  # 学習には長期データが必要

    # 予測モデル作成・学習
    predictor = StockPredictor()
    metrics = predictor.train(data, target_days=5)

    print(f"モデル性能: RMSE={metrics['rmse']:.2f}, R²={metrics['r2_score']:.3f}")

    # 予測実行
    prediction = predictor.predict(data)
    current_price = data['Close'].iloc[-1]

    print(f"現在価格: {current_price:.2f}")
    print(f"5日後予測: {prediction:.2f}")
    print(f"期待リターン: {((prediction / current_price - 1) * 100):.1f}%")
```

### 11.4 学習の進め方

1. **Phase 1から順番に**: 基礎から段階的に学習
2. **動かしながら理解**: 各コードを実際に動かして確認
3. **ログを活用**: 処理の流れをログで確認
4. **エラーを恐れない**: エラーが出たら一緒に解決
5. **質問は遠慮なく**: わからないことは積極的に質問

めんどくさくなったら遠慮なく実装をお任せください！
