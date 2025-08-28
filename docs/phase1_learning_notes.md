# Phase 1 学習メモ - Python基礎 + データ取得

**実装期間**: 2025-08-28
**学習目標**: Python基本構文、ライブラリ使用、API連携

## 🎯 達成したこと

### 1. プロジェクト構造の理解・作成
- ✅ `src/stock_analyzer/` ディレクトリ構造作成
- ✅ 各モジュール用ディレクトリ（`data/`, `cli/`, `analysis/`, `ml/`, `reports/`）
- ✅ `__init__.py` ファイルでPythonパッケージ化

### 2. 仮想環境・依存関係管理（uv使用）
- ✅ `uv add` でライブラリ追加
- ✅ 機械学習ライブラリ（lightgbm, scikit-learn, joblib等）追加
- ✅ 自動的な `pyproject.toml` 更新の確認

### 3. 基本的なデータ取得機能（yfinance）
- ✅ `get_stock_data()` 関数実装
- ✅ `get_company_info()` 関数実装
- ✅ エラーハンドリングとログ出力

### 4. pandasでのデータ操作基礎
- ✅ DataFrameの基本操作（`.iloc[-1]`, `.head()`, `.tail()`）
- ✅ インデックス操作（日付形式）
- ✅ 統計情報取得（`.max()`, `.min()`, `.mean()`）

### 5. 簡単なCLI作成（click）
- ✅ `@click.command()` デコレータ使用
- ✅ 引数・オプション設定（`@click.argument()`, `@click.option()`）
- ✅ ヘルプメッセージ・バージョン表示

### 6. エラーハンドリング基礎
- ✅ `try-except` 文の使用
- ✅ カスタム例外の発生
- ✅ ログを使った詳細エラー情報

## 💡 学んだ重要なPythonコンセプト

### API仕様の確認の重要性
**問題**: yfinanceで `'6m'` ではなく `'6mo'` が正しい期間指定
**学び**: エラーメッセージをしっかり読んで、APIドキュメントを確認する習慣

### 型ヒントの活用
```python
from typing import Literal
type YFinancePeriod = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

def get_stock_data(symbol: str, period: YFinancePeriod = "1y") -> pd.DataFrame:
```
**学び**: 型ヒントで引数の制限と戻り値の型を明確にする

### 相対インポートの罠
**問題**: 直接実行時に `from ..types import` でエラー
**解決策**:
```python
try:
    from ..types import YFinancePeriod, CompanyInfo
except ImportError:
    YFinancePeriod = str  # 直接実行時は文字列として扱う
    CompanyInfo = dict  # 直接実行時は辞書として扱う
```

### ログの活用
```python
logger.info(f"株価データを取得開始: {symbol}, 期間: {period}")
logger.error(f"データ取得エラー: {symbol}, エラー: {e}")
```
**学び**: 処理の流れとエラーの原因をログで追跡できる

## 🛠️ 実装したファイル

### `src/stock_analyzer/types.py`
- 株価データ用の型定義（`StockData`, `CompanyInfo`）
- yfinance期間指定の型（`YFinancePeriod`）

### `src/stock_analyzer/data/fetchers.py`
- `get_stock_data()`: 株価データ取得
- `get_company_info()`: 会社情報取得
- 型安全性・エラーハンドリング

### `src/stock_analyzer/cli/main.py`
- `get-data` コマンド: 単一銘柄の詳細情報表示
- `compare` コマンド: 複数銘柄の価格比較
- リッチな出力（絵文字・統計情報）

## 🧪 テスト結果

### データ取得テスト
```bash
uv run python src/stock_analyzer/data/fetchers.py
# ✅ Apple株価データ (125件)、会社情報取得成功
```

### CLIテスト
```bash
uv run stock-analyzer get-data AAPL --period 1mo --info
# ✅ 期間、統計、会社情報、最新5日間の詳細表示

uv run stock-analyzer compare AAPL MSFT GOOGL TSLA
# ✅ 4銘柄の価格比較、自動ソート
```

### 型チェックテスト
```bash
uv run pyright src/stock_analyzer/data/fetchers.py
# ✅ 0 errors, 0 warnings
```

## 🤔 つまずいたポイントと解決法

### 1. yfinance期間指定エラー
**エラー**: `Period '6m' is invalid, must be one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max`
**解決**: エラーメッセージから正しい指定方法を学習、docstringに記載

### 2. 相対インポートエラー
**エラー**: `ImportError: attempted relative import with no known parent package`
**解決**: 直接実行時用のフォールバック処理を追加

### 3. 型エラー
**エラー**: `Type "dict[str, Unknown]" is not assignable to return type "CompanyInfo"`
**解決**: 明示的な型アノテーション `company_info: CompanyInfo = {...}`

## 🚀 Phase 2への準備

次は **データ分析・可視化** に進みます：
- テクニカル指標計算（SMA、RSI等）
- データの前処理・特徴量エンジニアリング
- 基本的な可視化（matplotlib/seaborn）

## 💭 感想・気づき

1. **エラーは学習の機会**: APIエラーから正しい使い方を学べた
2. **型ヒントの威力**: 型チェックで事前にエラーを発見
3. **ログの重要性**: 処理の流れが見えて、デバッグが楽になった
4. **段階的実装**: 小さく動くものから始める方法の有効性
5. **CLIの使いやすさ**: click による直感的なコマンドライン設計

Phase 1 完了！🎉
