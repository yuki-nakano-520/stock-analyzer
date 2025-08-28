---
title: CLAUDE.md
created_at: 2025-06-14
updated_at: 2025-07-14
# このプロパティは、Claude Codeが関連するドキュメントの更新を検知するために必要です。消去しないでください。
---

# Stock Analyzer

米国株の傾向分析を行うCLIツール。株価データの取得・分析・可視化を通じて投資判断をサポートします。

## 技術スタック

**Python 3.12+** | uv | Ruff | pyright | pytest + Hypothesis | pre-commit | GitHub Actions

## 主な機能

- 株価データ取得（Alpha Vantage / Yahoo Finance API）
- トレンド分析（移動平均、RSI、MACD等）
- バックテスト機能
- レポート生成（CLI出力 / CSV / JSON）
- 銘柄スクリーニング

## プロジェクト構造

```
stock-analyzer/
├── .github/                     # GitHub Actions & templates
├── template/                    # 実装参考例（変更禁止）
├── src/
│   └── stock_analyzer/          # メインパッケージ
│       ├── __init__.py
│       ├── cli/                 # CLI関連
│       │   ├── __init__.py
│       │   └── main.py          # CLIエントリーポイント
│       ├── data/                # データ取得・管理
│       │   ├── __init__.py
│       │   ├── fetchers.py      # API経由でのデータ取得
│       │   └── storage.py       # データ永続化
│       ├── analysis/            # 分析ロジック
│       │   ├── __init__.py
│       │   ├── indicators.py    # テクニカル指標
│       │   ├── trends.py        # トレンド分析
│       │   └── backtesting.py   # バックテスト
│       ├── reports/             # レポート生成
│       │   ├── __init__.py
│       │   ├── formatters.py    # 出力フォーマット
│       │   └── generators.py    # レポート生成
│       ├── utils/               # 共通ユーティリティ
│       └── types.py             # 型定義
├── tests/                       # テストコード
├── docs/                        # ドキュメント
└── scripts/                     # ヘルパースクリプト
```

## 実装時の必須要件

### 0. 開発環境
- **パッケージ管理**: uvで環境を統一管理。Pythonコマンドは必ず `uv run` を前置
- **依存関係追加**: `uv add` (通常) / `uv add --dev` (開発用)
- **GitHub操作**: `make pr`/`make issue` または `gh`コマンド
- **品質保証**: pre-commitフック設定済み。`make check-all` で包括的チェック
- **開発支援**: Makefileに開発効率化コマンド集約。`make help` で一覧表示

### 1-5. 実装フロー
1. **品質**: format→lint→typecheck→test
2. **テスト**: 新機能::TDD必須
3. **ロギング**: 全コード::ログ必須
4. **性能**: 重い処理→プロファイル
5. **段階的**: Protocol»テスト»実装»最適化

### 6. エビデンスベース開発

**禁止語**: best, optimal, faster, always, never, perfect
**推奨語**: measured, documented, approximately, typically

**証拠要件**:
- **性能**: "measured Xms" | "reduces X%"
- **品質**: "coverage X%" | "complexity Y"
- **セキュリティ**: "scan detected X"
- **信頼性**: "uptime X%" | "error rate Y%"

### 7. 効率化テクニック

#### コミュニケーション記法
```yaml
→: "処理フロー"      # analyze→fix→test
|: "選択/区切り"     # option1|option2
&: "並列/結合"       # task1 & task2
::: "定義"           # variable :: value
»: "シーケンス"      # step1 » step2
@: "参照/場所"       # @file:line
```

#### 実行パターン
- **並列**: 依存なし&競合なし&順序不問 → 複数ファイル読込|独立テスト|並列ビルド
- **バッチ**: 同種操作&共通リソース → 一括フォーマット|インポート修正|バッチテスト
- **逐次**: 依存あり|状態変更|トランザクション → DBマイグレ|段階的リファクタ|依存インストール

#### エラーリカバリー
- **リトライ**: max3回 & 指数バックオフ
- **フォールバック**: 高速→確実
- **状態復元**: チェックポイント»ロールバック|正常状態»再開|失敗のみ»再実行

#### 建設的フィードバックの提供

**フィードバックのトリガー**:
- 非効率なアプローチを検出
- セキュリティリスクを発見
- 過剰設計を認識
- 不適切な実践を確認

**アプローチ**:
- 直接的な表現 > 婉曲的な表現
- エビデンスベースの代替案 > 単なる批判
- 例: "より効率的な方法: X" | "セキュリティリスク: SQLインジェクション"

## template/ディレクトリの活用

実装前に必ず参照:
- **クラス/関数**: @template/src/template_package/core/example.py (型ヒント|docstring|エラー処理)
- **型定義**: @template/src/template_package/types.py
- **ユーティリティ**: @template/src/template_package/utils/helpers.py
- **テスト**: @template/tests/{unit|property|integration}/
- **フィクスチャ**: @template/tests/conftest.py
- **ロギング**: @template/src/template_package/utils/logging_config.py
実装時: template/内の類似例確認»パターン踏襲»プロジェクト調整
注: template/は変更&削除禁止

## よく使うコマンド

```bash
# 初期セットアップ
make setup              # 依存関係インストール + pre-commitフック設定

# 基本
make help               # コマンド一覧
make check-all          # 全チェック実行
make clean              # キャッシュ削除

# 品質チェック
make format             # フォーマット
make lint               # リント
make typecheck          # 型チェック
make test               # テスト実行
make test-cov           # カバレッジ付きテスト

# GitHub操作
make pr TITLE="x" BODY="y" [LABEL="z"]     # PR作成
make issue TITLE="x" BODY="y"              # Issue作成
# 注: BODYにはファイルパス指定可能 (例: BODY="/tmp/pr_body.md")
# 注: 存在しないラベルは自動作成される

# 依存関係
uv add package_name               # 通常パッケージ
uv add --dev package_name         # 開発用パッケージ
make sync                         # 全依存関係を同期
uv lock --upgrade                 # 依存関係を更新
```

## Git規則

**ブランチ**: feature/ | fix/ | refactor/ | docs/ | test/
**ラベル**: enhancement | bug | refactor | documentation | test

## コーディング規約

### ディレクトリ構成

パッケージとテストは `template/` 内の構造を踏襲、コアロジックは必ず `src/project_name` 内に配置

```
src/
├── project_name/
│   ├── core/
│   ├── utils/
│   ├── __init__.py
│   └── ...
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   └── conftest.py
├── docs/
...
```

### Python コーディングスタイル
- 型ヒント: Python 3.12+スタイル必須（pyright + PEP 695）
- Docstring: NumPy形式
- 命名: クラス(PascalCase)、関数(snake_case)、定数(UPPER_SNAKE)、プライベート(_prefix)
- ベストプラクティス: @template/src/template_package/types.py

### エラーメッセージ

1. **具体的**: "Invalid input" → "Expected positive integer, got {count}"
2. **コンテキスト付き**: "Failed to process {source_file}: {e}"
3. **解決策を提示**: "Not found. Create by: python -m {__package__}.init"

### アンカーコメント
```python
# AIDEV-NOTE: 説明
# AIDEV-TODO: 課題
# AIDEV-QUESTION: 疑問
```

## テスト戦略（TDD）

t-wada流のテスト駆動開発（TDD）を徹底

### サイクル
🔴 Red » 🟢 Green » 🔵 Refactor

### 手順
1. TODO作成
2. 失敗テスト
3. 最小実装（仮実装OK）
4. リファクタ

### 原則
- 小さなステップで進める
- 三角測量で一般化
- 不安な部分から着手
- テストリストを常に更新

#### 三角測量の例
```python
# 1. 仮実装: return 5
assert add(2, 3) == 5

# 2. 一般化: return a + b
assert add(10, 20) == 30

# 3. エッジケース確認
assert add(-1, -2) == -3
```

#### 注意点
- 1test::1behavior
- Red»Greenでコミット
- 日本語テスト名推奨
- リファクタ: 重複|可読性|SOLID違反時

### テスト種別
1. **単体**: 基本動作 `template/tests/unit/`
2. **プロパティ**: Hypothesis自動生成 `template/tests/property/`
3. **統合**: 連携テスト `template/tests/integration/`

### テスト命名
`test_[正常系|異常系|エッジケース]_条件で結果()`

## ロギング

### 必須要件
1. モジュール冒頭::ロガー定義
2. 関数開始&終了::ログ出力
3. エラー時::exc_info=True
4. レベル: DEBUG|INFO|WARNING|ERROR

ベストプラクティス: @template/src/template_package/utils/logging_config.py & @template/src/template_package/core/example.py

### 設定
```python
setup_logging(level="INFO")
# または export LOG_LEVEL=INFO
```

### テスト時の設定
```bash
# 環境変数でテスト時のログレベル制御
export TEST_LOG_LEVEL=INFO  # デフォルトはDEBUG
```

```python
# 個別テストでログレベル変更
def test_カスタムログレベル(set_test_log_level):
    set_test_log_level("WARNING")
    # テスト実行
```

## パフォーマンス測定

```python
from template_package.utils.profiling import profile, timeit, Timer

@profile  # 詳細プロファイル
@timeit   # 実行時間計測
def func(): ...

with Timer("operation"):  # ブロック計測
    ...
```

ベストプラクティス: @template/src/template_package/utils/profiling.py

## 更新トリガー

- 仕様/依存関係/構造/規約の変更時
- 同一質問2回以上 → FAQ追加
- エラーパターン2回以上 → トラブルシューティング追加

## トラブルシューティング/FAQ

適宜更新

## Stock Analyzer 固有ガイド

### 株価データ取得
- **無料API制限**: Alpha Vantage（500/日）/ Yahoo Finance（レート制限あり）
- **データ形式**: OHLCV（始値・高値・安値・終値・出来高）
- **エラー処理**: API制限→フォールバック→キャッシュ利用

### テクニカル分析
- **移動平均**: SMA/EMA（5,20,50,200日）
- **モメンタム**: RSI（14日）、MACD（12,26,9）
- **ボラティリティ**: Bollinger Bands（20日,2σ）
- **トレンド**: ADX、Parabolic SAR

### バックテスト設計
- **期間**: 最低1年、推奨3-5年
- **手数料**: 売買コスト0.25%想定
- **指標**: 勝率、最大ドローダウン、シャープレシオ
- **検証**: Walk-forward analysis推奨

### CLI使用例
```bash
# 基本分析
uv run stock-analyzer analyze AAPL --period 1y

# 複数銘柄スクリーニング
uv run stock-analyzer screen --rsi-oversold --volume-spike

# バックテスト
uv run stock-analyzer backtest --strategy sma_crossover --period 3y
```

## カスタムガイド

`docs/`に追加可能。追加時はCLAUDE.mdに概要記載必須。
