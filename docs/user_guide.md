# Stock Analyzer ユーザーガイド

## 目次

1. [概要](#概要)
2. [インストールと設定](#インストールと設定)
3. [基本コマンド](#基本コマンド)
4. [高度な機能](#高度な機能)
5. [予測とシミュレーション](#予測とシミュレーション)
6. [ポートフォリオ分析](#ポートフォリオ分析)
7. [設定管理](#設定管理)
8. [使用例とワークフロー](#使用例とワークフロー)
9. [トラブルシューティング](#トラブルシューティング)

---

## 概要

Stock Analyzerは、米国株式市場の分析を行うためのコマンドラインツールです。株価データの取得、テクニカル分析、機械学習による予測、ポートフォリオ最適化など、投資判断に必要な機能を包括的に提供します。

### 主な機能

- 📊 **株価データ取得**: Yahoo Finance APIを使用したリアルタイムデータ
- 📈 **テクニカル分析**: RSI、MACD、移動平均、ボリンジャーバンドなど
- 🤖 **機械学習予測**: LightGBMを使用した価格動向予測
- 🎯 **バックテスト**: 過去データを使った予測精度検証
- 🔮 **未来予測**: 現在データから未来の価格動向を予測
- 💼 **ポートフォリオ分析**: 複数銘柄の最適化と比較
- 📋 **CSV出力**: 分析結果の詳細レポート生成
- ⚙️ **設定管理**: ウォッチリストと分析パラメータの管理

---

## インストールと設定

### 前提条件

- Python 3.12+
- uv (Pythonパッケージマネージャー)

### セットアップ

```bash
# プロジェクトディレクトリに移動
cd /workspaces/stock-analyzer

# 依存関係をインストール
uv sync

# 設定ファイルを初期化
uv run python -m stock_analyzer.cli.main config --init
```

---

## 基本コマンド

### 1. 株価データ取得

```bash
# 基本的なデータ取得
uv run python -m stock_analyzer.cli.main get-data AAPL

# 期間を指定してデータ取得
uv run python -m stock_analyzer.cli.main get-data AAPL --period 6mo

# 会社情報も含めて表示
uv run python -m stock_analyzer.cli.main get-data AAPL --info
```

**利用可能な期間**:
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

**出力例**:
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

### 2. 複数銘柄の価格比較

```bash
# 複数銘柄の現在価格を比較
uv run python -m stock_analyzer.cli.main compare AAPL MSFT GOOGL AMZN TSLA
```

**出力例**:
```
🔍 5銘柄の価格比較:
✅ AAPL: $230.51
✅ MSFT: $416.42
✅ GOOGL: $164.74
✅ AMZN: $178.25
✅ TSLA: $238.59

💰 価格順（高い順）:
1. MSFT: $416.42
2. TSLA: $238.59
3. AAPL: $230.51
4. AMZN: $178.25
5. GOOGL: $164.74
```

### 3. テクニカル分析

```bash
# 基本的なテクニカル分析
uv run python -m stock_analyzer.cli.main analyze AAPL

# 売買シグナル分析も含める
uv run python -m stock_analyzer.cli.main analyze AAPL --signals

# 長期間のデータでより詳細な分析
uv run python -m stock_analyzer.cli.main analyze AAPL --period 1y --signals
```

**出力例**:
```
🎯 AAPL テクニカル分析結果:
期間: 2024-02-28 ～ 2024-08-28
現在価格: $230.51

📈 移動平均線:
SMA5:  $228.45
SMA20: $225.82
SMA50: $220.15

⚡ 主要指標:
RSI (14日): 58.2 (中立 ➡️)
MACD: 0.825
出来高比率: 1.15x (通常)
ボリンジャーバンド位置: 65.3% (中央付近 ➡️)

🎯 売買シグナル分析:
RSI: 中立 ➡️
MACD: 買いシグナル 📈
MA: ゴールデンクロス 📈
```

---

## 高度な機能

### 1. 詳細銘柄比較

```bash
# 複数銘柄の詳細比較分析
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL

# プリセット銘柄グループを使用
uv run python -m stock_analyzer.cli.main compare-advanced --preset tech-giants

# 結果をCSVで出力
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL --export-csv

# 投資スコアでソート
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL --sort-by investment_score
```

**出力例**:
```
📊 銘柄比較結果 (並び順: investment_score):
================================================================================
銘柄     価格($)   投資ｽｺｱ  ﾘｽｸｽｺｱ  推奨     5日予測%   30日予測%
--------------------------------------------------------------------------------
AAPL     230.51   78.5     35.2     強い買い   2.5       6.8
MSFT     416.42   72.1     28.9     買い      1.8       5.2
GOOGL    164.74   68.3     42.7     買い      1.2       4.1

🏆 投資スコア上位3銘柄:
  🥇 AAPL: スコア 78.5 - 強い買い
  🥈 MSFT: スコア 72.1 - 買い
  🥉 GOOGL: スコア 68.3 - 買い
```

### 2. ウォッチリストを使用した分析

```bash
# ウォッチリストを作成
uv run python -m stock_analyzer.cli.main config --add-watchlist "tech-favorites" --watchlist-symbols "AAPL,MSFT,GOOGL,META,NVDA"

# ウォッチリストを使って比較
uv run python -m stock_analyzer.cli.main compare-advanced --watchlist tech-favorites
```

---

## 予測とシミュレーション

### 1. 未来予測（リアルタイム予測）

```bash
# 30日後の価格動向を予測
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 30

# 60日後のリターン率を予測
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 60 --prediction-type return

# より長い訓練期間を使用
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 30 --training-months 36
```

**出力例**:
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

⚠️  注意事項:
• この予測は過去データに基づく推定です
• 市場の突発的な変動は予測できません
• 投資判断は複数の情報を総合して行ってください
• 2025-10-27に実際の結果を確認してください

🔍 2025-10-27以降に予測精度を検証するコマンド:
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2025-08-28 --validation-date 2025-10-27
```

### 2. バックテスト（過去データ検証）

```bash
# 特定期間の予測精度を検証
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2024-07-01 --validation-date 2024-08-25

# リターン率予測の検証
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2024-06-01 --validation-date 2024-07-31 --prediction-type return

# より長い訓練期間での検証
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2024-07-01 --validation-date 2024-08-25 --training-months 36
```

**出力例**:
```
📊 シミュレーション結果:
============================================================
銘柄: AAPL
投資期間: 2024-07-01 → 2024-08-25
予測期間: 55日
予測タイプ: direction

🔮 予測結果:
予測方向: 上昇
信頼度: 21.3%

📈 実際の結果:
投資価格: $215.50
検証価格: $225.79
実際リターン: 4.78%
実際方向: 上昇

🎯 評価結果:
予測精度: ✅ 正解
信頼性スコア: 84.3/100
結果: 予測: 上昇 (信頼度: 21.3%) | 実際: 上昇 | 正解: ○

💡 推奨: 🔥 このモデルは高精度です。実戦投入を検討できます。
```

---

## ポートフォリオ分析

### 1. 基本的なポートフォリオ分析

```bash
# 複数銘柄でポートフォリオを作成
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL AMZN TSLA

# 投資金額と最大銘柄数を指定
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL --investment-amount 100000 --max-stocks 3

# リスク許容度を指定
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL --risk-tolerance 0.5
```

**出力例**:
```
🎯 ポートフォリオ分析結果:
==================================================
分析銘柄数: 5
選択銘柄数: 3
総投資金額: $100,000
ポートフォリオリスク: 中程度

💰 推奨ポートフォリオ構成:
  AAPL: 40.0% ($40,000) - 強い買い (スコア: 78.5)
  MSFT: 35.0% ($35,000) - 買い (スコア: 72.1)
  GOOGL: 25.0% ($25,000) - 買い (スコア: 68.3)

🎯 推奨アクション: 積極的投資
リスク評価: 適度なリスクで良好なリターン期待
理由:
  • 高い投資スコアの銘柄が中心
  • リスクの分散が適切
  • 成長期待の高いテック株中心
```

### 2. プリセットとウォッチリストを使用

```bash
# 利用可能なプリセット一覧を表示
uv run python -m stock_analyzer.cli.main portfolio --list-presets

# プリセットを使用したポートフォリオ分析
uv run python -m stock_analyzer.cli.main portfolio --preset tech-giants --investment-amount 50000

# ウォッチリストを使用
uv run python -m stock_analyzer.cli.main portfolio --watchlist my-stocks
```

### 3. CSV出力とレポート生成

```bash
# 詳細レポートをCSVで出力
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL --export-csv

# ファイルから銘柄リストを読み込み
uv run python -m stock_analyzer.cli.main portfolio --symbols-file stocks.csv --export-csv
```

---

## 設定管理

### 1. 基本設定

```bash
# 現在の設定を表示
uv run python -m stock_analyzer.cli.main config --show

# デフォルト設定ファイルを作成
uv run python -m stock_analyzer.cli.main config --init

# 設定値を変更
uv run python -m stock_analyzer.cli.main config --set general.default_period --value 1y
uv run python -m stock_analyzer.cli.main config --set general.default_investment_amount --value 50000
```

### 2. ウォッチリスト管理

```bash
# ウォッチリストを追加
uv run python -m stock_analyzer.cli.main config --add-watchlist "tech-stocks" --watchlist-symbols "AAPL,MSFT,GOOGL,META,NVDA"

# 利用可能なウォッチリストを表示
uv run python -m stock_analyzer.cli.main portfolio --list-watchlists
```

### 3. 設定例

```yaml
general:
  default_period: "1y"
  default_investment_amount: 100000.0
  default_max_stocks: 10
  default_risk_tolerance: 0.3
  auto_export_csv: false

analysis:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9

portfolio:
  min_allocation: 0.05
  max_allocation: 0.4
  rebalance_threshold: 0.05

watchlists:
  tech-giants:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
    - "META"
    - "NVDA"
  dividend-stocks:
    - "JNJ"
    - "PG"
    - "KO"
```

---

## 使用例とワークフロー

### 1. 日常的な銘柄チェック

```bash
# 1. ウォッチリストの銘柄を一括チェック
uv run python -m stock_analyzer.cli.main compare --watchlist daily-check

# 2. 気になる銘柄の詳細分析
uv run python -m stock_analyzer.cli.main analyze AAPL --signals

# 3. 未来予測で投資タイミングを判断
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 30
```

### 2. 新規投資の検討

```bash
# 1. 候補銘柄の比較分析
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL AMZN --sort-by investment_score

# 2. ポートフォリオ最適化
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL --investment-amount 100000 --export-csv

# 3. バックテストで過去の性能確認
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2024-01-01 --validation-date 2024-06-30
```

### 3. 予測精度の検証

```bash
# 1. 予測を実行して記録
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 30 > prediction_log.txt

# 2. 30日後に精度を検証
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2025-08-28 --validation-date 2025-09-27

# 3. 結果を分析してモデル改善
```

### 4. 定期的なポートフォリオ見直し

```bash
# 1. 月次ポートフォリオ分析
uv run python -m stock_analyzer.cli.main portfolio --watchlist my-portfolio --export-csv

# 2. 各銘柄の最新分析
uv run python -m stock_analyzer.cli.main compare-advanced --watchlist my-portfolio --export-csv

# 3. リバランス判断
# CSVファイルを確認して配分調整を検討
```

---

## トラブルシューティング

### よくある問題と解決策

#### 1. データ取得エラー

**問題**: `❌ エラー: Failed to fetch data for AAPL`

**解決策**:
```bash
# ネットワーク接続を確認
ping yahoo.com

# シンボルが正しいか確認
uv run python -m stock_analyzer.cli.main get-data AAPL --info

# 期間を短くして試行
uv run python -m stock_analyzer.cli.main get-data AAPL --period 1mo
```

#### 2. 予測エラー

**問題**: `❌ エラー: 投資日または検証日の価格データが見つかりません`

**解決策**:
- 投資日と検証日が営業日かチェック
- 未来日付の場合は`predict`コマンドを使用
- 期間が長すぎる場合は短縮

```bash
# 正しい使用例
uv run python -m stock_analyzer.cli.main backtest AAPL --investment-date 2024-07-01 --validation-date 2024-08-25
uv run python -m stock_analyzer.cli.main predict AAPL --prediction-days 30
```

#### 3. 設定ファイルエラー

**問題**: 設定が反映されない

**解決策**:
```bash
# 設定を初期化
uv run python -m stock_analyzer.cli.main config --init

# 設定内容を確認
uv run python -m stock_analyzer.cli.main config --show

# 設定ファイルの場所を確認
echo $HOME/.config/stock_analyzer/config.yaml
```

#### 4. パフォーマンスの問題

**問題**: 分析が遅い

**解決策**:
- 期間を短縮（`--period 6mo`など）
- 銘柄数を減らす
- 訓練期間を短縮（`--training-months 12`など）

### ログレベルの調整

```bash
# デバッグ情報を表示
export LOG_LEVEL=DEBUG
uv run python -m stock_analyzer.cli.main analyze AAPL

# エラーのみ表示
export LOG_LEVEL=ERROR
uv run python -m stock_analyzer.cli.main portfolio AAPL MSFT GOOGL
```

---

## サポート情報

- **プロジェクトURL**: Stock Analyzer GitHub Repository
- **Python要件**: Python 3.12+
- **依存パッケージ**: pandas, numpy, lightgbm, click, structlog
- **データソース**: Yahoo Finance API

### パフォーマンス目安

| 操作 | 1銘柄 | 5銘柄 | 10銘柄 |
|------|-------|-------|--------|
| データ取得 | ~2秒 | ~8秒 | ~15秒 |
| テクニカル分析 | ~3秒 | ~12秒 | ~25秒 |
| 機械学習予測 | ~5秒 | ~20秒 | ~40秒 |
| ポートフォリオ分析 | N/A | ~25秒 | ~45秒 |

### 推奨設定

- **通常使用**: 期間6mo-1y、訓練期間24ヶ月
- **高精度分析**: 期間1y-2y、訓練期間36ヶ月
- **高速分析**: 期間3mo-6mo、訓練期間12ヶ月

---

このドキュメントがStock Analyzerの効果的な活用に役立てば幸いです。ご質問やご不明な点がございましたら、お気軽にお尋ねください。
