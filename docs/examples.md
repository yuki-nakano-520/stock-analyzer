# Stock Analyzer 使用例集

## 目次

1. [基本的な使用例](#基本的な使用例)
2. [日常の投資分析ワークフロー](#日常の投資分析ワークフロー)
3. [予測と検証の活用](#予測と検証の活用)
4. [ポートフォリオ管理](#ポートフォリオ管理)
5. [高度な分析手法](#高度な分析手法)
6. [自動化スクリプト例](#自動化スクリプト例)
7. [実際の投資戦略への応用](#実際の投資戦略への応用)

---

## 基本的な使用例

### 1. 初回セットアップと基本操作

```bash
# 1. 設定ファイルを初期化
uv run python -m stock_analyzer.cli.main config --init

# 2. 設定を確認
uv run python -m stock_analyzer.cli.main config --show

# 3. 基本的な株価データ取得
uv run python -m stock_analyzer.cli.main get-data AAPL

# 4. 会社情報も含めて詳細表示
uv run python -m stock_analyzer.cli.main get-data AAPL --info --period 1y
```

**実行結果例**:
```
📊 AAPL 株価データ:
期間: 2023-08-28 ～ 2024-08-28 (252日分)
最新価格: $230.51
期間最高値: $237.23
期間最安値: $164.08
平均出来高: 52,441,832

🏢 AAPL 会社情報:
会社名: Apple Inc.
セクター: Technology
業界: Consumer Electronics
時価総額: $3,520,000,000,000

📈 最新5日間の終値:
2024-08-24: $228.87 (+0.24, +0.1%) 📈
2024-08-25: $227.37 (-1.50, -0.7%) 📉
2024-08-26: $227.52 (+0.15, +0.1%) 📈
2024-08-27: $228.03 (+0.51, +0.2%) 📈
2024-08-28: $230.51 (+2.48, +1.1%) 📈
```

### 2. テクニカル分析の基本

```bash
# 基本的なテクニカル分析
uv run python -m stock_analyzer.cli.main analyze AAPL --period 6mo

# 売買シグナル付きの詳細分析
uv run python -m stock_analyzer.cli.main analyze AAPL --period 1y --signals
```

**実行結果例**:
```
🎯 AAPL テクニカル分析結果:
期間: 2023-08-28 ～ 2024-08-28
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
BB: 中央レンジ ➡️
```

### 3. 複数銘柄の簡単比較

```bash
# 主要テック株の価格比較
uv run python -m stock_analyzer.cli.main compare AAPL MSFT GOOGL META NVDA

# 詳細比較分析
uv run python -m stock_analyzer.cli.main compare-advanced AAPL MSFT GOOGL --sort-by investment_score
```

---

## 日常の投資分析ワークフロー

### 朝の市況チェックルーチン

```bash
#!/bin/bash
# morning_check.sh - 朝の市況チェックスクリプト

echo "=== 朝の株式市場チェック $(date) ==="

# 1. 注目銘柄の価格確認
echo "🔍 主要銘柄価格チェック"
uv run python -m stock_analyzer.cli.main compare AAPL MSFT GOOGL AMZN NVDA META TSLA

echo ""
echo "📊 詳細テクニカル分析"

# 2. 注目銘柄のテクニカル分析
SYMBOLS=("AAPL" "MSFT" "GOOGL")
for symbol in "${SYMBOLS[@]}"; do
    echo "--- $symbol 分析 ---"
    uv run python -m stock_analyzer.cli.main analyze "$symbol" --signals
    echo ""
done

# 3. 30日予測の更新
echo "🔮 30日予測更新"
for symbol in "${SYMBOLS[@]}"; do
    echo "--- $symbol 30日予測 ---"
    uv run python -m stock_analyzer.cli.main predict "$symbol" --prediction-days 30
    echo ""
done
```

### 週末の詳細分析

```bash
#!/bin/bash
# weekend_analysis.sh - 週末の詳細分析

DATE=$(date +%Y%m%d)
REPORT_DIR="weekly_reports/$DATE"
mkdir -p "$REPORT_DIR"

echo "=== 週末詳細分析 $(date) ==="

# 1. ポートフォリオ全体分析
echo "💼 ポートフォリオ分析"
uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist my-portfolio \
  --export-csv > "$REPORT_DIR/portfolio_analysis.txt"

# 2. 候補銘柄のスクリーニング
echo "🔍 新規候補銘柄スクリーニング"
uv run python -m stock_analyzer.cli.main compare-advanced \
  --preset sp500-top20 \
  --sort-by investment_score \
  --export-csv > "$REPORT_DIR/candidate_screening.txt"

# 3. バックテスト精度検証（先月の予測を検証）
echo "📈 予測精度検証"
LAST_MONTH=$(date -d "1 month ago" +%Y-%m-01)
THIS_MONTH=$(date +%Y-%m-01)

MAIN_STOCKS=("AAPL" "MSFT" "GOOGL")
for symbol in "${MAIN_STOCKS[@]}"; do
    echo "--- $symbol バックテスト ---" >> "$REPORT_DIR/backtest_results.txt"
    uv run python -m stock_analyzer.cli.main backtest "$symbol" \
      --investment-date "$LAST_MONTH" \
      --validation-date "$THIS_MONTH" >> "$REPORT_DIR/backtest_results.txt"
    echo "" >> "$REPORT_DIR/backtest_results.txt"
done

echo "分析完了。レポートは $REPORT_DIR に保存されました。"
```

---

## 予測と検証の活用

### 1. 系統的な予測検証プロセス

```bash
# ステップ1: 予測を実行して記録
echo "=== 2024年9月の予測記録 ==="

SYMBOLS=("AAPL" "MSFT" "GOOGL" "AMZN" "NVDA")
PREDICTION_DATE=$(date +%Y-%m-%d)
TARGET_DATE=$(date -d "+30 days" +%Y-%m-%d)

for symbol in "${SYMBOLS[@]}"; do
    echo "--- $symbol 30日予測 ($PREDICTION_DATE -> $TARGET_DATE) ---"

    # 予測実行
    uv run python -m stock_analyzer.cli.main predict "$symbol" \
      --prediction-days 30 \
      --prediction-type direction > "predictions/${symbol}_${PREDICTION_DATE}.txt"

    # サマリーを記録
    echo "$PREDICTION_DATE,$TARGET_DATE,$symbol,direction,30" >> "prediction_log.csv"
done

# ステップ2: 30日後に精度検証を実行
echo "=== 予測精度検証（30日後実行） ==="

# prediction_log.csvを読み込んで過去の予測を検証
while IFS=',' read -r pred_date target_date symbol pred_type days; do
    if [[ $target_date <= $(date +%Y-%m-%d) ]]; then
        echo "--- $symbol の予測検証 ($pred_date -> $target_date) ---"

        uv run python -m stock_analyzer.cli.main backtest "$symbol" \
          --investment-date "$pred_date" \
          --validation-date "$target_date" \
          --prediction-type "$pred_type"
    fi
done < prediction_log.csv
```

### 2. 異なる期間での予測精度比較

```bash
#!/bin/bash
# multi_period_prediction.sh - 複数期間での予測比較

SYMBOL="AAPL"
PERIODS=(7 14 30 60)

echo "=== $SYMBOL 複数期間予測 ==="

for days in "${PERIODS[@]}"; do
    echo "--- ${days}日後の予測 ---"

    # 方向性予測
    echo "方向性予測:"
    uv run python -m stock_analyzer.cli.main predict "$SYMBOL" \
      --prediction-days "$days" \
      --prediction-type direction

    echo ""

    # リターン予測
    echo "リターン予測:"
    uv run python -m stock_analyzer.cli.main predict "$SYMBOL" \
      --prediction-days "$days" \
      --prediction-type return

    echo "=================================================="
done
```

### 3. 予測精度の統計的分析

```bash
#!/bin/bash
# accuracy_analysis.sh - 予測精度の統計分析

SYMBOL="AAPL"
START_DATE="2024-01-01"
END_DATE="2024-06-30"

echo "=== $SYMBOL 予測精度統計分析 ($START_DATE ~ $END_DATE) ==="

# 1ヶ月間隔で予測精度を検証
current_date="$START_DATE"
correct_predictions=0
total_predictions=0

while [[ $current_date < $END_DATE ]]; do
    validation_date=$(date -d "$current_date +30 days" +%Y-%m-%d)

    if [[ $validation_date <= $END_DATE ]]; then
        echo "検証期間: $current_date -> $validation_date"

        # バックテスト実行（結果を一時ファイルに保存）
        uv run python -m stock_analyzer.cli.main backtest "$SYMBOL" \
          --investment-date "$current_date" \
          --validation-date "$validation_date" > temp_result.txt

        # 正解かどうかを判定（出力から"正解: ○"を検索）
        if grep -q "正解: ○" temp_result.txt; then
            ((correct_predictions++))
        fi
        ((total_predictions++))
    fi

    # 次の日付に進む（2週間間隔）
    current_date=$(date -d "$current_date +14 days" +%Y-%m-%d)
done

# 統計結果を表示
accuracy=$(echo "scale=2; $correct_predictions * 100 / $total_predictions" | bc)
echo ""
echo "=== 統計結果 ==="
echo "総予測回数: $total_predictions"
echo "正解回数: $correct_predictions"
echo "予測精度: $accuracy%"

# 一時ファイルを削除
rm -f temp_result.txt
```

---

## ポートフォリオ管理

### 1. 月次ポートフォリオリバランス

```bash
#!/bin/bash
# monthly_rebalance.sh - 月次ポートフォリオリバランス

DATE=$(date +%Y%m%d)
PORTFOLIO_DIR="portfolio_reports/$DATE"
mkdir -p "$PORTFOLIO_DIR"

echo "=== 月次ポートフォリオリバランス $(date) ==="

# 1. 現在のポートフォリオ分析
echo "🏦 現在のポートフォリオ分析"
uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist current-portfolio \
  --investment-amount 100000 \
  --export-csv > "$PORTFOLIO_DIR/current_analysis.txt"

# 2. 新規候補銘柄の評価
echo "🔍 新規候補銘柄評価"
uv run python -m stock_analyzer.cli.main compare-advanced \
  --watchlist candidate-stocks \
  --sort-by investment_score \
  --export-csv > "$PORTFOLIO_DIR/candidates.txt"

# 3. リスク分析
echo "⚠️ リスク許容度別分析"

# 保守的なポートフォリオ
uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist current-portfolio \
  --investment-amount 100000 \
  --risk-tolerance 0.2 \
  --max-stocks 5 > "$PORTFOLIO_DIR/conservative.txt"

# 積極的なポートフォリオ
uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist current-portfolio \
  --investment-amount 100000 \
  --risk-tolerance 0.6 \
  --max-stocks 12 > "$PORTFOLIO_DIR/aggressive.txt"

# 4. セクター分散分析
echo "🏭 セクター別分析"
TECH_STOCKS="AAPL MSFT GOOGL META NVDA"
FINANCE_STOCKS="JPM BAC WFC GS MS"
HEALTH_STOCKS="JNJ PFE UNH ABBV MRK"

echo "テック株ポートフォリオ:" > "$PORTFOLIO_DIR/sector_analysis.txt"
uv run python -m stock_analyzer.cli.main portfolio $TECH_STOCKS \
  --investment-amount 100000 >> "$PORTFOLIO_DIR/sector_analysis.txt"

echo -e "\n金融株ポートフォリオ:" >> "$PORTFOLIO_DIR/sector_analysis.txt"
uv run python -m stock_analyzer.cli.main portfolio $FINANCE_STOCKS \
  --investment-amount 100000 >> "$PORTFOLIO_DIR/sector_analysis.txt"

echo "レポートが $PORTFOLIO_DIR に生成されました。"
```

### 2. ドルコスト平均法シミュレーション

```bash
#!/bin/bash
# dollar_cost_averaging.sh - ドルコスト平均法シミュレーション

SYMBOL="AAPL"
MONTHLY_INVESTMENT=5000
MONTHS=12
START_DATE="2023-09-01"

echo "=== $SYMBOL ドルコスト平均法シミュレーション ==="
echo "月次投資額: \$$MONTHLY_INVESTMENT"
echo "投資期間: ${MONTHS}ヶ月"
echo "開始日: $START_DATE"
echo ""

total_investment=0
total_shares=0
current_date="$START_DATE"

for ((month=1; month<=MONTHS; month++)); do
    # 各月の初日の価格を取得
    price_data=$(uv run python -m stock_analyzer.cli.main get-data "$SYMBOL" --period 2y)

    echo "第${month}ヶ月目 ($current_date):"
    echo "  投資額: \$$MONTHLY_INVESTMENT"

    # 簡易的な価格計算（実際の実装では特定日の価格を取得）
    # この例では概算値を使用
    estimated_price=$((200 + month * 2))  # 仮想的な価格推移
    shares=$(echo "scale=4; $MONTHLY_INVESTMENT / $estimated_price" | bc)

    total_investment=$((total_investment + MONTHLY_INVESTMENT))
    total_shares=$(echo "scale=4; $total_shares + $shares" | bc)

    echo "  株価: \$$estimated_price"
    echo "  購入株数: $shares 株"
    echo "  累計投資額: \$$total_investment"
    echo "  累計株数: $total_shares 株"
    echo ""

    # 次の月に進む
    current_date=$(date -d "$current_date +1 month" +%Y-%m-%d)
done

# 最終評価
final_price=$((200 + MONTHS * 2 + 10))  # 最終価格（仮想）
portfolio_value=$(echo "scale=2; $total_shares * $final_price" | bc)
total_return=$(echo "scale=2; $portfolio_value - $total_investment" | bc)
return_rate=$(echo "scale=2; $total_return * 100 / $total_investment" | bc)

echo "=== 最終結果 ==="
echo "総投資額: \$$total_investment"
echo "総株数: $total_shares 株"
echo "平均取得価格: \$$(echo "scale=2; $total_investment / $total_shares" | bc)"
echo "最終株価: \$$final_price"
echo "ポートフォリオ価値: \$$portfolio_value"
echo "総リターン: \$$total_return ($return_rate%)"
```

### 3. リスク・リターン最適化

```bash
#!/bin/bash
# risk_return_optimization.sh - リスク・リターン最適化

echo "=== ポートフォリオ最適化分析 ==="

# 異なるリスク許容度での最適化
RISK_LEVELS=(0.1 0.2 0.3 0.4 0.5 0.6)
INVESTMENT_AMOUNT=100000
SYMBOLS="AAPL MSFT GOOGL AMZN NVDA META TSLA BRK.B JNJ PG"

for risk in "${RISK_LEVELS[@]}"; do
    echo "--- リスク許容度: $risk ---"

    uv run python -m stock_analyzer.cli.main portfolio $SYMBOLS \
      --investment-amount "$INVESTMENT_AMOUNT" \
      --risk-tolerance "$risk" \
      --max-stocks 8 > "optimization_risk_${risk}.txt"

    echo "結果を optimization_risk_${risk}.txt に保存"
    echo ""
done

# 異なる投資金額での最適化
AMOUNTS=(50000 100000 200000 500000)
for amount in "${AMOUNTS[@]}"; do
    echo "--- 投資金額: \$$amount ---"

    uv run python -m stock_analyzer.cli.main portfolio $SYMBOLS \
      --investment-amount "$amount" \
      --risk-tolerance 0.3 \
      --max-stocks 10 > "optimization_amount_${amount}.txt"

    echo "結果を optimization_amount_${amount}.txt に保存"
    echo ""
done
```

---

## 高度な分析手法

### 1. 相関分析とペア取引

```bash
#!/bin/bash
# correlation_analysis.sh - 銘柄間相関分析

echo "=== 銘柄間相関分析 ==="

# 関連性の高そうな銘柄ペア
PAIRS=(
    "AAPL MSFT"      # テック大手
    "JPM BAC"        # 大手銀行
    "KO PEP"         # 飲料大手
    "JNJ PFE"        # 製薬大手
    "XOM CVX"        # エネルギー大手
)

for pair in "${PAIRS[@]}"; do
    read -r stock1 stock2 <<< "$pair"
    echo "--- $stock1 vs $stock2 ---"

    # 各銘柄の詳細分析
    echo "$stock1 分析:"
    uv run python -m stock_analyzer.cli.main analyze "$stock1" --period 1y
    echo ""

    echo "$stock2 分析:"
    uv run python -m stock_analyzer.cli.main analyze "$stock2" --period 1y
    echo ""

    # ペア比較
    echo "ペア比較:"
    uv run python -m stock_analyzer.cli.main compare-advanced "$stock1" "$stock2"
    echo "=================================================="
done
```

### 2. セクターローテーション分析

```bash
#!/bin/bash
# sector_rotation.sh - セクターローテーション分析

echo "=== セクターローテーション分析 ==="

# 主要セクター代表銘柄
declare -A sectors
sectors[Technology]="AAPL MSFT GOOGL NVDA"
sectors[Healthcare]="JNJ PFE UNH ABBV"
sectors[Finance]="JPM BAC WFC GS"
sectors[Energy]="XOM CVX SLB EOG"
sectors[ConsumerGoods]="PG KO PEP WMT"
sectors[Industrial]="GE CAT BA HON"

for sector in "${!sectors[@]}"; do
    echo "=== $sector セクター ==="
    stocks=${sectors[$sector]}

    # セクター全体のポートフォリオ分析
    uv run python -m stock_analyzer.cli.main portfolio $stocks \
      --investment-amount 100000 \
      --period 1y

    # セクター内比較
    uv run python -m stock_analyzer.cli.main compare-advanced $stocks \
      --sort-by investment_score

    echo ""
done

# セクター間比較（代表銘柄）
echo "=== セクター間比較 ==="
SECTOR_REPRESENTATIVES="AAPL JNJ JPM XOM PG GE"
uv run python -m stock_analyzer.cli.main compare-advanced $SECTOR_REPRESENTATIVES \
  --sort-by investment_score \
  --export-csv
```

### 3. 経済指標との相関分析

```bash
#!/bin/bash
# economic_correlation.sh - 経済指標相関分析

echo "=== 経済指標相関分析 ==="

# 金利敏感株（銀行、不動産）
echo "--- 金利敏感株分析 ---"
INTEREST_SENSITIVE="JPM BAC WFC GS REITs"
for symbol in $INTEREST_SENSITIVE; do
    if [[ $symbol != "REITs" ]]; then  # REITsは例外処理
        echo "$symbol 分析:"
        uv run python -m stock_analyzer.cli.main analyze "$symbol" --period 2y
        echo ""
    fi
done

# インフレ対応株（コモディティ、エネルギー）
echo "--- インフレ対応株分析 ---"
INFLATION_HEDGE="XOM CVX GOLD SLV"
for symbol in $INFLATION_HEDGE; do
    echo "$symbol 分析:"
    uv run python -m stock_analyzer.cli.main analyze "$symbol" --period 2y
    echo ""
done

# 景気循環株（工業、材料）
echo "--- 景気循環株分析 ---"
CYCLICAL="CAT GE AA X"
for symbol in $CYCLICAL; do
    echo "$symbol 分析:"
    uv run python -m stock_analyzer.cli.main analyze "$symbol" --period 2y
    echo ""
done

# 安全資産（ディフェンシブ株）
echo "--- ディフェンシブ株分析 ---"
DEFENSIVE="JNJ PG KO WMT MCD"
uv run python -m stock_analyzer.cli.main portfolio $DEFENSIVE \
  --investment-amount 100000 \
  --risk-tolerance 0.2
```

---

## 自動化スクリプト例

### 1. 日次自動レポート生成

```bash
#!/bin/bash
# auto_daily_report.sh - 日次自動レポート生成

# crontab設定例: 0 9 * * 1-5 /path/to/auto_daily_report.sh

DATE=$(date +%Y%m%d)
REPORT_DIR="daily_reports/$DATE"
mkdir -p "$REPORT_DIR"

# ログ設定
exec > "$REPORT_DIR/daily_report.log" 2>&1

echo "=== Stock Analyzer 日次レポート $(date) ==="

# 設定読み込み
WATCHLIST="my-portfolio"
CANDIDATES="candidates"
ALERT_THRESHOLD=5.0  # 5%以上の変動で警告

# 1. ポートフォリオ健康チェック
echo "🏥 ポートフォリオ健康チェック"
uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist "$WATCHLIST" \
  --export-csv > "$REPORT_DIR/portfolio_health.txt"

# 2. 価格変動アラート
echo "⚠️ 価格変動アラート"
PORTFOLIO_STOCKS=$(uv run python -m stock_analyzer.cli.main config --show | grep -A 10 "$WATCHLIST" | grep -E "^\s+-" | cut -d'"' -f2)

for stock in $PORTFOLIO_STOCKS; do
    # 1日の変動率をチェック（簡易実装）
    result=$(uv run python -m stock_analyzer.cli.main get-data "$stock" --period 5d)

    # 大幅変動の検出（grep で最新の変動率をチェック）
    if echo "$result" | grep -E "\+[0-9]+\.[0-9]+%" | grep -E "\+[5-9]\.[0-9]+%|\+[0-9]{2,}\.[0-9]+%" > /dev/null; then
        echo "📈 $stock: 大幅上昇検出！" >> "$REPORT_DIR/alerts.txt"
    fi

    if echo "$result" | grep -E "\-[5-9]\.[0-9]+%|\-[0-9]{2,}\.[0-9]+%" > /dev/null; then
        echo "📉 $stock: 大幅下落検出！" >> "$REPORT_DIR/alerts.txt"
    fi
done

# 3. 新規投資機会の検出
echo "💡 新規投資機会"
uv run python -m stock_analyzer.cli.main compare-advanced \
  --watchlist "$CANDIDATES" \
  --sort-by investment_score \
  --export-csv > "$REPORT_DIR/opportunities.txt"

# 4. 予測精度モニタリング
echo "🎯 予測精度モニタリング"
# 30日前の予測を検証
PAST_DATE=$(date -d "30 days ago" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

MAIN_STOCKS=("AAPL" "MSFT" "GOOGL")
for stock in "${MAIN_STOCKS[@]}"; do
    uv run python -m stock_analyzer.cli.main backtest "$stock" \
      --investment-date "$PAST_DATE" \
      --validation-date "$TODAY" >> "$REPORT_DIR/accuracy_check.txt"
done

# 5. 市場概況サマリー
echo "🌍 市場概況"
MARKET_INDICES="^GSPC ^IXIC ^DJI"  # S&P500, NASDAQ, Dow Jones
for index in $MARKET_INDICES; do
    uv run python -m stock_analyzer.cli.main get-data "$index" --period 5d >> "$REPORT_DIR/market_summary.txt"
done

# 6. メール通知（設定されている場合）
if [[ -n "$EMAIL_RECIPIENT" ]]; then
    {
        echo "Stock Analyzer 日次レポート - $DATE"
        echo ""
        echo "=== アラート ==="
        cat "$REPORT_DIR/alerts.txt" 2>/dev/null || echo "アラートなし"
        echo ""
        echo "=== 新規投資機会 ==="
        head -20 "$REPORT_DIR/opportunities.txt"
        echo ""
        echo "詳細レポート: $REPORT_DIR/"
    } | mail -s "Stock Analyzer Daily Report - $DATE" "$EMAIL_RECIPIENT"
fi

echo "日次レポート生成完了: $REPORT_DIR/"
```

### 2. 予測精度トラッキング

```bash
#!/bin/bash
# prediction_tracker.sh - 予測精度自動トラッキング

TRACKING_FILE="prediction_tracking.csv"
ACCURACY_REPORT="accuracy_report.txt"

# CSVヘッダーを作成（初回のみ）
if [[ ! -f "$TRACKING_FILE" ]]; then
    echo "date,symbol,prediction_type,prediction_days,predicted_value,actual_result,accuracy,confidence_score" > "$TRACKING_FILE"
fi

echo "=== 予測精度トラッキング $(date) ==="

# 30日前の予測を検証
PREDICTION_DATE=$(date -d "30 days ago" +%Y-%m-%d)
VALIDATION_DATE=$(date +%Y-%m-%d)

TRACKED_SYMBOLS=("AAPL" "MSFT" "GOOGL" "AMZN" "NVDA")

for symbol in "${TRACKED_SYMBOLS[@]}"; do
    echo "検証中: $symbol ($PREDICTION_DATE -> $VALIDATION_DATE)"

    # バックテスト実行
    result=$(uv run python -m stock_analyzer.cli.main backtest "$symbol" \
      --investment-date "$PREDICTION_DATE" \
      --validation-date "$VALIDATION_DATE" \
      --prediction-type direction)

    # 結果を解析してCSVに記録
    if echo "$result" | grep -q "正解: ○"; then
        accuracy="TRUE"
    else
        accuracy="FALSE"
    fi

    confidence_score=$(echo "$result" | grep -o "信頼性スコア: [0-9.]*" | cut -d' ' -f2)
    predicted_direction=$(echo "$result" | grep -o "予測方向: [上昇下降]*" | cut -d' ' -f2)

    # CSVに記録
    echo "$VALIDATION_DATE,$symbol,direction,30,$predicted_direction,actual,$accuracy,$confidence_score" >> "$TRACKING_FILE"
done

# 精度統計の生成
echo "=== 予測精度統計 ===" > "$ACCURACY_REPORT"
echo "生成日: $(date)" >> "$ACCURACY_REPORT"
echo "" >> "$ACCURACY_REPORT"

# 全体精度
total_predictions=$(tail -n +2 "$TRACKING_FILE" | wc -l)
correct_predictions=$(tail -n +2 "$TRACKING_FILE" | grep -c "TRUE")
overall_accuracy=$(echo "scale=2; $correct_predictions * 100 / $total_predictions" | bc)

echo "全体統計:" >> "$ACCURACY_REPORT"
echo "  総予測数: $total_predictions" >> "$ACCURACY_REPORT"
echo "  正解数: $correct_predictions" >> "$ACCURACY_REPORT"
echo "  全体精度: $overall_accuracy%" >> "$ACCURACY_REPORT"
echo "" >> "$ACCURACY_REPORT"

# 銘柄別精度
echo "銘柄別精度:" >> "$ACCURACY_REPORT"
for symbol in "${TRACKED_SYMBOLS[@]}"; do
    symbol_total=$(grep ",$symbol," "$TRACKING_FILE" | wc -l)
    symbol_correct=$(grep ",$symbol," "$TRACKING_FILE" | grep -c "TRUE")

    if [[ $symbol_total -gt 0 ]]; then
        symbol_accuracy=$(echo "scale=2; $symbol_correct * 100 / $symbol_total" | bc)
        echo "  $symbol: $symbol_accuracy% ($symbol_correct/$symbol_total)" >> "$ACCURACY_REPORT"
    fi
done

echo "精度レポートを $ACCURACY_REPORT に生成しました。"
```

### 3. 自動リバランス提案

```bash
#!/bin/bash
# auto_rebalance.sh - 自動リバランス提案

CURRENT_PORTFOLIO="current-portfolio"
TARGET_ALLOCATION_FILE="target_allocation.json"
REBALANCE_REPORT="rebalance_proposal.txt"
THRESHOLD=0.05  # 5%以上のずれでリバランス提案

echo "=== 自動リバランス提案 $(date) ===" > "$REBALANCE_REPORT"

# 現在のポートフォリオ分析
current_analysis=$(uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist "$CURRENT_PORTFOLIO" \
  --investment-amount 100000)

echo "現在のポートフォリオ:" >> "$REBALANCE_REPORT"
echo "$current_analysis" >> "$REBALANCE_REPORT"
echo "" >> "$REBALANCE_REPORT"

# 最適ポートフォリオの生成
optimal_analysis=$(uv run python -m stock_analyzer.cli.main portfolio \
  --watchlist "$CURRENT_PORTFOLIO" \
  --investment-amount 100000 \
  --risk-tolerance 0.3)

echo "最適ポートフォリオ提案:" >> "$REBALANCE_REPORT"
echo "$optimal_analysis" >> "$REBALANCE_REPORT"
echo "" >> "$REBALANCE_REPORT"

# リバランス提案の生成
echo "=== リバランス提案 ===" >> "$REBALANCE_REPORT"
echo "閾値: ${THRESHOLD}以上のずれでリバランス推奨" >> "$REBALANCE_REPORT"
echo "" >> "$REBALANCE_REPORT"

# 実際の計算は簡略化（実装では詳細な比較が必要）
echo "推奨アクション:" >> "$REBALANCE_REPORT"
echo "1. AAPL: 35% -> 40% (5%増加)" >> "$REBALANCE_REPORT"
echo "2. MSFT: 30% -> 25% (5%減少)" >> "$REBALANCE_REPORT"
echo "3. GOOGL: 20% -> 20% (変更なし)" >> "$REBALANCE_REPORT"
echo "4. AMZN: 15% -> 15% (変更なし)" >> "$REBALANCE_REPORT"
echo "" >> "$REBALANCE_REPORT"

echo "リバランス提案を $REBALANCE_REPORT に生成しました。"

# 重要な変更がある場合はアラート
if grep -E "増加|減少" "$REBALANCE_REPORT" > /dev/null; then
    echo "⚠️ リバランスが推奨されます。$REBALANCE_REPORT を確認してください。"

    # メール通知（設定されている場合）
    if [[ -n "$EMAIL_RECIPIENT" ]]; then
        mail -s "Portfolio Rebalance Alert - $(date +%Y-%m-%d)" "$EMAIL_RECIPIENT" < "$REBALANCE_REPORT"
    fi
else
    echo "✅ 現在のポートフォリオは最適な状態です。"
fi
```

---

## 実際の投資戦略への応用

### 1. モメンタム戦略

```bash
#!/bin/bash
# momentum_strategy.sh - モメンタム投資戦略

echo "=== モメンタム投資戦略 ==="

# S&P500の主要銘柄
SP500_MAJOR="AAPL MSFT GOOGL AMZN NVDA META TSLA BRK.B UNH JNJ"

# 1. 過去3ヶ月のパフォーマンス分析
echo "📈 過去3ヶ月パフォーマンス分析"
for symbol in $SP500_MAJOR; do
    echo "--- $symbol ---"
    uv run python -m stock_analyzer.cli.main analyze "$symbol" --period 3mo
    echo ""
done

# 2. モメンタム指標による順位付け
echo "🚀 モメンタムランキング"
uv run python -m stock_analyzer.cli.main compare-advanced $SP500_MAJOR \
  --sort-by investment_score \
  --period 3mo > momentum_ranking.txt

# 3. 上位銘柄でのポートフォリオ構築
echo "🏆 モメンタム上位ポートフォリオ"
# 上位5銘柄を抽出（簡略化）
TOP_5="AAPL MSFT GOOGL AMZN NVDA"

uv run python -m stock_analyzer.cli.main portfolio $TOP_5 \
  --investment-amount 100000 \
  --risk-tolerance 0.5 \
  --max-stocks 5

# 4. エントリー・エグジット シグナル
echo "📊 売買シグナル分析"
for symbol in $TOP_5; do
    echo "--- $symbol 売買シグナル ---"
    uv run python -m stock_analyzer.cli.main analyze "$symbol" --signals --period 6mo
    echo ""
done
```

### 2. バリュー投資戦略

```bash
#!/bin/bash
# value_strategy.sh - バリュー投資戦略

echo "=== バリュー投資戦略 ==="

# バリュー株候補（低RSI、割安指標）
VALUE_CANDIDATES="WMT JNJ PG KO PEP MCD IBM GE F GM"

echo "💎 バリュー株スクリーニング"

# 1. RSIによる売られすぎ判定
echo "--- RSI分析（30以下で売られすぎ） ---"
for symbol in $VALUE_CANDIDATES; do
    result=$(uv run python -m stock_analyzer.cli.main analyze "$symbol" --period 6mo)
    rsi_value=$(echo "$result" | grep -o "RSI.*: [0-9.]*" | grep -o "[0-9.]*$")

    if (( $(echo "$rsi_value < 30" | bc -l) )); then
        echo "🔥 $symbol: RSI $rsi_value (売られすぎ)"
    elif (( $(echo "$rsi_value < 40" | bc -l) )); then
        echo "📉 $symbol: RSI $rsi_value (やや割安)"
    fi
done

# 2. 長期トレンド分析
echo "--- 長期トレンド分析 ---"
for symbol in $VALUE_CANDIDATES; do
    echo "$symbol 2年チャート分析:"
    uv run python -m stock_analyzer.cli.main analyze "$symbol" --period 2y --signals
    echo ""
done

# 3. バリュー ポートフォリオ構築
echo "--- バリューポートフォリオ ---"
uv run python -m stock_analyzer.cli.main portfolio $VALUE_CANDIDATES \
  --investment-amount 100000 \
  --risk-tolerance 0.2 \
  --max-stocks 8

# 4. 長期予測
echo "--- 90日長期予測 ---"
SELECTED_VALUE="WMT JNJ PG KO"
for symbol in $SELECTED_VALUE; do
    echo "$symbol 90日予測:"
    uv run python -m stock_analyzer.cli.main predict "$symbol" \
      --prediction-days 90 \
      --training-months 36
    echo ""
done
```

### 3. ペア取引戦略

```bash
#!/bin/bash
# pairs_trading.sh - ペア取引戦略

echo "=== ペア取引戦略 ==="

# 相関の高いペア
declare -A pairs
pairs["Tech Giants"]="AAPL MSFT"
pairs["Oil Majors"]="XOM CVX"
pairs["Big Banks"]="JPM BAC"
pairs["Telecom"]="VZ T"
pairs["Retail"]="WMT TGT"

for pair_name in "${!pairs[@]}"; do
    stocks=${pairs[$pair_name]}
    read -r stock1 stock2 <<< "$stocks"

    echo "=== $pair_name ペア: $stock1 vs $stock2 ==="

    # 1. 個別分析
    echo "--- 個別分析 ---"
    echo "$stock1:"
    uv run python -m stock_analyzer.cli.main analyze "$stock1" --period 6mo
    echo ""

    echo "$stock2:"
    uv run python -m stock_analyzer.cli.main analyze "$stock2" --period 6mo
    echo ""

    # 2. 相対的強さ比較
    echo "--- 相対比較 ---"
    uv run python -m stock_analyzer.cli.main compare-advanced "$stock1" "$stock2" \
      --sort-by investment_score

    # 3. 予測比較
    echo "--- 30日予測比較 ---"
    echo "$stock1 予測:"
    uv run python -m stock_analyzer.cli.main predict "$stock1" --prediction-days 30
    echo ""

    echo "$stock2 予測:"
    uv run python -m stock_analyzer.cli.main predict "$stock2" --prediction-days 30
    echo ""

    # 4. ペア取引提案
    echo "--- ペア取引提案 ---"
    # 簡易的な提案（実際の実装では価格比率の統計分析が必要）
    echo "推奨: 相対的に強い銘柄をロング、弱い銘柄をショート"
    echo "リスク管理: 10%以上の逆行で損切り"
    echo ""
    echo "=================================================="
done
```

### 4. 金利変動対応戦略

```bash
#!/bin/bash
# interest_rate_strategy.sh - 金利変動対応戦略

echo "=== 金利変動対応投資戦略 ==="

# 金利上昇局面で有利な銘柄
HIGH_RATE_BENEFICIARIES="JPM BAC WFC GS XOM CVX"

# 金利下降局面で有利な銘柄
LOW_RATE_BENEFICIARIES="AAPL MSFT GOOGL AMZN NVDA TSLA"

# 金利中立的な銘柄
RATE_NEUTRAL="JNJ PG KO WMT MCD"

echo "🔺 金利上昇局面対応ポートフォリオ"
uv run python -m stock_analyzer.cli.main portfolio $HIGH_RATE_BENEFICIARIES \
  --investment-amount 100000 \
  --risk-tolerance 0.4 \
  --period 1y

echo ""
echo "🔻 金利下降局面対応ポートフォリオ"
uv run python -m stock_analyzer.cli.main portfolio $LOW_RATE_BENEFICIARIES \
  --investment-amount 100000 \
  --risk-tolerance 0.5 \
  --period 1y

echo ""
echo "➡️ 金利中立ポートフォリオ（安定志向）"
uv run python -m stock_analyzer.cli.main portfolio $RATE_NEUTRAL \
  --investment-amount 100000 \
  --risk-tolerance 0.2 \
  --period 1y

# 各戦略の過去パフォーマンス検証
echo ""
echo "=== 過去パフォーマンス検証 ==="

BACK_TEST_PERIODS=("2024-01-01 2024-03-31" "2024-04-01 2024-06-30" "2024-07-01 2024-08-28")

for period in "${BACK_TEST_PERIODS[@]}"; do
    read -r start_date end_date <<< "$period"
    echo "--- $start_date から $end_date ---"

    # 代表銘柄での検証
    for symbol in "JPM" "AAPL" "JNJ"; do
        echo "$symbol バックテスト:"
        uv run python -m stock_analyzer.cli.main backtest "$symbol" \
          --investment-date "$start_date" \
          --validation-date "$end_date"
        echo ""
    done
done
```

---

この使用例集は、Stock Analyzerの実践的な活用方法を幅広くカバーしています。基本的な操作から高度な投資戦略まで、段階的に習得していくことで、効果的な投資判断支援ツールとしてご活用いただけます。

各スクリプトは実際の投資環境に合わせてカスタマイズしてください。特に投資金額、リスク許容度、監視銘柄などは、個人の投資方針に応じて調整することをお勧めします。
