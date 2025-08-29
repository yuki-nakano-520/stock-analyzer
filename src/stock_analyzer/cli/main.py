"""Stock Analyzer CLI (Command Line Interface)"""

from typing import Any

import click

# ML予測機能をインポート
from ..analysis.backtesting import BacktestSimulator

# フィーチャー生成機能をインポート
# テクニカル分析機能をインポート
from ..analysis.indicators import analyze_signals, calculate_all_indicators

# ポートフォリオ分析機能をインポート
from ..analysis.portfolio import PortfolioAnalyzer, PortfolioConfig, compare_stocks

# 設定管理機能をインポート
from ..config import PresetManager, get_config, get_preset_symbols

# データ取得機能をインポート
from ..data.fetchers import get_company_info, get_stock_data

# CSV出力機能をインポート
from ..reports.csv_exporter import JapaneseCsvExporter


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="cli")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """
    Stock Analyzer - 株価分析CLIツール

    初心者向けの株価データ取得・分析ツールです。
    """
    pass


@cli.command()
@click.argument("symbol", type=str)
@click.option(
    "--period",
    default="1y",
    type=click.Choice(
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    ),
    help="データ取得期間",
)
@click.option("--info", is_flag=True, help="会社情報も表示する")
def get_data(symbol: str, period: str, info: bool) -> None:
    """
    株価データを取得して表示する

    SYMBOL: 株式シンボル（例：AAPL, MSFT, GOOGL）
    """
    try:
        logger.info(f"CLI: データ取得開始 - {symbol}, 期間: {period}")

        click.echo(f"🔍 {symbol} の株価データを取得中...")

        # 株価データ取得
        data = get_stock_data(symbol, period)

        # 基本情報表示
        current_price = data["Close"].iloc[-1]
        data_count = len(data)
        start_date = data.index[0].strftime("%Y-%m-%d")
        end_date = data.index[-1].strftime("%Y-%m-%d")

        click.echo(f"\n📊 {symbol} 株価データ:")
        click.echo(f"期間: {start_date} ～ {end_date} ({data_count}日分)")
        click.echo(f"最新価格: ${current_price:.2f}")

        # 簡単な統計情報
        high_price = data["High"].max()
        low_price = data["Low"].min()
        avg_volume = data["Volume"].mean()

        click.echo(f"期間最高値: ${high_price:.2f}")
        click.echo(f"期間最安値: ${low_price:.2f}")
        click.echo(f"平均出来高: {avg_volume:,.0f}")

        # 会社情報表示（オプション）
        if info:
            click.echo(f"\n🏢 {symbol} 会社情報:")
            company_info = get_company_info(symbol)
            click.echo(f"会社名: {company_info['company_name']}")
            click.echo(f"セクター: {company_info['sector']}")
            click.echo(f"業界: {company_info['industry']}")
            click.echo(f"時価総額: ${company_info['market_cap']:,}")

        # 最新数日のデータ表示
        click.echo("\n📈 最新5日間の終値:")
        recent_data = data.tail(5)
        for date, row in recent_data.iterrows():
            formatted_date = date.strftime("%Y-%m-%d")
            change = row["Close"] - row["Open"]
            change_pct = (change / row["Open"]) * 100
            change_symbol = "📈" if change >= 0 else "📉"
            click.echo(
                f"{formatted_date}: ${row['Close']:.2f} ({change:+.2f}, {change_pct:+.1f}%) {change_symbol}"
            )

        logger.info(f"CLI: データ取得完了 - {symbol}")

    except Exception as e:
        logger.error(f"CLI: エラー発生 - {symbol}: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
def compare(symbols) -> None:
    """
    複数銘柄の現在価格を比較する

    SYMBOLS: 比較する銘柄（例：AAPL MSFT GOOGL）
    """
    try:
        click.echo(f"🔍 {len(symbols)}銘柄の価格比較:")

        results = []
        for symbol in symbols:
            try:
                data = get_stock_data(symbol, "1d")
                current_price = data["Close"].iloc[-1]
                results.append((symbol, current_price))
                click.echo(f"✅ {symbol}: ${current_price:.2f}")
            except Exception as e:
                click.echo(f"❌ {symbol}: エラー - {e}")

        # ソートして表示
        if results:
            click.echo("\n💰 価格順（高い順）:")
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            for i, (symbol, price) in enumerate(sorted_results, 1):
                click.echo(f"{i}. {symbol}: ${price:.2f}")

    except Exception as e:
        logger.error(f"CLI: 比較エラー: {e}")
        click.echo(f"❌ エラー: {e}", err=True)


@cli.command()
@click.argument("symbol", type=str)
@click.option(
    "--period",
    default="6mo",
    type=click.Choice(
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    ),
    help="データ取得期間（テクニカル分析には3mo以上推奨）",
)
@click.option("--signals", is_flag=True, help="売買シグナルも表示する")
def analyze(symbol: str, period: str, signals: bool) -> None:
    """
    株価のテクニカル分析を実行する

    SYMBOL: 株式シンボル（例：AAPL, MSFT, GOOGL）
    """
    try:
        logger.info(f"CLI: テクニカル分析開始 - {symbol}, 期間: {period}")

        click.echo(f"📊 {symbol} のテクニカル分析を実行中...")

        # 株価データ取得
        data = get_stock_data(symbol, period)

        # テクニカル指標計算
        indicators = calculate_all_indicators(data)

        # 基本情報
        current_price = data["Close"].iloc[-1]
        start_date = data.index[0].strftime("%Y-%m-%d")
        end_date = data.index[-1].strftime("%Y-%m-%d")

        click.echo(f"\n🎯 {symbol} テクニカル分析結果:")
        click.echo(f"期間: {start_date} ～ {end_date}")
        click.echo(f"現在価格: ${current_price:.2f}")

        # 移動平均線
        click.echo("\n📈 移動平均線:")
        click.echo(f"SMA5:  ${indicators['sma_5']:.2f}")
        click.echo(f"SMA20: ${indicators['sma_20']:.2f}")
        click.echo(f"SMA50: ${indicators['sma_50']:.2f}")

        # 主要指標
        click.echo("\n⚡ 主要指標:")
        click.echo(f"RSI (14日): {indicators['rsi']:.1f} ", nl=False)
        if indicators["rsi"] > 70:
            click.echo("(買われすぎ ⚠️)")
        elif indicators["rsi"] < 30:
            click.echo("(売られすぎ 📉)")
        else:
            click.echo("(中立 ➡️)")

        click.echo(f"MACD: {indicators['macd']:.3f}")
        click.echo(f"出来高比率: {indicators['volume_ratio']:.2f}x ", nl=False)
        if indicators["volume_ratio"] > 1.5:
            click.echo("(高出来高 🔥)")
        else:
            click.echo("(通常)")

        # ボリンジャーバンド位置
        bb_pos = indicators["bb_position"]
        click.echo(f"ボリンジャーバンド位置: {bb_pos:.1%} ", nl=False)
        if bb_pos > 0.8:
            click.echo("(上限付近 ⚠️)")
        elif bb_pos < 0.2:
            click.echo("(下限付近 📉)")
        else:
            click.echo("(中央付近 ➡️)")

        # 売買シグナル分析
        if signals:
            click.echo("\n🎯 売買シグナル分析:")
            signal_analysis = analyze_signals(indicators)
            for signal_type, signal_desc in signal_analysis.items():
                emoji = (
                    "📈"
                    if "買い" in signal_desc
                    else "📉"
                    if "売り" in signal_desc
                    else "➡️"
                )
                signal_name = signal_type.replace("_signal", "").upper()
                click.echo(f"{signal_name}: {signal_desc} {emoji}")

        logger.info(f"CLI: テクニカル分析完了 - {symbol}")

    except Exception as e:
        logger.error(f"CLI: 分析エラー - {symbol}: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--period",
    default=None,
    type=click.Choice(
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    ),
    help="データ取得期間",
)
@click.option("--investment-amount", default=None, type=float, help="総投資金額（USD）")
@click.option("--max-stocks", default=None, type=int, help="ポートフォリオ最大銘柄数")
@click.option("--risk-tolerance", default=None, type=float, help="リスク許容度（0-1）")
@click.option("--export-csv", is_flag=True, help="結果をCSVファイルにエクスポート")
@click.option(
    "--symbols-file", type=str, help="銘柄リストファイル (.txt, .csv, .json, .yaml)"
)
@click.option(
    "--preset", type=str, help="プリセット銘柄グループ (例: tech-giants, sp500-top20)"
)
@click.option("--watchlist", type=str, help="設定ファイルのウォッチリスト名")
@click.option("--list-presets", is_flag=True, help="利用可能なプリセット一覧を表示")
@click.option(
    "--list-watchlists", is_flag=True, help="利用可能なウォッチリスト一覧を表示"
)
def portfolio(
    symbols,
    period: str,
    investment_amount: float,
    max_stocks: int,
    risk_tolerance: float,
    export_csv: bool,
    symbols_file: str,
    preset: str,
    watchlist: str,
    list_presets: bool,
    list_watchlists: bool,
) -> None:
    """
    複数銘柄のポートフォリオ分析を実行する

    SYMBOLS: 分析する銘柄リスト（例：AAPL MSFT GOOGL AMZN TSLA）
    または --preset, --watchlist, --symbols-file オプションを使用
    """
    try:
        config = get_config()

        # リスト表示オプションの処理
        if list_presets:
            preset_manager = PresetManager()
            presets = preset_manager.list_presets()
            click.echo("🎯 利用可能なプリセット:")
            for preset_name in sorted(presets):
                info = preset_manager.get_preset_info(preset_name)
                click.echo(
                    f"  {preset_name}: {info['description']} ({info['symbol_count']}銘柄)"
                )
            return

        if list_watchlists:
            watchlists = config.list_watchlists()
            click.echo("📋 利用可能なウォッチリスト:")
            if watchlists:
                for wl_name in sorted(watchlists):
                    wl_symbols = config.get_watchlist(wl_name)
                    click.echo(f"  {wl_name}: {len(wl_symbols)}銘柄")
            else:
                click.echo("  ウォッチリストが設定されていません")
            return

        # 銘柄リストの決定
        final_symbols = []

        if symbols_file:
            click.echo(f"📄 ファイルから銘柄を読み込み: {symbols_file}")
            final_symbols = config.load_symbols_from_file(symbols_file)
        elif preset:
            click.echo(f"🎯 プリセットから銘柄を読み込み: {preset}")
            final_symbols = get_preset_symbols(preset)
        elif watchlist:
            click.echo(f"📋 ウォッチリストから銘柄を読み込み: {watchlist}")
            final_symbols = config.get_watchlist(watchlist)
        elif symbols:
            final_symbols = list(symbols)
        else:
            # デフォルト銘柄を使用
            default_symbols = config.get(
                "default_symbols", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            )
            click.echo("ℹ️  銘柄が指定されていません。デフォルト銘柄を使用します")
            final_symbols = default_symbols

        if not final_symbols:
            click.echo("❌ 分析する銘柄がありません")
            return

        # 設定値の決定（コマンドライン引数 > 設定ファイル > デフォルト）
        period = period or config.get("general.default_period", "1y")
        investment_amount = investment_amount or config.get(
            "general.default_investment_amount", 100000.0
        )
        max_stocks = max_stocks or config.get("general.default_max_stocks", 10)
        risk_tolerance = risk_tolerance or config.get(
            "general.default_risk_tolerance", 0.3
        )
        export_csv = export_csv or config.get("general.auto_export_csv", False)

        logger.info(f"CLI: ポートフォリオ分析開始 - {len(final_symbols)}銘柄")

        click.echo("🎯 ポートフォリオ分析を開始します")
        click.echo(
            f"対象銘柄: {', '.join(final_symbols[:10])}"
            + ("..." if len(final_symbols) > 10 else "")
        )
        click.echo(f"総銘柄数: {len(final_symbols)}")
        click.echo(f"投資期間: {period}")
        click.echo(f"総投資金額: ${investment_amount:,.0f}")
        click.echo(f"最大銘柄数: {max_stocks}")

        # ポートフォリオ設定
        config = PortfolioConfig(
            max_stocks=max_stocks,
            investment_amount=investment_amount,
            risk_tolerance=risk_tolerance,
        )

        # 各銘柄の詳細分析を実行
        click.echo("\n📊 各銘柄の分析中...")
        analysis_results = {}

        for symbol in final_symbols:
            try:
                click.echo(f"  • {symbol} を分析中...", nl=False)

                # データ取得
                stock_data = get_stock_data(symbol, period)

                # 分析結果を作成（テクニカル分析ベース）
                current_price = stock_data["Close"].iloc[-1]
                indicators = calculate_all_indicators(stock_data)

                # RSIベースの予測（簡易版）
                rsi = indicators["rsi"]
                bb_position = indicators["bb_position"]
                macd = indicators["macd"]
                volume_ratio = indicators["volume_ratio"]

                # 簡易予測ロジック
                if rsi < 30:  # 売られすぎ
                    return_5d = 3.0 + (30 - rsi) * 0.2
                    return_30d = 8.0 + (30 - rsi) * 0.5
                elif rsi > 70:  # 買われすぎ
                    return_5d = -2.0 - (rsi - 70) * 0.1
                    return_30d = -5.0 - (rsi - 70) * 0.3
                else:  # 中間
                    return_5d = 1.5 + macd * 2.0
                    return_30d = 5.0 + macd * 3.0

                # ボリンジャーバンドとMACD調整
                if bb_position > 0.8:
                    return_5d -= 1.0
                    return_30d -= 2.0
                elif bb_position < 0.2:
                    return_5d += 1.0
                    return_30d += 2.0

                # 出来高調整
                if volume_ratio > 1.5:
                    return_5d *= 1.2
                    return_30d *= 1.1

                # 投資スコア計算
                investment_score = min(
                    100,
                    max(0, 50 + return_5d * 8 + return_30d * 3 - abs(rsi - 50) * 0.3),
                )

                # リスクスコア計算
                volatility_risk = abs(bb_position - 0.5) * 100
                rsi_risk = (
                    max(abs(rsi - 30), abs(rsi - 70))
                    if rsi < 30 or rsi > 70
                    else abs(rsi - 50)
                )
                risk_score = min(100, max(0, volatility_risk + rsi_risk))

                # 推奨度決定
                if investment_score >= 75 and risk_score < 40:
                    recommendation = "強い買い"
                elif investment_score >= 60 and risk_score < 60:
                    recommendation = "買い"
                elif investment_score >= 40:
                    recommendation = "ホールド"
                else:
                    recommendation = "売り"

                # 擬似的な AnalysisResult 構造体を作成
                from types import SimpleNamespace

                analysis_result = SimpleNamespace(
                    symbol=symbol,
                    current_price=current_price,
                    investment_score=investment_score,
                    risk_score=risk_score,
                    recommendation=recommendation,
                    predictions={"return_5d": return_5d, "return_30d": return_30d},
                    technical_indicators=indicators,
                )

                analysis_results[symbol] = analysis_result
                click.echo(" ✅")

            except Exception as e:
                click.echo(f" ❌ エラー: {e}")
                logger.warning(f"銘柄 {symbol} の分析に失敗: {e}")

        if not analysis_results:
            click.echo("❌ 分析できる銘柄がありませんでした")
            return

        click.echo("\n🎯 ポートフォリオ最適化中...")

        # ポートフォリオ分析実行
        portfolio_analyzer = PortfolioAnalyzer(config)
        portfolio_result = portfolio_analyzer.analyze_multiple_stocks(
            list(analysis_results.keys()), analysis_results, period
        )

        # 結果表示
        click.echo("\n📈 ポートフォリオ分析結果:")
        click.echo("=" * 50)

        # サマリー表示
        summary = portfolio_result["analysis_summary"]
        for key, value in summary.items():
            click.echo(f"{key}: {value}")

        # ポートフォリオ構成表示
        click.echo("\n💰 推奨ポートフォリオ構成:")
        portfolio_stocks = portfolio_result["portfolio_stocks"]
        for stock in portfolio_stocks:
            weight_pct = stock.weight * 100
            click.echo(
                f"  {stock.symbol}: {weight_pct:.1f}% (${stock.allocation_amount:,.0f}) "
                f"- {stock.recommendation} (スコア: {stock.investment_score:.1f})"
            )

        # 推奨アクション表示
        recommendations = portfolio_result["recommendations"]
        click.echo(f"\n🎯 推奨アクション: {recommendations['action']}")
        click.echo(f"リスク評価: {recommendations['risk_assessment']}")
        if recommendations["reasoning"]:
            click.echo("理由:")
            for reason in recommendations["reasoning"]:
                click.echo(f"  • {reason}")

        # CSV出力
        if export_csv:
            click.echo("\n💾 CSV出力中...")
            try:
                exporter = JapaneseCsvExporter()

                # 個別銘柄のCSV出力
                output_files = []
                for symbol, result in analysis_results.items():
                    # 簡易データ準備
                    data = get_stock_data(symbol, "5d")  # 最新5日分

                    output_file = exporter.export_analysis_to_csv(
                        data=data, analysis_result=result, symbol=symbol
                    )
                    output_files.append(output_file)

                # ポートフォリオサマリーのCSV出力
                summary_file = exporter.export_portfolio_summary(
                    portfolio_result, analysis_results
                )
                output_files.append(summary_file)

                click.echo("CSV出力完了:")
                for file in output_files:
                    click.echo(f"  📄 {file}")

            except Exception as e:
                click.echo(f"❌ CSV出力エラー: {e}")
                logger.error(f"CSV出力エラー: {e}")

        logger.info(f"CLI: ポートフォリオ分析完了 - {len(portfolio_stocks)}銘柄選択")

    except Exception as e:
        logger.error(f"CLI: ポートフォリオ分析エラー: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--period",
    default=None,
    type=click.Choice(
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    ),
    help="データ取得期間",
)
@click.option("--sort-by", default="investment_score", help="ソート基準")
@click.option("--export-csv", is_flag=True, help="結果をCSVファイルにエクスポート")
@click.option(
    "--symbols-file", type=str, help="銘柄リストファイル (.txt, .csv, .json, .yaml)"
)
@click.option(
    "--preset", type=str, help="プリセット銘柄グループ (例: tech-giants, sp500-top20)"
)
@click.option("--watchlist", type=str, help="設定ファイルのウォッチリスト名")
def compare_advanced(
    symbols,
    period: str,
    sort_by: str,
    export_csv: bool,
    symbols_file: str,
    preset: str,
    watchlist: str,
) -> None:
    """
    複数銘柄の詳細比較を実行する

    SYMBOLS: 比較する銘柄リスト（例：AAPL MSFT GOOGL）
    または --preset, --watchlist, --symbols-file オプションを使用
    """
    try:
        config = get_config()

        # 銘柄リストの決定
        final_symbols = []

        if symbols_file:
            click.echo(f"📄 ファイルから銘柄を読み込み: {symbols_file}")
            final_symbols = config.load_symbols_from_file(symbols_file)
        elif preset:
            click.echo(f"🎯 プリセットから銘柄を読み込み: {preset}")
            final_symbols = get_preset_symbols(preset)
        elif watchlist:
            click.echo(f"📋 ウォッチリストから銘柄を読み込み: {watchlist}")
            final_symbols = config.get_watchlist(watchlist)
        elif symbols:
            final_symbols = list(symbols)
        else:
            # デフォルト銘柄を使用
            default_symbols = config.get(
                "default_symbols", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            )
            click.echo("ℹ️  銘柄が指定されていません。デフォルト銘柄を使用します")
            final_symbols = default_symbols

        if not final_symbols:
            click.echo("❌ 比較する銘柄がありません")
            return

        # 設定値の決定
        period = period or config.get("general.default_period", "1y")
        export_csv = export_csv or config.get("general.auto_export_csv", False)

        logger.info(f"CLI: 詳細比較開始 - {len(final_symbols)}銘柄")

        click.echo(f"📊 {len(final_symbols)}銘柄の詳細比較を開始します")
        click.echo(
            f"対象銘柄: {', '.join(final_symbols[:10])}"
            + ("..." if len(final_symbols) > 10 else "")
        )

        # 各銘柄の分析
        analysis_results = {}

        for symbol in final_symbols:
            try:
                click.echo(f"  • {symbol} を分析中...", nl=False)

                # データ取得と分析（portfolioコマンドと同様の処理）
                stock_data = get_stock_data(symbol, period)
                current_price = stock_data["Close"].iloc[-1]
                indicators = calculate_all_indicators(stock_data)

                # RSIベースの予測（簡易版）
                rsi = indicators["rsi"]
                bb_position = indicators["bb_position"]
                macd = indicators["macd"]
                volume_ratio = indicators["volume_ratio"]

                # 簡易予測ロジック
                if rsi < 30:  # 売られすぎ
                    return_5d = 3.0 + (30 - rsi) * 0.2
                    return_30d = 8.0 + (30 - rsi) * 0.5
                elif rsi > 70:  # 買われすぎ
                    return_5d = -2.0 - (rsi - 70) * 0.1
                    return_30d = -5.0 - (rsi - 70) * 0.3
                else:  # 中間
                    return_5d = 1.5 + macd * 2.0
                    return_30d = 5.0 + macd * 3.0

                # ボリンジャーバンドとMACD調整
                if bb_position > 0.8:
                    return_5d -= 1.0
                    return_30d -= 2.0
                elif bb_position < 0.2:
                    return_5d += 1.0
                    return_30d += 2.0

                # 出来高調整
                if volume_ratio > 1.5:
                    return_5d *= 1.2
                    return_30d *= 1.1

                # 投資スコア計算
                investment_score = min(
                    100,
                    max(0, 50 + return_5d * 8 + return_30d * 3 - abs(rsi - 50) * 0.3),
                )

                # リスクスコア計算
                volatility_risk = abs(bb_position - 0.5) * 100
                rsi_risk = (
                    max(abs(rsi - 30), abs(rsi - 70))
                    if rsi < 30 or rsi > 70
                    else abs(rsi - 50)
                )
                risk_score = min(100, max(0, volatility_risk + rsi_risk))

                # 推奨度決定
                if investment_score >= 75 and risk_score < 40:
                    recommendation = "強い買い"
                elif investment_score >= 60 and risk_score < 60:
                    recommendation = "買い"
                elif investment_score >= 40:
                    recommendation = "ホールド"
                else:
                    recommendation = "売り"

                from types import SimpleNamespace

                analysis_result = SimpleNamespace(
                    symbol=symbol,
                    current_price=current_price,
                    investment_score=investment_score,
                    risk_score=risk_score,
                    recommendation=recommendation,
                    predictions={"return_5d": return_5d, "return_30d": return_30d},
                    technical_indicators=indicators,
                )

                analysis_results[symbol] = analysis_result
                click.echo(" ✅")

            except Exception as e:
                click.echo(f" ❌ エラー: {e}")
                logger.warning(f"銘柄 {symbol} の分析に失敗: {e}")

        if not analysis_results:
            click.echo("❌ 分析できる銘柄がありませんでした")
            return

        # 比較テーブル作成
        comparison_df = compare_stocks(analysis_results, sort_by=sort_by)

        # 結果表示
        click.echo(f"\n📊 銘柄比較結果 (並び順: {sort_by}):")
        click.echo("=" * 80)

        # ヘッダー
        click.echo(
            f"{'銘柄':<8} {'価格($)':<10} {'投資ｽｺｱ':<8} {'ﾘｽｸｽｺｱ':<8} {'推奨':<8} {'5日予測%':<10} {'30日予測%':<10}"
        )
        click.echo("-" * 80)

        # データ行
        for _, row in comparison_df.iterrows():
            click.echo(
                f"{row['銘柄']:<8} "
                f"{row['現在価格']:<10.2f} "
                f"{row['投資スコア']:<8.1f} "
                f"{row['リスクスコア']:<8.1f} "
                f"{row['推奨度']:<8} "
                f"{row['5日後リターン予測']:<10.1f} "
                f"{row['30日後リターン予測']:<10.1f}"
            )

        # トップ3表示
        click.echo("\n🏆 投資スコア上位3銘柄:")
        top_3 = comparison_df.nlargest(3, "投資スコア")
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            click.echo(
                f"  {emoji} {row['銘柄']}: スコア {row['投資スコア']:.1f} - {row['推奨度']}"
            )

        # CSV出力
        if export_csv:
            click.echo("\n💾 CSV出力中...")
            try:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = f"stock_comparison_{timestamp}.csv"
                comparison_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
                click.echo(f"  📄 {csv_file}")

            except Exception as e:
                click.echo(f"❌ CSV出力エラー: {e}")

        logger.info(f"CLI: 詳細比較完了 - {len(analysis_results)}銘柄")

    except Exception as e:
        logger.error(f"CLI: 詳細比較エラー: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.option("--show", is_flag=True, help="現在の設定を表示")
@click.option("--init", is_flag=True, help="デフォルト設定ファイルを作成")
@click.option(
    "--set", "set_key", type=str, help="設定値を変更 (例: general.default_period)"
)
@click.option("--value", type=str, help="設定する値")
@click.option("--add-watchlist", type=str, help="新しいウォッチリストを追加")
@click.option("--watchlist-symbols", type=str, help="ウォッチリスト銘柄 (カンマ区切り)")
def config(
    show: bool,
    init: bool,
    set_key: str,
    value: str,
    add_watchlist: str,
    watchlist_symbols: str,
) -> None:
    """
    設定ファイルの管理
    """
    try:
        config_manager = get_config()

        if init:
            click.echo("🔧 デフォルト設定ファイルを作成しています...")
            # 既存の設定を再作成
            config_manager._create_default_config()
            click.echo(f"✅ 設定ファイルを作成しました: {config_manager.config_file}")
            return

        if show:
            click.echo("⚙️  現在の設定:")
            click.echo("=" * 50)

            # 主要設定を表示
            sections = ["general", "analysis", "portfolio", "output"]
            for section in sections:
                click.echo(f"\n[{section}]")
                section_data = config_manager.get(section, {})
                if isinstance(section_data, dict):
                    for config_key, config_value in section_data.items():
                        click.echo(f"  {config_key} = {config_value}")

            # ウォッチリスト表示
            click.echo("\n[watchlists]")
            watchlists = config_manager.get("watchlists", {})
            for name, symbols in watchlists.items():
                symbols_str = ", ".join(symbols[:5])
                if len(symbols) > 5:
                    symbols_str += f"... ({len(symbols)}銘柄)"
                click.echo(f"  {name} = [{symbols_str}]")

            click.echo(f"\n設定ファイル: {config_manager.config_file}")
            return

        if set_key and value:
            click.echo(f"🔧 設定を変更: {set_key} = {value}")

            # 型変換の試行
            parsed_value: Any = value
            try:
                # 数値の変換を試行
                if value.lower() in ["true", "false"]:
                    parsed_value = value.lower() == "true"
                elif value.replace(".", "").replace("-", "").isdigit():
                    parsed_value = float(value) if "." in value else int(value)
                # リストの変換を試行
                elif value.startswith("[") and value.endswith("]"):
                    import ast

                    parsed_value = ast.literal_eval(value)
            except ValueError:
                pass  # 文字列のまま使用

            config_manager.set(set_key, parsed_value)
            config_manager.save_config()
            click.echo("✅ 設定を保存しました")
            return

        if add_watchlist and watchlist_symbols:
            click.echo(f"📋 ウォッチリスト '{add_watchlist}' を追加")
            symbols = [s.strip().upper() for s in watchlist_symbols.split(",")]
            config_manager.add_watchlist(add_watchlist, symbols)
            config_manager.save_config()
            click.echo(f"✅ {len(symbols)}銘柄のウォッチリストを追加しました")
            return

        # デフォルト：設定表示
        click.echo("⚙️  設定管理コマンド")
        click.echo("利用可能なオプション:")
        click.echo("  --show                 現在の設定を表示")
        click.echo("  --init                 デフォルト設定ファイルを作成")
        click.echo("  --set KEY --value VAL  設定値を変更")
        click.echo(
            "  --add-watchlist NAME --watchlist-symbols 'SYM1,SYM2'  ウォッチリスト追加"
        )

    except Exception as e:
        logger.error(f"CLI: 設定管理エラー: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("symbol", type=str)
@click.option(
    "--investment-date",
    required=True,
    type=str,
    help="投資日（例: 2024-07-01）",
)
@click.option(
    "--validation-date",
    required=True,
    type=str,
    help="検証日（例: 2024-08-25）",
)
@click.option(
    "--training-months",
    default=24,
    type=int,
    help="訓練期間（投資日から遡る月数）",
)
@click.option(
    "--prediction-type",
    default="direction",
    type=click.Choice(["direction", "return"]),
    help="予測タイプ（direction: 上昇/下降, return: リターン率）",
)
def backtest(
    symbol: str,
    investment_date: str,
    validation_date: str,
    training_months: int,
    prediction_type: str,
) -> None:
    """
    過去データを使った予測精度シミュレーション

    指定した投資日時点での予測が、検証日時点で正しかったかを検証します。
    データリーケージを防ぐため、投資日以降のデータは一切使用しません。

    SYMBOL: 株式シンボル（例：AAPL, MSFT, GOOGL）
    """
    try:
        logger.info(f"CLI: バックテスト開始 - {symbol}")

        click.echo("🎯 予測精度シミュレーションを実行中...")
        click.echo(f"銘柄: {symbol}")
        click.echo(f"投資日: {investment_date}")
        click.echo(f"検証日: {validation_date}")
        click.echo(f"訓練期間: {training_months}ヶ月")
        click.echo(f"予測タイプ: {prediction_type}")

        # BacktestSimulatorを初期化
        simulator = BacktestSimulator()

        # シミュレーション実行
        result = simulator.run_point_in_time_simulation(
            symbol=symbol,
            investment_date=investment_date,
            validation_date=validation_date,
            training_period_months=training_months,
            prediction_type=prediction_type,
        )

        # 結果表示
        click.echo("\n📊 シミュレーション結果:")
        click.echo("=" * 60)

        # 基本情報
        click.echo(f"銘柄: {result['symbol']}")
        click.echo(
            f"投資期間: {result['investment_date']} → {result['validation_date']}"
        )
        click.echo(f"予測期間: {result['prediction_days']}日")
        click.echo(f"予測タイプ: {result['prediction_type']}")

        # 予測結果
        click.echo("\n🔮 予測結果:")
        prediction = result["prediction"]
        if prediction_type == "direction":
            confidence_pct = prediction["confidence"] * 100
            click.echo(f"予測方向: {prediction['predicted_direction']}")
            click.echo(f"信頼度: {confidence_pct:.1f}%")
        else:
            click.echo(f"予測リターン: {prediction['predicted_value']:.2f}%")

        # 実際の結果
        click.echo("\n📈 実際の結果:")
        actual = result["actual"]
        click.echo(f"投資価格: ${actual['investment_price']:.2f}")
        click.echo(f"検証価格: ${actual['validation_price']:.2f}")
        click.echo(f"実際リターン: {actual['actual_return']:.2f}%")
        click.echo(f"実際方向: {actual['actual_direction']}")

        # 評価結果
        click.echo("\n🎯 評価結果:")
        accuracy = result.get("accuracy", False)
        accuracy_symbol = "✅ 正解" if accuracy else "❌ 不正解"
        click.echo(f"予測精度: {accuracy_symbol}")
        click.echo(f"信頼性スコア: {result['confidence_score']:.1f}/100")
        click.echo(f"結果: {result['prediction_summary']}")

        # 推奨アクション
        confidence_score = result["confidence_score"]
        if confidence_score >= 80:
            recommendation = "🔥 このモデルは高精度です。実戦投入を検討できます。"
        elif confidence_score >= 60:
            recommendation = "⚡ モデル精度は中程度です。追加検証を推奨します。"
        elif confidence_score >= 40:
            recommendation = "⚠️  モデル精度が低めです。パラメータ調整が必要です。"
        else:
            recommendation = "🚨 モデル精度が不十分です。再設計を検討してください。"

        click.echo(f"\n💡 推奨: {recommendation}")

        logger.info(f"CLI: バックテスト完了 - {symbol}, 精度: {confidence_score:.1f}")

    except Exception as e:
        logger.error(f"CLI: バックテストエラー - {symbol}: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("symbol", type=str)
@click.option(
    "--prediction-days",
    default=30,
    type=int,
    help="予測日数（例: 30, 60）",
)
@click.option(
    "--training-months",
    default=24,
    type=int,
    help="訓練期間（現在から遡る月数）",
)
@click.option(
    "--prediction-type",
    default="direction",
    type=click.Choice(["direction", "return"]),
    help="予測タイプ（direction: 上昇/下降, return: リターン率）",
)
def predict(
    symbol: str,
    prediction_days: int,
    training_months: int,
    prediction_type: str,
) -> None:
    """
    現在データを使った未来予測

    最新の株価データを使って未来の価格動向を予測します。
    実際の検証は予測期間後に可能になります。

    SYMBOL: 株式シンボル（例：AAPL, MSFT, GOOGL）
    """
    try:
        from datetime import datetime, timedelta

        import pandas as pd

        logger.info(f"CLI: 未来予測開始 - {symbol}")

        # 現在日時を取得
        today = datetime.now().strftime("%Y-%m-%d")
        prediction_target_date = (
            datetime.now() + timedelta(days=prediction_days)
        ).strftime("%Y-%m-%d")

        click.echo("🔮 未来予測を実行中...")
        click.echo(f"銘柄: {symbol}")
        click.echo(f"予測開始日: {today}")
        click.echo(f"予測対象日: {prediction_target_date}")
        click.echo(f"予測期間: {prediction_days}日後")
        click.echo(f"訓練期間: {training_months}ヶ月")
        click.echo(f"予測タイプ: {prediction_type}")

        # 現在価格を取得
        recent_data = get_stock_data(symbol, "5d")
        current_price = recent_data["Close"].iloc[-1]
        latest_date = recent_data.index[-1].strftime("%Y-%m-%d")

        click.echo("\n📊 現在の状況:")
        click.echo(f"最新日付: {latest_date}")
        click.echo(f"現在価格: ${current_price:.2f}")

        # 訓練データ期間を計算
        training_start = datetime.now() - timedelta(days=training_months * 30)
        training_start_str = training_start.strftime("%Y-%m-%d")

        # 予測に使用する訓練データを取得
        training_data = get_stock_data(symbol, f"{training_months}mo")

        # 特徴量エンジニアリング
        from ..analysis.features import FeatureEngineering

        feature_engineer = FeatureEngineering()
        features = feature_engineer.create_features(training_data)

        # 目的変数作成（予測用）
        def create_prediction_targets(data, prediction_days, prediction_type):
            targets = pd.DataFrame(index=data.index)

            if prediction_type == "direction":
                # 方向性予測: 上昇=1, 下降=0
                price_change = (
                    data["Close"].pct_change(prediction_days).shift(-prediction_days)
                )
                targets[f"direction_{prediction_days}d"] = (price_change > 0).astype(
                    int
                )

            elif prediction_type == "return":
                # リターン予測: パーセント変化
                targets[f"return_{prediction_days}d"] = (
                    data["Close"].pct_change(prediction_days).shift(-prediction_days)
                    * 100
                )

            # NaNを除去
            targets = targets.dropna()
            return targets

        targets = create_prediction_targets(
            training_data, prediction_days, prediction_type
        )

        # モデル訓練
        from ..ml.lightgbm_predictor import LightGBMStockPredictor

        predictor = LightGBMStockPredictor(f"predict_model_{prediction_days}d")

        target_columns = [f"{prediction_type}_{prediction_days}d"]
        predictor.train_model(
            features, targets, target_columns=target_columns, n_splits=3
        )

        # 最新データで予測実行
        latest_features = feature_engineer.create_features(training_data.tail(1))
        predictions = predictor.predict(latest_features, target_columns)

        prediction_value = predictions[f"{prediction_type}_{prediction_days}d"][0]

        # 結果表示
        click.echo(f"\n🔮 {prediction_days}日後の予測結果:")
        click.echo("=" * 60)

        if prediction_type == "direction":
            predicted_direction = "上昇" if prediction_value > 0.5 else "下降"
            confidence = abs(prediction_value - 0.5) * 2  # 0-1の信頼度
            confidence_pct = confidence * 100

            click.echo(f"予測方向: {predicted_direction}")
            click.echo(f"信頼度: {confidence_pct:.1f}%")
            click.echo(f"予測スコア: {prediction_value:.3f}")

            # 価格目標の推定
            if predicted_direction == "上昇":
                estimated_return = 2 + confidence * 5  # 簡易推定
                target_price = current_price * (1 + estimated_return / 100)
                click.echo(f"推定リターン: +{estimated_return:.1f}%")
            else:
                estimated_return = 2 + confidence * 5
                target_price = current_price * (1 - estimated_return / 100)
                click.echo(f"推定リターン: -{estimated_return:.1f}%")

            click.echo(f"目標価格: ${target_price:.2f}")

        else:  # return prediction
            click.echo(f"予測リターン: {prediction_value:.2f}%")
            target_price = current_price * (1 + prediction_value / 100)
            click.echo(f"目標価格: ${target_price:.2f}")

        # 投資判断
        click.echo("\n💡 投資判断:")
        if prediction_type == "direction":
            if predicted_direction == "上昇" and confidence > 0.6:
                recommendation = "🚀 強い買い推奨"
            elif predicted_direction == "上昇" and confidence > 0.3:
                recommendation = "📈 買い検討"
            elif predicted_direction == "下降" and confidence > 0.6:
                recommendation = "🔻 売り/回避推奨"
            elif predicted_direction == "下降" and confidence > 0.3:
                recommendation = "📉 慎重検討"
            else:
                recommendation = "➡️ 中立・様子見"
        elif prediction_value > 5:
            recommendation = "🚀 強い買い推奨"
        elif prediction_value > 2:
            recommendation = "📈 買い検討"
        elif prediction_value < -5:
            recommendation = "🔻 売り/回避推奨"
        elif prediction_value < -2:
            recommendation = "📉 慎重検討"
        else:
            recommendation = "➡️ 中立・様子見"

        click.echo(f"{recommendation}")

        # 注意事項
        click.echo("\n⚠️  注意事項:")
        click.echo("• この予測は過去データに基づく推定です")
        click.echo("• 市場の突発的な変動は予測できません")
        click.echo("• 投資判断は複数の情報を総合して行ってください")
        click.echo(f"• {prediction_target_date}に実際の結果を確認してください")

        # 検証コマンドの提案
        validation_cmd = (
            f"uv run python -m stock_analyzer.cli.main backtest {symbol} "
            f"--investment-date {today} --validation-date {prediction_target_date}"
        )
        click.echo(f"\n🔍 {prediction_target_date}以降に予測精度を検証するコマンド:")
        click.echo(f"{validation_cmd}")

        logger.info(f"CLI: 未来予測完了 - {symbol}, 予測: {prediction_value:.3f}")

    except Exception as e:
        logger.error(f"CLI: 未来予測エラー - {symbol}: {e}")
        click.echo(f"❌ エラー: {e}", err=True)
        raise click.ClickException(str(e)) from e


# CLIのエントリーポイント
if __name__ == "__main__":
    # 直接実行時のログ設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    cli()
