"""Stock Analyzer CLI (Command Line Interface)"""

from typing import Any

import click

# テクニカル分析機能をインポート
from ..analysis.indicators import analyze_signals, calculate_all_indicators

# データ取得機能をインポート
from ..data.fetchers import get_company_info, get_stock_data


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
        raise click.ClickException(str(e))


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
        raise click.ClickException(str(e))


# CLIのエントリーポイント
if __name__ == "__main__":
    # 直接実行時のログ設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    cli()
