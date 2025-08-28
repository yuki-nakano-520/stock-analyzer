"""株価データを取得するモジュール（初心者向け実装例）"""

import logging
from typing import Any

import pandas as pd
import yfinance as yf

# 型定義は現在未使用（将来の機能拡張で使用予定）


# ロガーを遅延初期化で循環インポートを回避
def _get_logger() -> Any:
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="data_fetcher")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    株価データを取得する関数

    Args:
        symbol (str): 株式シンボル（例：'AAPL'）
        period (str): 取得期間（例：'1y', '6mo', '3mo', '1mo'）
                     有効な値: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

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


def get_company_info(symbol: str) -> dict[str, Any]:
    """
    会社の基本情報を取得する関数

    Args:
        symbol (str): 株式シンボル

    Returns:
        CompanyInfo: 会社情報
    """
    try:
        logger.debug(f"会社情報を取得開始: {symbol}")

        ticker = yf.Ticker(symbol)
        info = ticker.info

        # 必要な情報のみ抽出（型安全性を保証）
        company_info = {
            "symbol": symbol,
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "current_price": info.get("currentPrice", 0.0),
        }

        logger.debug(f"会社情報取得成功: {company_info['company_name']}")
        return company_info

    except Exception as e:
        logger.error(f"会社情報取得エラー: {symbol}, エラー: {e}")
        # 直接実行時は辞書を返す
        return {
            "symbol": symbol,
            "company_name": symbol,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": 0,
            "current_price": 0.0,
        }


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    # ロギングの基本設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    # 基本的な使い方
    print("=== Apple株価データ取得テスト ===")
    try:
        # yfinanceの正しい期間指定: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        apple_data = get_stock_data("AAPL", "6mo")
        print("Apple株価データ:")
        print(apple_data.head())

        print(f"\nデータ件数: {len(apple_data)}")
        print(f"最新価格: {apple_data['Close'].iloc[-1]:.2f}")

        # 会社情報も取得してみる
        print("\n=== Apple会社情報 ===")
        apple_info = get_company_info("AAPL")
        for key, value in apple_info.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
