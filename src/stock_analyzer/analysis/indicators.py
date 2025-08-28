"""テクニカル指標を計算するモジュール（学習用）"""

import logging
from typing import Any, Dict

import pandas as pd
import ta  # テクニカル分析ライブラリ


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="indicators")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    単純移動平均（SMA: Simple Moving Average）を計算

    Args:
        data (pd.DataFrame): 株価データ
        window (int): 期間（例：20日移動平均なら20）

    Returns:
        pd.Series: SMA値

    Example:
        >>> sma_20 = calculate_sma(data, 20)
        >>> print(f"最新のSMA20: {sma_20.iloc[-1]:.2f}")
    """
    logger.debug(f"SMA計算開始: 期間={window}日")
    sma = data["Close"].rolling(window=window).mean()
    logger.debug(f"SMA計算完了: {len(sma)}件")
    return pd.Series(sma)  # Ensure return type is Series


def calculate_ema(data: pd.DataFrame, window: int = 12) -> pd.Series:
    """
    指数移動平均（EMA: Exponential Moving Average）を計算

    Args:
        data (pd.DataFrame): 株価データ
        window (int): 期間（例：12日EMAなら12）

    Returns:
        pd.Series: EMA値

    Note:
        EMAは直近のデータにより重みを置いた移動平均
        SMAより価格変動に敏感に反応する
    """
    logger.debug(f"EMA計算開始: 期間={window}日")
    ema = data["Close"].ewm(span=window, adjust=False).mean()
    logger.debug(f"EMA計算完了: {len(ema)}件")
    return pd.Series(ema)  # Ensure return type is Series


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    RSI（Relative Strength Index）を計算
    値が70以上で買われすぎ、30以下で売られすぎとされる

    Args:
        data (pd.DataFrame): 株価データ
        window (int): 計算期間（通常14日）

    Returns:
        pd.Series: RSI値（0-100の範囲）

    Example:
        >>> rsi = calculate_rsi(data, 14)
        >>> latest_rsi = rsi.iloc[-1]
        >>> if latest_rsi > 70:
        ...     print("買われすぎ状態")
        >>> elif latest_rsi < 30:
        ...     print("売られすぎ状態")
    """
    logger.debug(f"RSI計算開始: 期間={window}日")
    rsi = ta.momentum.RSIIndicator(data["Close"], window=window).rsi()
    logger.debug(f"RSI計算完了: {len(rsi)}件")
    return rsi


def calculate_macd(
    data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """
    MACD（Moving Average Convergence Divergence）を計算

    Args:
        data (pd.DataFrame): 株価データ
        fast (int): 短期EMA期間
        slow (int): 長期EMA期間
        signal (int): シグナル線の期間

    Returns:
        Dict[str, pd.Series]: MACD、シグナル線、ヒストグラム

    Note:
        - MACDがシグナル線を上抜けると買いシグナル
        - MACDがシグナル線を下抜けると売りシグナル
    """
    logger.debug(f"MACD計算開始: fast={fast}, slow={slow}, signal={signal}")

    macd_indicator = ta.trend.MACD(
        data["Close"], window_fast=fast, window_slow=slow, window_sign=signal
    )

    result = {
        "macd": macd_indicator.macd(),
        "macd_signal": macd_indicator.macd_signal(),
        "macd_histogram": macd_indicator.macd_diff(),
    }

    logger.debug("MACD計算完了")
    return result


def calculate_bollinger_bands(
    data: pd.DataFrame, window: int = 20, std: float = 2.0
) -> Dict[str, pd.Series]:
    """
    ボリンジャーバンド（Bollinger Bands）を計算

    Args:
        data (pd.DataFrame): 株価データ
        window (int): 移動平均期間
        std (float): 標準偏差の倍数

    Returns:
        Dict[str, pd.Series]: 上限線、中央線（SMA）、下限線

    Note:
        - 価格が上限線に近づくと買われすぎ
        - 価格が下限線に近づくと売られすぎ
    """
    logger.debug(f"ボリンジャーバンド計算開始: 期間={window}日, 標準偏差={std}σ")

    bb_indicator = ta.volatility.BollingerBands(
        data["Close"], window=window, window_dev=std
    )

    result = {
        "bb_upper": bb_indicator.bollinger_hband(),
        "bb_middle": bb_indicator.bollinger_mavg(),  # SMAと同じ
        "bb_lower": bb_indicator.bollinger_lband(),
    }

    logger.debug("ボリンジャーバンド計算完了")
    return result


def calculate_volume_indicators(
    data: pd.DataFrame, window: int = 20
) -> Dict[str, pd.Series]:
    """
    出来高関連指標を計算

    Args:
        data (pd.DataFrame): 株価データ（Volume列必須）
        window (int): 移動平均期間

    Returns:
        Dict[str, pd.Series]: 出来高移動平均、出来高レシオ
    """
    logger.debug(f"出来高指標計算開始: 期間={window}日")

    volume_sma = data["Volume"].rolling(window=window).mean()
    volume_ratio = data["Volume"] / volume_sma

    result = {"volume_sma": volume_sma, "volume_ratio": volume_ratio}

    logger.debug("出来高指標計算完了")
    return result


def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """
    すべてのテクニカル指標をまとめて計算

    Args:
        data (pd.DataFrame): 株価データ

    Returns:
        Dict[str, Any]: 各指標の最新値を含む辞書

    Example:
        >>> from stock_analyzer.data.fetchers import get_stock_data
        >>> data = get_stock_data("AAPL", "6mo")
        >>> indicators = calculate_all_indicators(data)
        >>> print(f"RSI: {indicators['rsi']:.1f}")
    """
    logger.info("全テクニカル指標の計算開始")

    indicators = {}

    # 移動平均線（短期〜長期）
    logger.debug("移動平均線を計算中...")
    indicators["sma_5"] = calculate_sma(data, 5).iloc[-1]
    indicators["sma_20"] = calculate_sma(data, 20).iloc[-1]
    indicators["sma_50"] = calculate_sma(data, 50).iloc[-1]
    indicators["sma_200"] = calculate_sma(data, 200).iloc[-1]

    # 指数移動平均線
    indicators["ema_12"] = calculate_ema(data, 12).iloc[-1]
    indicators["ema_26"] = calculate_ema(data, 26).iloc[-1]

    # モメンタム指標
    logger.debug("モメンタム指標を計算中...")
    indicators["rsi"] = calculate_rsi(data, 14).iloc[-1]

    # MACD
    macd_data = calculate_macd(data)
    indicators["macd"] = macd_data["macd"].iloc[-1]
    indicators["macd_signal"] = macd_data["macd_signal"].iloc[-1]
    indicators["macd_histogram"] = macd_data["macd_histogram"].iloc[-1]

    # ボリンジャーバンド
    logger.debug("ボリンジャーバンドを計算中...")
    bb_data = calculate_bollinger_bands(data)
    indicators["bb_upper"] = bb_data["bb_upper"].iloc[-1]
    indicators["bb_middle"] = bb_data["bb_middle"].iloc[-1]
    indicators["bb_lower"] = bb_data["bb_lower"].iloc[-1]

    # 出来高指標
    logger.debug("出来高指標を計算中...")
    volume_data = calculate_volume_indicators(data)
    indicators["volume_sma"] = volume_data["volume_sma"].iloc[-1]
    indicators["volume_ratio"] = volume_data["volume_ratio"].iloc[-1]

    # 価格変動率
    indicators["price_change_1d"] = (
        (data["Close"].iloc[-1] / data["Close"].iloc[-2]) - 1
    ) * 100
    indicators["price_change_5d"] = (
        ((data["Close"].iloc[-1] / data["Close"].iloc[-6]) - 1) * 100
        if len(data) > 5
        else 0
    )

    # 現在価格とボリンジャーバンドとの位置関係
    current_price = data["Close"].iloc[-1]
    bb_position = (current_price - indicators["bb_lower"]) / (
        indicators["bb_upper"] - indicators["bb_lower"]
    )
    indicators["bb_position"] = bb_position

    logger.info("全テクニカル指標の計算完了")

    # ログで主要指標を表示
    logger.info(
        f"主要指標 - RSI: {indicators['rsi']:.1f}, "
        f"SMA20: {indicators['sma_20']:.2f}, "
        f"MACD: {indicators['macd']:.3f}"
    )

    return indicators


def analyze_signals(indicators: Dict[str, Any]) -> Dict[str, str]:
    """
    テクニカル指標から売買シグナルを分析

    Args:
        indicators (Dict[str, Any]): calculate_all_indicators()の結果

    Returns:
        Dict[str, str]: 各指標のシグナル分析結果
    """
    logger.debug("シグナル分析開始")

    signals = {}

    # RSIシグナル
    rsi = indicators["rsi"]
    if rsi > 70:
        signals["rsi_signal"] = "売り（買われすぎ）"
    elif rsi < 30:
        signals["rsi_signal"] = "買い（売られすぎ）"
    else:
        signals["rsi_signal"] = "中立"

    # 移動平均シグナル（ゴールデンクロス・デッドクロス）
    if indicators["sma_5"] > indicators["sma_20"]:
        signals["sma_signal"] = "買い（短期が長期を上回る）"
    else:
        signals["sma_signal"] = "売り（短期が長期を下回る）"

    # MACDシグナル
    if indicators["macd"] > indicators["macd_signal"]:
        signals["macd_signal"] = "買い（MACDがシグナル上）"
    else:
        signals["macd_signal"] = "売り（MACDがシグナル下）"

    # ボリンジャーバンドシグナル
    bb_pos = indicators["bb_position"]
    if bb_pos > 0.8:
        signals["bb_signal"] = "売り（上限線付近）"
    elif bb_pos < 0.2:
        signals["bb_signal"] = "買い（下限線付近）"
    else:
        signals["bb_signal"] = "中立"

    # 出来高シグナル
    if indicators["volume_ratio"] > 1.5:
        signals["volume_signal"] = "注目（出来高急増）"
    else:
        signals["volume_signal"] = "通常"

    logger.debug("シグナル分析完了")
    return signals


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    # ロギング設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    # テスト用にAppleの株価データを取得
    print("=== テクニカル指標計算テスト ===")
    try:
        from ..data.fetchers import get_stock_data

        # 6ヶ月分のデータで十分な期間を確保
        data = get_stock_data("AAPL", "6mo")
        print(
            f"データ期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}"
        )

        # 全指標計算
        indicators = calculate_all_indicators(data)

        print("\n📊 テクニカル指標:")
        print(f"RSI (14日): {indicators['rsi']:.1f}")
        print(f"SMA (20日): ${indicators['sma_20']:.2f}")
        print(f"MACD: {indicators['macd']:.3f}")
        print(f"ボリンジャーバンド位置: {indicators['bb_position']:.2f} (0-1)")
        print(f"出来高比率: {indicators['volume_ratio']:.2f}")

        # シグナル分析
        signals = analyze_signals(indicators)
        print("\n📈 売買シグナル分析:")
        for signal_type, signal in signals.items():
            print(f"{signal_type}: {signal}")

    except ImportError:
        print("データ取得モジュールが見つかりません。")
        print("以下のコマンドで実行してください:")
        print("uv run python -m stock_analyzer.analysis.indicators")
    except Exception as e:
        print(f"エラー: {e}")
