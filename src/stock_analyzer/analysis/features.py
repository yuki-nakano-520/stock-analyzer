"""機械学習のための特徴量エンジニアリング（学習用）"""

import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

# テクニカル指標モジュールをインポート
from .indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_volume_indicators,
)


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="features")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


def create_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    価格ベースの特徴量を作成

    Args:
        data (pd.DataFrame): 株価データ（OHLCV）

    Returns:
        pd.DataFrame: 価格特徴量
    """
    logger.debug("価格特徴量の作成開始")

    features = pd.DataFrame(index=data.index)

    # 基本的な価格変動率
    features["price_change_1d"] = data["Close"].pct_change()
    features["price_change_3d"] = data["Close"].pct_change(periods=3)
    features["price_change_5d"] = data["Close"].pct_change(periods=5)
    features["price_change_10d"] = data["Close"].pct_change(periods=10)

    # 高値・安値との関係
    features["high_low_ratio"] = data["High"] / data["Low"]
    features["close_high_ratio"] = data["Close"] / data["High"]
    features["close_low_ratio"] = data["Close"] / data["Low"]

    # OHLC関連
    features["open_close_ratio"] = data["Open"] / data["Close"]
    features["body_ratio"] = abs(data["Close"] - data["Open"]) / (
        data["High"] - data["Low"]
    )

    # ボラティリティ（価格変動の大きさ）
    features["volatility_5d"] = data["Close"].rolling(5).std()
    features["volatility_10d"] = data["Close"].rolling(10).std()
    features["volatility_20d"] = data["Close"].rolling(20).std()

    # 価格レンジ
    features["price_range"] = (data["High"] - data["Low"]) / data["Close"]
    features["upper_shadow"] = (
        data["High"] - np.maximum(data["Open"], data["Close"])
    ) / data["Close"]
    features["lower_shadow"] = (
        np.minimum(data["Open"], data["Close"]) - data["Low"]
    ) / data["Close"]

    logger.debug(f"価格特徴量作成完了: {len(features.columns)}個の特徴量")
    return features


def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    テクニカル指標ベースの特徴量を作成

    Args:
        data (pd.DataFrame): 株価データ（OHLCV）

    Returns:
        pd.DataFrame: テクニカル特徴量
    """
    logger.debug("テクニカル特徴量の作成開始")

    features = pd.DataFrame(index=data.index)

    # 移動平均線系
    sma_5 = calculate_sma(data, 5)
    sma_10 = calculate_sma(data, 10)
    sma_20 = calculate_sma(data, 20)
    sma_50 = calculate_sma(data, 50)

    features["sma_5"] = sma_5
    features["sma_10"] = sma_10
    features["sma_20"] = sma_20
    features["sma_50"] = sma_50

    # 移動平均からの乖離率
    features["close_sma5_ratio"] = data["Close"] / sma_5
    features["close_sma10_ratio"] = data["Close"] / sma_10
    features["close_sma20_ratio"] = data["Close"] / sma_20
    features["close_sma50_ratio"] = data["Close"] / sma_50

    # 移動平均の傾き（トレンドの強さ）
    features["sma5_slope"] = sma_5.diff()
    features["sma20_slope"] = sma_20.diff()

    # 指数移動平均
    ema_12 = calculate_ema(data, 12)
    ema_26 = calculate_ema(data, 26)
    features["ema_12"] = ema_12
    features["ema_26"] = ema_26

    # RSI（複数期間）
    features["rsi_14"] = calculate_rsi(data, 14)
    features["rsi_21"] = calculate_rsi(data, 21)

    # MACD
    macd_data = calculate_macd(data)
    features["macd"] = macd_data["macd"]
    features["macd_signal"] = macd_data["macd_signal"]
    features["macd_histogram"] = macd_data["macd_histogram"]

    # ボリンジャーバンド
    bb_data = calculate_bollinger_bands(data, 20, 2.0)
    features["bb_upper"] = bb_data["bb_upper"]
    features["bb_lower"] = bb_data["bb_lower"]
    features["bb_width"] = bb_data["bb_upper"] - bb_data["bb_lower"]
    features["bb_position"] = (data["Close"] - bb_data["bb_lower"]) / (
        bb_data["bb_upper"] - bb_data["bb_lower"]
    )

    logger.debug(f"テクニカル特徴量作成完了: {len(features.columns)}個の特徴量")
    return features


def create_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    出来高ベースの特徴量を作成

    Args:
        data (pd.DataFrame): 株価データ（Volume列必須）

    Returns:
        pd.DataFrame: 出来高特徴量
    """
    logger.debug("出来高特徴量の作成開始")

    features = pd.DataFrame(index=data.index)

    # 基本的な出来高指標
    volume_data = calculate_volume_indicators(data, 20)
    features["volume_sma"] = volume_data["volume_sma"]
    features["volume_ratio"] = volume_data["volume_ratio"]

    # 出来高の変動率
    features["volume_change_1d"] = data["Volume"].pct_change()
    features["volume_change_5d"] = data["Volume"].pct_change(periods=5)

    # 価格と出来高の関係
    features["price_volume"] = data["Close"] * data["Volume"]
    features["volume_price_trend"] = (
        (data["Close"].diff() * data["Volume"]).rolling(5).sum()
    )

    # 出来高の標準化（Z-score）
    volume_mean = data["Volume"].rolling(20).mean()
    volume_std = data["Volume"].rolling(20).std()
    features["volume_zscore"] = (data["Volume"] - volume_mean) / volume_std

    logger.debug(f"出来高特徴量作成完了: {len(features.columns)}個の特徴量")
    return features


def create_pattern_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    価格パターンベースの特徴量を作成

    Args:
        data (pd.DataFrame): 株価データ（OHLCV）

    Returns:
        pd.DataFrame: パターン特徴量
    """
    logger.debug("パターン特徴量の作成開始")

    features = pd.DataFrame(index=data.index)

    # 高値・安値更新パターン
    features["new_high_20d"] = (data["High"] == data["High"].rolling(20).max()).astype(
        int
    )
    features["new_low_20d"] = (data["Low"] == data["Low"].rolling(20).min()).astype(int)

    # 連続上昇・下降日数
    price_direction = (data["Close"] > data["Close"].shift(1)).astype(int)
    features["consecutive_up"] = price_direction.groupby(
        (price_direction != price_direction.shift()).cumsum()
    ).cumsum()

    price_direction_down = (data["Close"] < data["Close"].shift(1)).astype(int)
    features["consecutive_down"] = price_direction_down.groupby(
        (price_direction_down != price_direction_down.shift()).cumsum()
    ).cumsum()

    # ギャップ（窓）
    features["gap_up"] = (
        (data["Open"] > data["Close"].shift(1))
        & (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)
        > 0.02
    ).astype(int)
    features["gap_down"] = (
        (data["Open"] < data["Close"].shift(1))
        & (data["Close"].shift(1) - data["Open"]) / data["Close"].shift(1)
        > 0.02
    ).astype(int)

    # サポート・レジスタンス
    rolling_max = data["High"].rolling(20).max()
    rolling_min = data["Low"].rolling(20).min()
    features["distance_to_resistance"] = (rolling_max - data["Close"]) / data["Close"]
    features["distance_to_support"] = (data["Close"] - rolling_min) / data["Close"]

    logger.debug(f"パターン特徴量作成完了: {len(features.columns)}個の特徴量")
    return features


def create_target_variables(
    data: pd.DataFrame, target_days: List[int] | None = None
) -> pd.DataFrame:
    """
    予測対象（目的変数）を作成

    Args:
        data (pd.DataFrame): 株価データ
        target_days (List[int]): 予測対象の日数

    Returns:
        pd.DataFrame: 目的変数
    """
    if target_days is None:
        target_days = [1, 5, 10, 30]
    logger.debug(f"目的変数の作成開始: {target_days}日後")

    targets = pd.DataFrame(index=data.index)

    for days in target_days:
        # 未来の価格
        future_price = data["Close"].shift(-days)
        current_price = data["Close"]

        # リターン率（%）
        targets[f"return_{days}d"] = ((future_price / current_price) - 1) * 100

        # 上昇・下降の方向（分類用）
        targets[f"direction_{days}d"] = (future_price > current_price).astype(int)

        # 大幅上昇フラグ（10%以上）
        targets[f"big_gain_{days}d"] = (targets[f"return_{days}d"] > 10).astype(int)

        # 大幅下落フラグ（-10%以下）
        targets[f"big_loss_{days}d"] = (targets[f"return_{days}d"] < -10).astype(int)

    logger.debug(f"目的変数作成完了: {len(targets.columns)}個の目的変数")
    return targets


def create_all_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    すべての特徴量と目的変数を作成

    Args:
        data (pd.DataFrame): 株価データ（OHLCV）

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (特徴量, 目的変数)
    """
    logger.info("全特徴量の作成開始")

    # 各種特徴量を作成
    price_features = create_price_features(data)
    technical_features = create_technical_features(data)
    volume_features = create_volume_features(data)
    pattern_features = create_pattern_features(data)

    # 特徴量をまとめる
    all_features = pd.concat(
        [price_features, technical_features, volume_features, pattern_features], axis=1
    )

    # 目的変数を作成
    targets = create_target_variables(data)

    logger.info(
        f"全特徴量作成完了: 特徴量{len(all_features.columns)}個, 目的変数{len(targets.columns)}個"
    )

    return all_features, targets


def clean_features(
    features: pd.DataFrame, targets: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    特徴量と目的変数をクリーニング（欠損値処理、無限値処理）

    Args:
        features (pd.DataFrame): 特徴量
        targets (pd.DataFrame): 目的変数

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: クリーニング済み(特徴量, 目的変数)
    """
    logger.debug("特徴量クリーニング開始")

    # 無限値をNaNに置換
    features = features.replace([np.inf, -np.inf], np.nan)
    targets = targets.replace([np.inf, -np.inf], np.nan)

    # クリーニング前の状態をログ
    total_values = len(features) * len(features.columns)
    nan_count_before = features.isnull().sum().sum()
    logger.debug(
        f"クリーニング前: 全{total_values}値中{nan_count_before}個のNaN ({nan_count_before / total_values * 100:.1f}%)"
    )

    # 欠損値が多すぎる列を削除（50%以上NaN）
    nan_ratio = features.isnull().sum() / len(features)
    high_nan_columns = nan_ratio[nan_ratio > 0.5].index
    if len(high_nan_columns) > 0:
        logger.warning(f"欠損値が多い列を削除: {list(high_nan_columns)}")
        features = features.drop(columns=high_nan_columns)

    # 残った欠損値を前方埋め→後方埋め
    features = features.ffill().bfill()
    targets = targets.ffill().bfill()

    # それでも残った欠損値は0で埋める
    features = features.fillna(0)
    targets = targets.fillna(0)

    # クリーニング後の状態をログ
    nan_count_after = features.isnull().sum().sum()
    logger.debug(f"クリーニング後: {nan_count_after}個のNaN")

    logger.debug("特徴量クリーニング完了")

    return features, targets


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    # ロギング設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    print("=== 特徴量エンジニアリングテスト ===")
    try:
        from ..data.fetchers import get_stock_data

        # 1年分のデータで十分な履歴を確保
        data = get_stock_data("AAPL", "1y")
        print(
            f"データ期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}"
        )

        # 全特徴量作成
        features, targets = create_all_features(data)
        print(f"\n📊 作成された特徴量: {len(features.columns)}個")
        print("特徴量の例:")
        for i, col in enumerate(features.columns[:10]):
            print(f"  {i + 1}. {col}")
        if len(features.columns) > 10:
            print(f"  ... 他{len(features.columns) - 10}個")

        print(f"\n🎯 目的変数: {len(targets.columns)}個")
        for col in targets.columns:
            print(f"  - {col}")

        # クリーニング
        clean_features_df, clean_targets_df = clean_features(features, targets)

        print("\n🧹 クリーニング後:")
        print(
            f"特徴量: {clean_features_df.shape} (欠損値: {clean_features_df.isnull().sum().sum()}個)"
        )
        print(
            f"目的変数: {clean_targets_df.shape} (欠損値: {clean_targets_df.isnull().sum().sum()}個)"
        )

        # 統計情報
        print("\n📈 サンプル統計 (最新10日):")
        latest_data = clean_features_df.tail(10)
        for col in ["price_change_1d", "rsi_14", "volume_ratio"]:
            if col in latest_data.columns:
                print(
                    f"{col}: 平均={latest_data[col].mean():.3f}, 標準偏差={latest_data[col].std():.3f}"
                )

    except ImportError:
        print("データ取得モジュールが見つかりません。")
        print("以下のコマンドで実行してください:")
        print("uv run python -m stock_analyzer.analysis.features")
    except Exception as e:
        print(f"エラー: {e}")
