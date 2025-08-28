"""株価データの可視化モジュール（学習用）"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # GUI不要のバックエンドを設定
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

# テクニカル指標とデータ取得をインポート
from .indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="visualizer")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()

# 日本語フォント設定とスタイル設定
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_price_chart(
    data: pd.DataFrame, symbol: str, save_path: Optional[str] = None
) -> None:
    """
    基本的な価格チャートを描画

    Args:
        data (pd.DataFrame): 株価データ（OHLCV）
        symbol (str): 銘柄シンボル
        save_path (Optional[str]): 保存先パス（Noneの場合は表示のみ）
    """
    logger.debug(f"価格チャート描画開始: {symbol}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])

    # 価格チャート
    ax1.plot(data.index, data["Close"], label="終値", linewidth=2, color="blue")
    ax1.plot(data.index, data["High"], alpha=0.3, color="green", linewidth=0.5)
    ax1.plot(data.index, data["Low"], alpha=0.3, color="red", linewidth=0.5)
    ax1.fill_between(data.index, data["Low"], data["High"], alpha=0.1, color="gray")

    ax1.set_title(f"{symbol} 株価チャート", fontsize=16, fontweight="bold")
    ax1.set_ylabel("価格 ($)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 出来高チャート
    colors = [
        "red" if close < open_ else "green"
        for close, open_ in zip(data["Close"], data["Open"], strict=False)
    ]
    ax2.bar(data.index, data["Volume"], color=colors, alpha=0.6)
    ax2.set_ylabel("出来高", fontsize=12)
    ax2.set_xlabel("日付", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # レイアウト調整
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"チャートを保存: {save_path}")
    else:
        plt.show()

    plt.close()
    logger.debug("価格チャート描画完了")


def plot_technical_analysis(
    data: pd.DataFrame, symbol: str, save_path: Optional[str] = None
) -> None:
    """
    テクニカル指標付きの詳細チャートを描画

    Args:
        data (pd.DataFrame): 株価データ（OHLCV）
        symbol (str): 銘柄シンボル
        save_path (Optional[str]): 保存先パス
    """
    logger.debug(f"テクニカル分析チャート描画開始: {symbol}")

    # テクニカル指標を計算
    sma_20 = calculate_sma(data, 20)
    sma_50 = calculate_sma(data, 50)
    rsi = calculate_rsi(data, 14)
    macd_data = calculate_macd(data)
    bb_data = calculate_bollinger_bands(data)

    # 4つのサブプロット作成
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[3, 1, 1, 1])

    # 1. 価格 + 移動平均 + ボリンジャーバンド
    ax1 = axes[0]
    ax1.plot(data.index, data["Close"], label="終値", linewidth=2, color="blue")
    ax1.plot(
        data.index, sma_20, label="SMA20", linewidth=1.5, color="orange", alpha=0.8
    )
    ax1.plot(data.index, sma_50, label="SMA50", linewidth=1.5, color="red", alpha=0.8)

    # ボリンジャーバンド
    ax1.plot(
        data.index,
        bb_data["bb_upper"],
        label="BB上限",
        linewidth=1,
        color="purple",
        alpha=0.6,
        linestyle="--",
    )
    ax1.plot(
        data.index,
        bb_data["bb_lower"],
        label="BB下限",
        linewidth=1,
        color="purple",
        alpha=0.6,
        linestyle="--",
    )
    ax1.fill_between(
        data.index,
        bb_data["bb_lower"],
        bb_data["bb_upper"],
        alpha=0.1,
        color="purple",
        label="ボリンジャーバンド",
    )

    ax1.set_title(f"{symbol} テクニカル分析チャート", fontsize=16, fontweight="bold")
    ax1.set_ylabel("価格 ($)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. 出来高
    ax2 = axes[1]
    volume_colors = [
        "red" if close < open_ else "green"
        for close, open_ in zip(data["Close"], data["Open"], strict=False)
    ]
    ax2.bar(data.index, data["Volume"], color=volume_colors, alpha=0.6)
    ax2.set_ylabel("出来高", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. RSI
    ax3 = axes[2]
    ax3.plot(data.index, rsi, label="RSI(14)", linewidth=2, color="purple")
    ax3.axhline(y=70, color="red", linestyle="--", alpha=0.7, label="買われすぎ(70)")
    ax3.axhline(y=30, color="green", linestyle="--", alpha=0.7, label="売られすぎ(30)")
    ax3.axhline(y=50, color="gray", linestyle="-", alpha=0.5, linewidth=0.5)
    ax3.fill_between(data.index, 30, 70, alpha=0.1, color="gray")
    ax3.set_ylabel("RSI", fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # 4. MACD
    ax4 = axes[3]
    ax4.plot(data.index, macd_data["macd"], label="MACD", linewidth=2, color="blue")
    ax4.plot(
        data.index,
        macd_data["macd_signal"],
        label="シグナル",
        linewidth=1.5,
        color="red",
    )
    ax4.bar(
        data.index,
        macd_data["macd_histogram"],
        label="ヒストグラム",
        alpha=0.6,
        color="gray",
    )
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    ax4.set_ylabel("MACD", fontsize=12)
    ax4.set_xlabel("日付", fontsize=12)
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    # レイアウト調整
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"テクニカル分析チャートを保存: {save_path}")
    else:
        plt.show()

    plt.close()
    logger.debug("テクニカル分析チャート描画完了")


def plot_correlation_heatmap(
    features: pd.DataFrame,
    target_col: str = "return_5d",
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    特徴量と目的変数の相関ヒートマップを描画

    Args:
        features (pd.DataFrame): 特徴量データ
        target_col (str): 目的変数のカラム名
        top_n (int): 表示する特徴量の数
        save_path (Optional[str]): 保存先パス
    """
    logger.debug(f"相関ヒートマップ描画開始: {target_col}")

    # 数値型の列のみを選択
    numeric_features = features.select_dtypes(include=[np.number])

    # 目的変数との相関を計算
    if target_col in numeric_features.columns:
        correlations = (
            numeric_features.corr()[target_col].abs().sort_values(ascending=False)
        )
        top_features = correlations.head(top_n).index

        # 相関行列を計算
        corr_matrix = numeric_features[top_features].corr()

        # ヒートマップ描画
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 上三角をマスク

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            f"特徴量相関ヒートマップ (vs {target_col})", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"相関ヒートマップを保存: {save_path}")
        else:
            plt.show()

        plt.close()
    else:
        logger.warning(f"目的変数 '{target_col}' が見つかりません")

    logger.debug("相関ヒートマップ描画完了")


def plot_feature_importance(
    features: List[str],
    importance: List[float],
    title: str = "特徴量重要度",
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    特徴量重要度を棒グラフで表示

    Args:
        features (List[str]): 特徴量名のリスト
        importance (List[float]): 重要度のリスト
        title (str): グラフタイトル
        top_n (int): 表示する特徴量の数
        save_path (Optional[str]): 保存先パス
    """
    logger.debug(f"特徴量重要度描画開始: {len(features)}個の特徴量")

    # 重要度でソート
    feature_importance = sorted(
        zip(features, importance, strict=False), key=lambda x: x[1], reverse=True
    )
    top_features = feature_importance[:top_n]

    names, values = zip(*top_features, strict=False)

    # 横棒グラフで描画
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(names))

    bars = plt.barh(y_pos, values, alpha=0.8)
    plt.yticks(y_pos, names)
    plt.xlabel("重要度", fontsize=12)
    plt.title(title, fontsize=16, fontweight="bold")

    # 値をバーに表示
    for i, (bar, value) in enumerate(zip(bars, values, strict=False)):
        plt.text(
            value + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.gca().invert_yaxis()  # 重要度が高い順に上から表示
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"特徴量重要度チャートを保存: {save_path}")
    else:
        plt.show()

    plt.close()
    logger.debug("特徴量重要度描画完了")


def plot_return_distribution(
    targets: pd.DataFrame,
    target_col: str = "return_5d",
    save_path: Optional[str] = None,
) -> None:
    """
    リターン分布をヒストグラムで表示

    Args:
        targets (pd.DataFrame): 目的変数データ
        target_col (str): リターンカラム名
        save_path (Optional[str]): 保存先パス
    """
    logger.debug(f"リターン分布描画開始: {target_col}")

    if target_col in targets.columns:
        returns = targets[target_col].dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ヒストグラム
        ax1.hist(returns, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(
            returns.mean(),
            color="red",
            linestyle="--",
            label=f"平均: {returns.mean():.2f}%",
        )
        ax1.axvline(
            returns.median(),
            color="green",
            linestyle="--",
            label=f"中央値: {returns.median():.2f}%",
        )
        ax1.set_xlabel("リターン (%)", fontsize=12)
        ax1.set_ylabel("頻度", fontsize=12)
        ax1.set_title(f"{target_col} 分布", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ボックスプロット
        ax2.boxplot(returns, vert=True)
        ax2.set_ylabel("リターン (%)", fontsize=12)
        ax2.set_title(f"{target_col} ボックスプロット", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # 統計情報を表示
        stats_text = (
            f"統計情報:\n"
            f"平均: {returns.mean():.2f}%\n"
            f"標準偏差: {returns.std():.2f}%\n"
            f"最大: {returns.max():.2f}%\n"
            f"最小: {returns.min():.2f}%\n"
            f"歪度: {returns.skew():.2f}\n"
            f"尖度: {returns.kurtosis():.2f}"
        )

        plt.figtext(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"リターン分布チャートを保存: {save_path}")
        else:
            plt.show()

        plt.close()
    else:
        logger.warning(f"目的変数 '{target_col}' が見つかりません")

    logger.debug("リターン分布描画完了")


def create_analysis_dashboard(
    data: pd.DataFrame, symbol: str, save_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    総合分析ダッシュボードを作成

    Args:
        data (pd.DataFrame): 株価データ
        symbol (str): 銘柄シンボル
        save_dir (Optional[str]): 保存ディレクトリ

    Returns:
        Dict[str, str]: 作成されたファイルのパス
    """
    logger.info(f"分析ダッシュボード作成開始: {symbol}")

    file_paths = {}

    # 基本価格チャート
    if save_dir:
        price_path = f"{save_dir}/{symbol}_price_chart.png"
        plot_price_chart(data, symbol, price_path)
        file_paths["price_chart"] = price_path
    else:
        plot_price_chart(data, symbol)

    # テクニカル分析チャート
    if save_dir:
        tech_path = f"{save_dir}/{symbol}_technical_analysis.png"
        plot_technical_analysis(data, symbol, tech_path)
        file_paths["technical_chart"] = tech_path
    else:
        plot_technical_analysis(data, symbol)

    logger.info("分析ダッシュボード作成完了")
    return file_paths


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    # ロギング設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    print("=== 株価可視化テスト ===")
    try:
        from ..data.fetchers import get_stock_data
        from .features import clean_features, create_all_features

        # テスト用データ取得
        data = get_stock_data("AAPL", "6mo")
        print(
            f"データ期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}"
        )

        print("\n📊 基本価格チャートを表示中...")
        plot_price_chart(data, "AAPL")

        print("\n📈 テクニカル分析チャートを表示中...")
        plot_technical_analysis(data, "AAPL")

        # 特徴量を使った可視化
        print("\n🔍 特徴量分析中...")
        features, targets = create_all_features(data)
        clean_features_df, clean_targets_df = clean_features(features, targets)

        # 結合したデータで相関分析
        combined_data = pd.concat([clean_features_df, clean_targets_df], axis=1)

        print("\n📊 相関ヒートマップを表示中...")
        plot_correlation_heatmap(combined_data, "return_5d", top_n=15)

        print("\n📈 リターン分布を表示中...")
        plot_return_distribution(clean_targets_df, "return_5d")

        print("\n✅ 可視化テスト完了")

    except ImportError as e:
        print(f"インポートエラー: {e}")
        print("以下のコマンドで実行してください:")
        print("uv run python -m stock_analyzer.analysis.visualizer")
    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
