"""日本語CSV出力機能（学習用）"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="csv_exporter")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


class JapaneseCsvExporter:
    """
    日本語対応のCSV出力機能

    特徴:
    - 日本語カラム名での出力
    - 投資判断に必要な指標を含む
    - スコア形式での優先度表示
    - 複数銘柄対応
    """

    def __init__(self):
        """CSV出力機能を初期化"""
        logger.info("日本語CSV出力機能初期化")

        # 日本語カラム名のマッピング
        self.column_mapping = {
            # 基本情報
            "symbol": "銘柄コード",
            "company_name": "会社名",
            "current_price": "現在価格($)",
            "sector": "セクター",
            "industry": "業界",
            "market_cap": "時価総額($)",
            "analysis_date": "分析日時",
            # 予測結果
            "return_5d": "5日後リターン予測(%)",
            "return_30d": "30日後リターン予測(%)",
            "direction_5d": "5日後上昇確率",
            "direction_30d": "30日後上昇確率",
            # テクニカル指標
            "sma_5": "単純移動平均5日($)",
            "sma_20": "単純移動平均20日($)",
            "sma_50": "単純移動平均50日($)",
            "rsi_14": "RSI(14日)",
            "macd": "MACD",
            "bb_position": "ボリンジャーバンド位置",
            "volume_ratio": "出来高比率",
            # スコア・ランキング
            "investment_score": "投資スコア(0-100)",
            "risk_score": "リスクスコア(0-100)",
            "recommendation": "推奨度",
            "priority_rank": "優先順位",
            # その他
            "volatility_20d": "20日ボラティリティ",
            "price_change_5d": "5日価格変動率(%)",
            "confidence_level": "予測信頼度",
        }

        # 推奨レベルの定義
        self.recommendation_levels = {
            "strong_buy": "強い買い",
            "buy": "買い",
            "hold": "ホールド",
            "sell": "売り",
            "strong_sell": "強い売り",
        }

    def calculate_investment_score(
        self,
        return_5d: float,
        return_30d: float,
        volatility: float,
        rsi: float,
        volume_ratio: float,
    ) -> float:
        """
        投資スコアを計算

        Args:
            return_5d: 5日後リターン予測
            return_30d: 30日後リターン予測
            volatility: ボラティリティ
            rsi: RSI値
            volume_ratio: 出来高比率

        Returns:
            float: 投資スコア (0-100)
        """
        score = 50  # ベーススコア

        # リターンによるスコア調整
        avg_return = (return_5d + return_30d) / 2
        if avg_return > 10:
            score += 30
        elif avg_return > 5:
            score += 20
        elif avg_return > 0:
            score += 10
        elif avg_return < -10:
            score -= 30
        elif avg_return < -5:
            score -= 20
        else:
            score -= 10

        # ボラティリティによる調整（高すぎるとリスク）
        if volatility > 0.4:  # 40%以上
            score -= 15
        elif volatility > 0.3:  # 30%以上
            score -= 10
        elif volatility > 0.2:  # 20%以上
            score -= 5

        # RSIによる調整（買われすぎ/売られすぎ）
        if 30 <= rsi <= 70:  # 適正範囲
            score += 5
        elif rsi > 80 or rsi < 20:  # 極端
            score -= 10

        # 出来高による調整
        if volume_ratio > 1.5:  # 高出来高
            score += 5
        elif volume_ratio < 0.5:  # 低出来高
            score -= 5

        return max(0, min(100, score))

    def calculate_risk_score(
        self, volatility: float, bb_position: float, return_std: float = 0
    ) -> float:
        """
        リスクスコアを計算

        Args:
            volatility: ボラティリティ
            bb_position: ボリンジャーバンド位置
            return_std: リターンの標準偏差

        Returns:
            float: リスクスコア (0-100、高いほどリスキー)
        """
        score = 20  # ベースリスク

        # ボラティリティによるリスク
        score += min(50, volatility * 100)

        # ボリンジャーバンド位置によるリスク
        if bb_position > 0.8 or bb_position < 0.2:
            score += 15  # 極端な位置はリスキー

        # リターン標準偏差によるリスク
        score += min(15, return_std * 2)

        return max(0, min(100, score))

    def get_recommendation(self, investment_score: float, risk_score: float) -> str:
        """
        投資推奨を決定

        Args:
            investment_score: 投資スコア
            risk_score: リスクスコア

        Returns:
            str: 推奨レベル
        """
        # 調整済みスコア = 投資スコア - リスクペナルティ
        adjusted_score = investment_score - (risk_score * 0.3)

        if adjusted_score >= 80:
            return self.recommendation_levels["strong_buy"]
        elif adjusted_score >= 65:
            return self.recommendation_levels["buy"]
        elif adjusted_score >= 40:
            return self.recommendation_levels["hold"]
        elif adjusted_score >= 20:
            return self.recommendation_levels["sell"]
        else:
            return self.recommendation_levels["strong_sell"]

    def prepare_stock_analysis_data(
        self,
        symbol: str,
        company_info: Dict[str, Any],
        indicators: Dict[str, float],
        predictions: Dict[str, np.ndarray],
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        単一銘柄の分析データを準備

        Args:
            symbol: 銘柄コード
            company_info: 会社情報
            indicators: テクニカル指標
            predictions: 予測結果
            features: 特徴量データ

        Returns:
            Dict[str, Any]: CSV出力用データ
        """
        logger.debug(f"銘柄分析データ準備: {symbol}")

        # 予測値を取得
        return_5d = (
            predictions.get("return_5d", [0])[0] if "return_5d" in predictions else 0
        )
        return_30d = (
            predictions.get("return_30d", [0])[0] if "return_30d" in predictions else 0
        )

        # ボラティリティ計算
        volatility = (
            features["volatility_20d"].iloc[-1]
            if "volatility_20d" in features.columns
            else 0
        )
        price_change_5d = (
            features["price_change_5d"].iloc[-1]
            if "price_change_5d" in features.columns
            else 0
        )

        # スコア計算
        investment_score = self.calculate_investment_score(
            return_5d,
            return_30d,
            volatility,
            indicators.get("rsi", 50),
            indicators.get("volume_ratio", 1),
        )

        risk_score = self.calculate_risk_score(
            volatility, indicators.get("bb_position", 0.5)
        )

        recommendation = self.get_recommendation(investment_score, risk_score)

        # 信頼度計算（仮実装）
        confidence = min(100, max(20, 70 - abs(indicators.get("rsi", 50) - 50)))

        data = {
            "symbol": symbol,
            "company_name": company_info.get("company_name", symbol),
            "current_price": company_info.get("current_price", 0),
            "sector": company_info.get("sector", "Unknown"),
            "industry": company_info.get("industry", "Unknown"),
            "market_cap": company_info.get("market_cap", 0),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "return_5d": round(return_5d, 2),
            "return_30d": round(return_30d, 2),
            "sma_5": round(indicators.get("sma_5", 0), 2),
            "sma_20": round(indicators.get("sma_20", 0), 2),
            "sma_50": round(indicators.get("sma_50", 0), 2),
            "rsi_14": round(indicators.get("rsi", 0), 1),
            "macd": round(indicators.get("macd", 0), 4),
            "bb_position": round(indicators.get("bb_position", 0), 2),
            "volume_ratio": round(indicators.get("volume_ratio", 0), 2),
            "investment_score": round(investment_score, 1),
            "risk_score": round(risk_score, 1),
            "recommendation": recommendation,
            "volatility_20d": round(volatility, 4),
            "price_change_5d": round(price_change_5d, 2),
            "confidence_level": round(confidence, 1),
        }

        logger.debug(f"データ準備完了: {symbol} - スコア {investment_score:.1f}")

        return data

    def export_to_csv(
        self,
        analysis_data: List[Dict[str, Any]],
        output_path: str = "analysis_results.csv",
    ) -> str:
        """
        分析データをCSVファイルに出力

        Args:
            analysis_data: 分析データのリスト
            output_path: 出力ファイルパス

        Returns:
            str: 実際の出力ファイルパス
        """
        logger.info(f"CSV出力開始: {len(analysis_data)}銘柄")

        if not analysis_data:
            logger.warning("出力するデータがありません")
            return ""

        # DataFrameに変換
        df = pd.DataFrame(analysis_data)

        # 投資スコアで降順ソート（優先順位）
        df = df.sort_values("investment_score", ascending=False).reset_index(drop=True)
        df["priority_rank"] = range(1, len(df) + 1)

        # カラム名を日本語に変換
        japanese_columns = {}
        for col in df.columns:
            japanese_columns[col] = self.column_mapping.get(col, col)

        df = df.rename(columns=japanese_columns)

        # 出力パスを準備
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = (
            output_file.parent / f"{output_file.stem}_{timestamp}{output_file.suffix}"
        )

        # CSV出力（UTF-8 with BOM for Excel compatibility）
        df.to_csv(
            filename_with_timestamp,
            index=False,
            encoding="utf-8-sig",  # Excel用のBOM付きUTF-8
            float_format="%.2f",
        )

        logger.info(f"CSV出力完了: {filename_with_timestamp}")

        # サマリー情報をログ出力
        self._log_summary(df)

        return str(filename_with_timestamp)

    def _log_summary(self, df: pd.DataFrame) -> None:
        """分析結果のサマリーをログ出力"""
        if df.empty:
            return

        total_stocks = len(df)
        strong_buy_count = (
            df[self.column_mapping["recommendation"]] == "強い買い"
        ).sum()
        buy_count = (df[self.column_mapping["recommendation"]] == "買い").sum()
        hold_count = (df[self.column_mapping["recommendation"]] == "ホールド").sum()

        avg_investment_score = df[self.column_mapping["investment_score"]].mean()
        avg_risk_score = df[self.column_mapping["risk_score"]].mean()

        logger.info(
            f"分析サマリー - 銘柄数: {total_stocks}, "
            f"強い買い: {strong_buy_count}, 買い: {buy_count}, ホールド: {hold_count}"
        )
        logger.info(
            f"平均投資スコア: {avg_investment_score:.1f}, "
            f"平均リスクスコア: {avg_risk_score:.1f}"
        )

    def create_summary_report(
        self,
        analysis_data: List[Dict[str, Any]],
        output_path: str = "summary_report.csv",
    ) -> str:
        """
        サマリーレポートを作成

        Args:
            analysis_data: 分析データ
            output_path: 出力パス

        Returns:
            str: 出力ファイルパス
        """
        logger.info("サマリーレポート作成開始")

        if not analysis_data:
            return ""

        df = pd.DataFrame(analysis_data)

        # セクター別集計
        sector_summary = (
            df.groupby("sector")
            .agg(
                {
                    "investment_score": ["count", "mean"],
                    "risk_score": "mean",
                    "return_5d": "mean",
                    "return_30d": "mean",
                }
            )
            .round(2)
        )

        # 推奨度別集計
        recommendation_summary = df["recommendation"].value_counts()

        # リスク別集計
        risk_categories = pd.cut(
            df["risk_score"],
            bins=[0, 30, 60, 100],
            labels=["低リスク", "中リスク", "高リスク"],
        )
        risk_summary = risk_categories.value_counts()

        # サマリーデータフレーム作成
        summary_data = {
            "項目": [
                "総銘柄数",
                "平均投資スコア",
                "平均リスクスコア",
                "平均5日リターン予測(%)",
                "平均30日リターン予測(%)",
                "強い買い推奨数",
                "買い推奨数",
                "ホールド推奨数",
                "低リスク銘柄数",
                "中リスク銘柄数",
                "高リスク銘柄数",
            ],
            "値": [
                len(df),
                round(df["investment_score"].mean(), 1),
                round(df["risk_score"].mean(), 1),
                round(df["return_5d"].mean(), 2),
                round(df["return_30d"].mean(), 2),
                recommendation_summary.get(self.recommendation_levels["strong_buy"], 0),
                recommendation_summary.get(self.recommendation_levels["buy"], 0),
                recommendation_summary.get(self.recommendation_levels["hold"], 0),
                risk_summary.get("低リスク", 0),
                risk_summary.get("中リスク", 0),
                risk_summary.get("高リスク", 0),
            ],
        }

        summary_df = pd.DataFrame(summary_data)

        # 出力
        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = (
            output_file.parent / f"{output_file.stem}_{timestamp}{output_file.suffix}"
        )

        summary_df.to_csv(filename_with_timestamp, index=False, encoding="utf-8-sig")

        logger.info(f"サマリーレポート出力完了: {filename_with_timestamp}")

        return str(filename_with_timestamp)

    def export_portfolio_summary(
        self, portfolio_result: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> str:
        """ポートフォリオサマリーをCSVファイルに出力する。

        Parameters
        ----------
        portfolio_result : Dict[str, Any]
            ポートフォリオ分析結果
        analysis_results : Dict[str, Any]
            個別銘柄分析結果

        Returns
        -------
        str
            出力されたCSVファイルのパス
        """
        logger.info("ポートフォリオサマリーCSV出力開始")

        try:
            # ポートフォリオ基本情報
            portfolio_stocks = portfolio_result.get("portfolio_stocks", [])
            portfolio_metrics = portfolio_result.get("portfolio_metrics")
            analysis_summary = portfolio_result.get("analysis_summary", {})

            # サマリーデータ準備
            summary_data = []

            # 基本統計
            for key, value in analysis_summary.items():
                summary_data.append({"項目": key, "値": value})

            # ポートフォリオメトリクス追加
            if portfolio_metrics:
                additional_metrics = {
                    "年間期待リターン": f"{portfolio_metrics.total_return:.2%}",
                    "年間ボラティリティ": f"{portfolio_metrics.volatility:.2%}",
                    "シャープレシオ": f"{portfolio_metrics.sharpe_ratio:.2f}",
                    "分散度": f"{portfolio_metrics.diversification_ratio:.2f}",
                }

                for key, value in additional_metrics.items():
                    summary_data.append({"項目": key, "値": value})

            # ポートフォリオ構成データ準備
            portfolio_data = []

            for stock in portfolio_stocks:
                portfolio_data.append(
                    {
                        "銘柄コード": stock.symbol,
                        "ポートフォリオ比重(%)": f"{stock.weight * 100:.1f}",
                        "投資金額($)": f"{stock.allocation_amount:,.0f}",
                        "投資スコア": f"{stock.investment_score:.1f}",
                        "リスクスコア": f"{stock.risk_score:.1f}",
                        "推奨度": stock.recommendation,
                        "5日後リターン予測(%)": f"{stock.expected_return * 5:.2f}",
                        "30日後リターン予測(%)": f"{stock.expected_return * 30:.2f}",
                    }
                )

            # DataFrames作成
            summary_df = pd.DataFrame(summary_data)
            portfolio_df = pd.DataFrame(portfolio_data)

            # 出力ファイル名生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # サマリーファイル出力
            summary_filename = f"portfolio_summary_{timestamp}.csv"
            summary_df.to_csv(summary_filename, index=False, encoding="utf-8-sig")

            # ポートフォリオ構成ファイル出力
            composition_filename = f"portfolio_composition_{timestamp}.csv"
            portfolio_df.to_csv(composition_filename, index=False, encoding="utf-8-sig")

            logger.info(
                f"ポートフォリオCSV出力完了: {summary_filename}, {composition_filename}"
            )

            return summary_filename

        except Exception as e:
            logger.error(f"ポートフォリオサマリー出力エラー: {e}")
            raise


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    print("=== 日本語CSV出力テスト ===")
    try:
        from ..analysis.indicators import calculate_all_indicators
        from ..data.fetchers import get_company_info, get_stock_data

        # テスト用データ
        symbols = ["AAPL", "MSFT", "GOOGL"]
        exporter = JapaneseCsvExporter()
        analysis_data = []

        for symbol in symbols:
            try:
                print(f"📊 {symbol} の分析中...")

                # データ取得
                data = get_stock_data(symbol, "6mo")
                company_info = get_company_info(symbol)
                indicators = calculate_all_indicators(data)

                # ダミー予測結果
                predictions = {
                    "return_5d": np.array([2.5]),
                    "return_30d": np.array([8.0]),
                }

                # 基本特徴量
                features = pd.DataFrame(
                    {
                        "volatility_20d": data["Close"].rolling(20).std()
                        / data["Close"].rolling(20).mean(),
                        "price_change_5d": data["Close"].pct_change(5) * 100,
                    }
                )

                # 分析データ準備
                stock_data = exporter.prepare_stock_analysis_data(
                    symbol, company_info, indicators, predictions, features
                )
                analysis_data.append(stock_data)

                print(
                    f"✅ {symbol} 完了 - スコア: {stock_data['investment_score']:.1f}"
                )

            except Exception as e:
                print(f"❌ {symbol} エラー: {e}")

        if analysis_data:
            # CSV出力
            csv_file = exporter.export_to_csv(analysis_data, "test_analysis.csv")
            print(f"\n📄 CSV出力完了: {csv_file}")

            # サマリーレポート
            summary_file = exporter.create_summary_report(
                analysis_data, "test_summary.csv"
            )
            print(f"📄 サマリー出力完了: {summary_file}")

        print("\n✅ CSV出力テスト完了")

    except ImportError:
        print("必要なモジュールが見つかりません。")
        print("以下のコマンドで実行してください:")
        print("uv run python -m stock_analyzer.reports.csv_exporter")
    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
