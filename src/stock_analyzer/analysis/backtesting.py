"""過去データを使った予測シミュレーション機能（学習用）"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ..analysis.features import FeatureEngineering
from ..data.fetchers import get_stock_data
from ..ml.lightgbm_predictor import LightGBMStockPredictor


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="backtesting")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


class BacktestSimulator:
    """
    時点指定型予測シミュレーター

    指定した投資日時点での予測が、指定した検証日時点で正しかったかを検証
    投資日時以降のデータは一切使用せず、リアルタイム予測環境を再現
    """

    def __init__(self):
        """シミュレーターを初期化"""
        logger.info("BacktestSimulator初期化")
        self.feature_engineer = FeatureEngineering()

    def run_point_in_time_simulation(
        self,
        symbol: str,
        investment_date: str,
        validation_date: str,
        training_period_months: int = 24,
        prediction_type: str = "direction",  # "direction" or "return"
    ) -> dict[str, Any]:
        """
        時点指定型シミュレーションを実行

        Args:
            symbol: 株式シンボル（例: "AAPL"）
            investment_date: 投資判断日（例: "2025-07-01"）
            validation_date: 検証日（例: "2025-08-25"）
            training_period_months: 訓練期間（投資日から遡る月数）
            prediction_type: 予測タイプ（"direction": 方向性, "return": リターン）

        Returns:
            dict[str, Any]: シミュレーション結果
        """
        logger.info(
            f"時点指定シミュレーション開始: {symbol} {investment_date} -> {validation_date}"
        )

        # 日付を解析
        investment_dt = pd.to_datetime(investment_date)
        validation_dt = pd.to_datetime(validation_date)

        # データリーケージ防止チェック
        if validation_dt <= investment_dt:
            raise ValueError("検証日は投資日より後である必要があります")

        # 予測期間（日数）を計算
        prediction_days = (validation_dt - investment_dt).days
        logger.debug(f"予測期間: {prediction_days}日")

        # 訓練期間を計算
        training_start = investment_dt - timedelta(days=training_period_months * 30)
        training_end = investment_dt - timedelta(days=1)  # 投資日前日まで

        logger.debug(f"訓練期間: {training_start.date()} ~ {training_end.date()}")

        # データ取得（投資日以降は絶対に含めない）
        historical_data = self._get_historical_data_safe(
            symbol, training_start, validation_dt
        )

        # 訓練用データと検証用データに分割
        train_data, validation_data = self._split_data_by_date(
            historical_data, investment_dt, validation_dt
        )

        if len(train_data) < 100:  # 最低限のデータ量チェック
            logger.warning(f"訓練データが不十分です: {len(train_data)}件")

        # 特徴量エンジニアリング（投資日時点で利用可能なデータのみ）
        train_features = self.feature_engineer.create_features(train_data)

        # 予測モデルを訓練（投資日時点で利用可能なデータのみ使用）
        predictor = self._train_prediction_model(
            train_features, prediction_days, prediction_type
        )

        # 投資日時点での予測実行
        investment_day_data = train_data.tail(1)  # 投資日前日までのデータ
        prediction_result = self._make_prediction(
            predictor, investment_day_data, prediction_days, prediction_type
        )

        # 実際の結果を取得（検証用）
        actual_result = self._get_actual_result(
            validation_data, investment_dt, validation_dt, prediction_type
        )

        # 結果を比較評価
        simulation_result = self._evaluate_prediction(
            prediction_result,
            actual_result,
            symbol,
            investment_date,
            validation_date,
            prediction_days,
            prediction_type,
        )

        logger.info(
            f"シミュレーション完了: {symbol} 予測精度: {simulation_result.get('accuracy', 'N/A')}"
        )
        return simulation_result

    def _get_historical_data_safe(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        データリーケージを防ぐ安全なデータ取得

        Args:
            symbol: 株式シンボル
            start_date: 開始日
            end_date: 終了日

        Returns:
            pd.DataFrame: 株価データ
        """
        logger.debug(
            f"安全なデータ取得: {symbol} {start_date.date()} ~ {end_date.date()}"
        )

        # 期間を文字列に変換してデータ取得
        period_months = ((end_date - start_date).days // 30) + 1
        period = f"{period_months}mo" if period_months <= 60 else "max"

        data = get_stock_data(symbol, period)

        # タイムゾーン情報を統一してフィルタリング
        if data.index.tz is not None:
            # データにタイムゾーンがある場合、日付もタイムゾーン付きに変換
            start_date = (
                start_date.tz_localize(data.index.tz)
                if start_date.tz is None
                else start_date.tz_convert(data.index.tz)
            )
            end_date = (
                end_date.tz_localize(data.index.tz)
                if end_date.tz is None
                else end_date.tz_convert(data.index.tz)
            )
        else:
            # データにタイムゾーンがない場合、タイムゾーン情報を削除
            start_date = (
                start_date.tz_localize(None)
                if start_date.tz is not None
                else start_date
            )
            end_date = (
                end_date.tz_localize(None) if end_date.tz is not None else end_date
            )

        filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        # 型チェッカーのために明示的にDataFrameであることを保証
        assert isinstance(filtered_data, pd.DataFrame)

        logger.debug(
            f"取得データ期間: {filtered_data.index.min()} ~ {filtered_data.index.max()} ({len(filtered_data)}件)"
        )
        return filtered_data

    def _split_data_by_date(
        self, data: pd.DataFrame, investment_dt: datetime, validation_dt: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        投資日を基準にデータを分割

        Args:
            data: 全データ
            investment_dt: 投資日時
            validation_dt: 検証日時

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (訓練データ, 検証データ)
        """
        # タイムゾーン情報を統一
        if data.index.tz is not None:
            investment_dt = (
                investment_dt.tz_localize(data.index.tz)
                if investment_dt.tz is None
                else investment_dt.tz_convert(data.index.tz)
            )
            validation_dt = (
                validation_dt.tz_localize(data.index.tz)
                if validation_dt.tz is None
                else validation_dt.tz_convert(data.index.tz)
            )
        else:
            investment_dt = (
                investment_dt.tz_localize(None)
                if investment_dt.tz is not None
                else investment_dt
            )
            validation_dt = (
                validation_dt.tz_localize(None)
                if validation_dt.tz is not None
                else validation_dt
            )

        # 投資日前日までを訓練データ（投資日当日のデータは使用しない）
        train_data = data[data.index < investment_dt]

        # 投資日以降検証日までを検証データ
        validation_data = data[
            (data.index >= investment_dt) & (data.index <= validation_dt)
        ]

        # 型チェッカーのために明示的にDataFrameであることを保証
        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(validation_data, pd.DataFrame)

        logger.debug(
            f"データ分割: 訓練={len(train_data)}件, 検証={len(validation_data)}件"
        )
        return train_data, validation_data

    def _train_prediction_model(
        self, features: pd.DataFrame, prediction_days: int, prediction_type: str
    ) -> LightGBMStockPredictor:
        """
        予測モデルを訓練

        Args:
            features: 特徴量データ
            prediction_days: 予測日数
            prediction_type: 予測タイプ

        Returns:
            LightGBMStockPredictor: 訓練済みモデル
        """
        logger.debug(
            f"予測モデル訓練開始: {prediction_days}日後の{prediction_type}予測"
        )

        # 目的変数を作成
        targets = self._create_target_variables(
            features, prediction_days, prediction_type
        )

        # LightGBMモデルで訓練
        predictor = LightGBMStockPredictor(f"backtest_model_{prediction_days}d")

        target_columns = [f"{prediction_type}_{prediction_days}d"]
        predictor.train_model(
            features, targets, target_columns=target_columns, n_splits=3
        )

        logger.debug("予測モデル訓練完了")
        return predictor

    def _create_target_variables(
        self, features: pd.DataFrame, prediction_days: int, prediction_type: str
    ) -> pd.DataFrame:
        """
        予測対象の目的変数を作成

        Args:
            features: 特徴量データ
            prediction_days: 予測日数
            prediction_type: 予測タイプ

        Returns:
            pd.DataFrame: 目的変数データ
        """
        targets = pd.DataFrame(index=features.index)

        if prediction_type == "direction":
            # 方向性予測: 上昇=1, 下降=0
            price_change = (
                features["close"].pct_change(prediction_days).shift(-prediction_days)
            )
            targets[f"direction_{prediction_days}d"] = (price_change > 0).astype(int)

        elif prediction_type == "return":
            # リターン予測: パーセント変化
            targets[f"return_{prediction_days}d"] = (
                features["close"].pct_change(prediction_days).shift(-prediction_days)
                * 100
            )

        # NaNを除去
        targets = targets.dropna()

        logger.debug(f"目的変数作成完了: {len(targets)}件")
        return targets

    def _make_prediction(
        self,
        predictor: LightGBMStockPredictor,
        investment_data: pd.DataFrame,
        prediction_days: int,
        prediction_type: str,
    ) -> dict[str, Any]:
        """
        投資日時点での予測を実行

        Args:
            predictor: 訓練済み予測モデル
            investment_data: 投資日時点のデータ
            prediction_days: 予測日数
            prediction_type: 予測タイプ

        Returns:
            dict[str, Any]: 予測結果
        """
        logger.debug(f"投資日時点での予測実行: {prediction_days}日後")

        # 特徴量作成
        features = self.feature_engineer.create_features(investment_data)

        # 予測実行
        target_columns = [f"{prediction_type}_{prediction_days}d"]
        predictions = predictor.predict(features, target_columns)

        prediction_value = predictions[f"{prediction_type}_{prediction_days}d"][0]

        result = {
            "prediction_type": prediction_type,
            "prediction_days": prediction_days,
            "predicted_value": prediction_value,
            "prediction_date": investment_data.index[-1],
        }

        if prediction_type == "direction":
            result["predicted_direction"] = "上昇" if prediction_value > 0.5 else "下降"
            result["confidence"] = abs(prediction_value - 0.5) * 2  # 0-1の信頼度

        logger.debug(f"予測結果: {result}")
        return result

    def _get_actual_result(
        self,
        validation_data: pd.DataFrame,
        investment_dt: datetime,
        validation_dt: datetime,
        prediction_type: str,
    ) -> dict[str, Any]:
        """
        実際の結果を取得

        Args:
            validation_data: 検証期間のデータ
            investment_dt: 投資日
            validation_dt: 検証日
            prediction_type: 予測タイプ

        Returns:
            dict[str, Any]: 実際の結果
        """
        logger.debug("実際の結果を計算中")

        # 投資日と検証日の価格を取得
        investment_price = None
        validation_price = None

        # デバッグ情報
        logger.debug(
            f"検証データ期間: {validation_data.index.min()} ~ {validation_data.index.max()}"
        )
        logger.debug(f"投資日: {investment_dt}, 検証日: {validation_dt}")

        # 投資日の価格（投資日当日または直近の営業日）
        for i in range(7):  # 7営業日以内で検索
            search_date = investment_dt + timedelta(days=i)
            # タイムゾーン統一
            if validation_data.index.tz is not None:
                search_date = (
                    search_date.tz_localize(validation_data.index.tz)
                    if search_date.tz is None
                    else search_date.tz_convert(validation_data.index.tz)
                )

            if search_date in validation_data.index:
                investment_price = validation_data.loc[search_date, "Close"]
                logger.debug(f"投資日の価格取得: {search_date} = ${investment_price}")
                break

        # 検証日の価格（検証日当日または直前の営業日）
        for i in range(7):  # 7営業日以内で検索
            search_date = validation_dt - timedelta(days=i)
            # タイムゾーン統一
            if validation_data.index.tz is not None:
                search_date = (
                    search_date.tz_localize(validation_data.index.tz)
                    if search_date.tz is None
                    else search_date.tz_convert(validation_data.index.tz)
                )

            if search_date in validation_data.index:
                validation_price = validation_data.loc[search_date, "Close"]
                logger.debug(f"検証日の価格取得: {search_date} = ${validation_price}")
                break

        if investment_price is None or validation_price is None:
            raise ValueError("投資日または検証日の価格データが見つかりません")

        # 実際の結果を計算
        actual_return = (validation_price - investment_price) / investment_price * 100
        actual_direction = "上昇" if actual_return > 0 else "下降"

        result = {
            "investment_price": investment_price,
            "validation_price": validation_price,
            "actual_return": actual_return,
            "actual_direction": actual_direction,
            "actual_direction_binary": 1 if actual_return > 0 else 0,
        }

        logger.debug(f"実際の結果: {result}")
        return result

    def _evaluate_prediction(
        self,
        prediction: dict[str, Any],
        actual: dict[str, Any],
        symbol: str,
        investment_date: str,
        validation_date: str,
        prediction_days: int,
        prediction_type: str,
    ) -> dict[str, Any]:
        """
        予測と実際の結果を比較評価

        Args:
            prediction: 予測結果
            actual: 実際の結果
            symbol: 株式シンボル
            investment_date: 投資日
            validation_date: 検証日
            prediction_days: 予測期間
            prediction_type: 予測タイプ

        Returns:
            dict[str, Any]: 評価結果
        """
        logger.debug("予測精度を評価中")

        result = {
            "symbol": symbol,
            "investment_date": investment_date,
            "validation_date": validation_date,
            "prediction_days": prediction_days,
            "prediction_type": prediction_type,
            "prediction": prediction,
            "actual": actual,
        }

        if prediction_type == "direction":
            # 方向性予測の精度評価
            predicted_direction_binary = 1 if prediction["predicted_value"] > 0.5 else 0
            actual_direction_binary = actual["actual_direction_binary"]

            result["direction_accuracy"] = (
                predicted_direction_binary == actual_direction_binary
            )
            result["accuracy"] = result["direction_accuracy"]
            result["prediction_summary"] = (
                f"予測: {prediction['predicted_direction']} "
                f"(信頼度: {prediction['confidence']:.1%}) | "
                f"実際: {actual['actual_direction']} | "
                f"正解: {'○' if result['direction_accuracy'] else '×'}"
            )

        elif prediction_type == "return":
            # リターン予測の精度評価
            predicted_return = prediction["predicted_value"]
            actual_return = actual["actual_return"]

            result["return_error"] = abs(predicted_return - actual_return)
            result["return_error_percentage"] = (
                abs(predicted_return - actual_return) / abs(actual_return) * 100
            )
            result["accuracy"] = 1 / (
                1 + result["return_error_percentage"] / 100
            )  # 0-1の精度スコア
            result["prediction_summary"] = (
                f"予測リターン: {predicted_return:.2f}% | "
                f"実際リターン: {actual_return:.2f}% | "
                f"誤差: {result['return_error']:.2f}%"
            )

        # 信頼性スコア（0-100）
        base_score = 50
        if prediction_type == "direction":
            if result["direction_accuracy"]:
                base_score += 30 + prediction["confidence"] * 20
            else:
                base_score -= 30 + prediction["confidence"] * 10
        else:
            base_score += (1 - result["return_error_percentage"] / 100) * 50

        result["confidence_score"] = max(0, min(100, base_score))

        logger.info(f"評価完了: {result['prediction_summary']}")
        return result


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    print("=== 時点指定型予測シミュレーションテスト ===")
    try:
        simulator = BacktestSimulator()

        # 例: 2024年7月1日にAAPL株を予測して8月25日の結果を検証
        result = simulator.run_point_in_time_simulation(
            symbol="AAPL",
            investment_date="2024-07-01",
            validation_date="2024-08-25",
            training_period_months=24,
            prediction_type="direction",
        )

        print("\n📊 シミュレーション結果:")
        print(f"銘柄: {result['symbol']}")
        print(f"期間: {result['investment_date']} → {result['validation_date']}")
        print(f"予測タイプ: {result['prediction_type']}")
        print(f"結果: {result['prediction_summary']}")
        print(f"信頼性スコア: {result['confidence_score']:.1f}/100")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
