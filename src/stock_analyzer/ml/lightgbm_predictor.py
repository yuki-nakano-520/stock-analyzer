"""LightGBMを使った高度な株価予測モデル（学習用）"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

warnings.filterwarnings("ignore", category=UserWarning)


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="lightgbm_predictor")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


class LightGBMStockPredictor:
    """
    LightGBMを使った高度な株価予測モデル

    特徴:
    - 高精度なグラディエントブースティング
    - 自動的な特徴量重要度分析
    - 時系列交差検証による堅牢な評価
    - 複数予測期間への対応
    """

    def __init__(self, model_name: str = "stock_predictor"):
        """
        LightGBM株価予測モデルを初期化

        Args:
            model_name (str): モデル名
        """
        logger.info(f"LightGBM予測モデル初期化: {model_name}")

        self.model_name = model_name
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.feature_importance: Dict[str, float] = {}

        # LightGBMのデフォルトパラメータ
        self.lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

    def prepare_training_data(
        self, features: pd.DataFrame, targets: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        訓練用データを準備

        Args:
            features (pd.DataFrame): 特徴量データ
            targets (pd.DataFrame): 目的変数データ

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: クリーニング済み(特徴量, 目的変数)
        """
        logger.debug("LightGBM訓練データ準備開始")

        # 数値型の特徴量のみを選択
        numeric_features = features.select_dtypes(include=[np.number])

        # 無限値とNaNを処理
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        targets = targets.replace([np.inf, -np.inf], np.nan)

        # 欠損値が多すぎる列を削除（50%以上）
        nan_ratio = numeric_features.isnull().sum() / len(numeric_features)
        high_nan_columns = nan_ratio[nan_ratio > 0.5].index
        if len(high_nan_columns) > 0:
            logger.warning(f"欠損値が多い特徴量を削除: {list(high_nan_columns)}")
            numeric_features = numeric_features.drop(columns=high_nan_columns)

        # 残った欠損値を前方埋め
        numeric_features = numeric_features.ffill().bfill().fillna(0)
        targets = targets.ffill().bfill().fillna(0)

        # 特徴量名を保存
        self.feature_names = list(numeric_features.columns)

        logger.debug(
            f"LightGBM訓練データ準備完了: 特徴量{numeric_features.shape}, 目的変数{targets.shape}"
        )

        return numeric_features, targets

    def train_model(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_columns: List[str] | None = None,
        test_size: float = 0.2,
        n_splits: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        複数の目的変数に対してモデルを訓練

        Args:
            features (pd.DataFrame): 特徴量データ
            targets (pd.DataFrame): 目的変数データ
            target_columns (List[str]): 予測対象の列名リスト
            test_size (float): テストデータの割合
            n_splits (int): 交差検証の分割数

        Returns:
            Dict[str, Dict[str, float]]: 各目的変数の評価結果
        """
        if target_columns is None:
            target_columns = ["return_5d", "return_30d"]
        logger.info(f"LightGBMモデル訓練開始: {target_columns}")

        # データ準備
        X, y_all = self.prepare_training_data(features, targets)

        results = {}

        for target_col in target_columns:
            if target_col not in y_all.columns:
                logger.warning(f"目的変数 '{target_col}' が見つかりません")
                continue

            logger.info(f"目的変数 '{target_col}' の訓練開始")

            y = y_all[target_col]

            # 有効なデータのインデックスを取得
            valid_indices = ~(
                X.isnull().any(axis=1) | y.isnull() | np.isinf(y) | np.isnan(y)
            )
            X_clean = X[valid_indices].copy()  # Ensure DataFrame type
            y_clean = y[valid_indices].copy()  # Ensure Series type

            # Type assertions to help pyright
            if not isinstance(X_clean, pd.DataFrame):
                raise TypeError("X_clean should be DataFrame")
            if not isinstance(y_clean, pd.Series):
                raise TypeError("y_clean should be Series")

            logger.debug(
                f"有効データ数: {len(X_clean)}/{len(X)} ({len(X_clean) / len(X) * 100:.1f}%)"
            )

            # 訓練・テストデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size, random_state=42, shuffle=False
            )

            # LightGBMモデル作成
            model = lgb.LGBMRegressor(
                n_estimators=1000, early_stopping_rounds=50, **self.lgb_params
            )

            # 訓練
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.log_evaluation(0)],  # ログ出力を抑制
            )

            # 予測と評価
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # 評価指標計算
            train_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "mae": mean_absolute_error(y_train, y_pred_train),
                "r2": r2_score(y_train, y_pred_train),
            }

            test_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "mae": mean_absolute_error(y_test, y_pred_test),
                "r2": r2_score(y_test, y_pred_test),
            }

            # 時系列交差検証
            cv_scores = self._cross_validate_timeseries(
                model, X_clean, y_clean, n_splits
            )

            results[target_col] = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_metrics": cv_scores,
                "feature_importance": dict(
                    zip(self.feature_names, model.feature_importances_, strict=False)
                ),
            }

            # モデルを保存
            self.models[target_col] = model

            logger.info(
                f"'{target_col}' 訓練完了 - テストR²: {test_metrics['r2']:.3f}, RMSE: {test_metrics['rmse']:.3f}"
            )

        self.is_fitted = True

        # 特徴量重要度を統合
        self._aggregate_feature_importance()

        logger.info(f"全モデル訓練完了: {len(results)}個のモデル")

        return results

    def _cross_validate_timeseries(
        self, model: lgb.LGBMRegressor, X: pd.DataFrame, y: pd.Series, n_splits: int
    ) -> Dict[str, float]:
        """
        時系列データに適した交差検証を実行

        Args:
            model: LightGBMモデル
            X: 特徴量
            y: 目的変数
            n_splits: 分割数

        Returns:
            Dict[str, float]: 交差検証結果
        """
        logger.debug(f"時系列交差検証開始: {n_splits}分割")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {"rmse": [], "mae": [], "r2": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            # 各フォールドで新しいモデルを作成
            fold_model = lgb.LGBMRegressor(**self.lgb_params, n_estimators=500)
            fold_model.fit(
                X_fold_train, y_fold_train, callbacks=[lgb.log_evaluation(0)]
            )

            y_fold_pred = fold_model.predict(X_fold_val)

            # 評価指標を計算
            cv_scores["rmse"].append(
                np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
            )
            cv_scores["mae"].append(mean_absolute_error(y_fold_val, y_fold_pred))
            cv_scores["r2"].append(r2_score(y_fold_val, y_fold_pred))

            logger.debug(f"フォールド {fold + 1}/{n_splits} 完了")

        # 平均と標準偏差を計算
        cv_results = {
            "rmse_mean": float(np.mean(cv_scores["rmse"])),
            "rmse_std": float(np.std(cv_scores["rmse"])),
            "mae_mean": float(np.mean(cv_scores["mae"])),
            "mae_std": float(np.std(cv_scores["mae"])),
            "r2_mean": float(np.mean(cv_scores["r2"])),
            "r2_std": float(np.std(cv_scores["r2"])),
        }

        logger.debug(
            f"交差検証完了 - R²: {cv_results['r2_mean']:.3f} (±{cv_results['r2_std']:.3f})"
        )

        return cv_results

    def _aggregate_feature_importance(self) -> None:
        """複数モデルの特徴量重要度を統合"""
        if not self.models:
            return

        # 全モデルの特徴量重要度を平均
        importance_sum = {}
        for model_name, model in self.models.items():
            logger.debug(f"特徴量重要度を集計中: {model_name}")
            for feature, importance in zip(
                self.feature_names, model.feature_importances_, strict=False
            ):
                if feature not in importance_sum:
                    importance_sum[feature] = []
                importance_sum[feature].append(importance)

        # 平均を計算
        self.feature_importance = {
            feature: np.mean(importances)
            for feature, importances in importance_sum.items()
        }

        # 重要度でソート
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[
                :20
            ]  # 上位20個
        )

    def predict(
        self,
        features: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        株価予測を実行

        Args:
            features (pd.DataFrame): 予測用特徴量
            target_columns (Optional[List[str]]): 予測する目的変数（Noneの場合は全て）

        Returns:
            Dict[str, np.ndarray]: 各目的変数の予測結果
        """
        if not self.is_fitted:
            raise ValueError(
                "モデルが訓練されていません。先にtrain_model()を実行してください。"
            )

        if target_columns is None:
            target_columns = list(self.models.keys())

        logger.debug(f"予測実行: {target_columns}")

        # 特徴量を準備
        X = features[self.feature_names].fillna(0)

        predictions = {}
        for target_col in target_columns:
            if target_col not in self.models:
                logger.warning(f"モデル '{target_col}' が見つかりません")
                continue

            pred = self.models[target_col].predict(X)
            predictions[target_col] = pred
            # Safe length calculation for different prediction result types
            if hasattr(pred, "shape") and pred.shape:
                pred_count = pred.shape[0]
            elif hasattr(pred, "__len__") and not hasattr(
                pred, "toarray"
            ):  # Avoid spmatrix
                try:
                    pred_count = len(pred)  # type: ignore[arg-type]
                except TypeError:
                    pred_count = (
                        getattr(pred, "shape", [1])[0] if hasattr(pred, "shape") else 1
                    )
            else:
                pred_count = (
                    getattr(pred, "shape", [1])[0] if hasattr(pred, "shape") else 1
                )
            logger.debug(f"'{target_col}' 予測完了: {pred_count}件")

        return predictions

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n (int): 取得する上位N個

        Returns:
            Dict[str, float]: 特徴量重要度
        """
        if not self.feature_importance:
            logger.warning("特徴量重要度が計算されていません")
            return {}

        return dict(list(self.feature_importance.items())[:top_n])

    def save_models(self, save_dir: str = "models") -> None:
        """
        モデルを保存

        Args:
            save_dir (str): 保存ディレクトリ
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 各モデルを保存
        for target_col, model in self.models.items():
            model_file = save_path / f"{self.model_name}_{target_col}.joblib"
            joblib.dump(model, model_file)
            logger.info(f"モデル保存: {model_file}")

        # メタデータを保存
        metadata = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "target_columns": list(self.models.keys()),
        }

        metadata_file = save_path / f"{self.model_name}_metadata.joblib"
        joblib.dump(metadata, metadata_file)
        logger.info(f"メタデータ保存: {metadata_file}")

    def load_models(self, save_dir: str = "models") -> None:
        """
        モデルを読み込み

        Args:
            save_dir (str): 読み込みディレクトリ
        """
        save_path = Path(save_dir)

        # メタデータを読み込み
        metadata_file = save_path / f"{self.model_name}_metadata.joblib"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"メタデータファイルが見つかりません: {metadata_file}"
            )

        metadata = joblib.load(metadata_file)
        self.feature_names = metadata["feature_names"]
        self.feature_importance = metadata["feature_importance"]

        # 各モデルを読み込み
        self.models = {}
        for target_col in metadata["target_columns"]:
            model_file = save_path / f"{self.model_name}_{target_col}.joblib"
            if model_file.exists():
                self.models[target_col] = joblib.load(model_file)
                logger.info(f"モデル読み込み: {model_file}")
            else:
                logger.warning(f"モデルファイルが見つかりません: {model_file}")

        self.is_fitted = len(self.models) > 0
        logger.info(f"モデル読み込み完了: {len(self.models)}個のモデル")


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    # ロギング設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    print("=== LightGBM株価予測テスト ===")
    try:
        from ..analysis.features import clean_features, create_all_features
        from ..data.fetchers import get_stock_data

        # 2年分のデータで十分な学習データを確保
        data = get_stock_data("AAPL", "2y")
        print(
            f"データ期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}"
        )

        # 特徴量作成
        features, targets = create_all_features(data)
        features, targets = clean_features(features, targets)

        print("\n📊 データサマリー:")
        print(f"特徴量: {features.shape}")
        print(f"目的変数: {targets.shape}")

        # LightGBMモデル作成・訓練
        print("\n🤖 LightGBMモデル訓練:")
        predictor = LightGBMStockPredictor("AAPL_predictor")

        # 複数期間を予測
        results = predictor.train_model(
            features, targets, target_columns=["return_5d", "return_30d"], n_splits=3
        )

        # 結果表示
        for target_col, result in results.items():
            test_metrics = result["test_metrics"]
            cv_metrics = result["cv_metrics"]

            # Type assertion to ensure these are dictionaries
            if not isinstance(test_metrics, dict):
                raise TypeError("test_metrics should be dict")
            if not isinstance(cv_metrics, dict):
                raise TypeError("cv_metrics should be dict")

            print(f"\n📈 {target_col} 予測結果:")
            print(f"テスト R²: {test_metrics['r2']:.3f}")
            print(f"テスト RMSE: {test_metrics['rmse']:.3f}")
            print(
                f"交差検証 R²: {cv_metrics['r2_mean']:.3f} (±{cv_metrics['r2_std']:.3f})"
            )

        # 特徴量重要度表示
        print("\n🔍 重要特徴量 (上位10個):")
        importance = predictor.get_feature_importance(10)
        for feature, score in importance.items():
            print(f"{feature}: {score:.3f}")

        # 予測実行
        print("\n🎯 最新データでの予測:")
        latest_features = features.tail(1)
        predictions = predictor.predict(latest_features)

        for target_col, pred in predictions.items():
            days = target_col.split("_")[1][:-1]  # "5d" -> "5"
            print(f"{days}日後リターン予測: {pred[0]:.2f}%")

        print("\n✅ LightGBMテスト完了")

    except ImportError:
        print("必要なモジュールが見つかりません。")
        print("以下のコマンドで実行してください:")
        print("uv run python -m stock_analyzer.ml.lightgbm_predictor")
    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
