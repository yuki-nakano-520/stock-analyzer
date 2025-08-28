"""scikit-learnを使った基本的な機械学習モデル（学習用）"""

import logging
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def _get_logger() -> Any:
    """ロガーを取得"""
    try:
        from ..utils.logging_config import get_logger

        return get_logger(__name__, module="ml_models")
    except ImportError:
        import logging

        return logging.getLogger(__name__)


logger: Any = _get_logger()


class StockPricePredictor:
    """
    株価予測のためのシンプルな機械学習モデル（学習用）

    各アルゴリズムの特徴:
    - LinearRegression: 最もシンプル、解釈しやすい
    - RandomForest: 非線形関係を捉える、過学習に強い
    - SVR: 複雑なパターン認識、小データでも有効
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        株価予測モデルを初期化

        Args:
            model_type (str): 使用するモデル ('linear', 'random_forest', 'svr')
        """
        logger.info(f"株価予測モデル初期化: {model_type}")

        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

        # モデルの初期化
        if model_type == "linear":
            self.model = LinearRegression()
            logger.debug("線形回帰モデルを選択")
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            logger.debug("ランダムフォレスト回帰モデルを選択")
        elif model_type == "svr":
            self.model = SVR(kernel="rbf", C=100, gamma="scale")
            logger.debug("サポートベクター回帰モデルを選択")
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")

    def prepare_data(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_column: str = "return_5d",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        機械学習用にデータを準備

        Args:
            features (pd.DataFrame): 特徴量データ
            targets (pd.DataFrame): 目的変数データ
            target_column (str): 予測対象の列名

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) 形式の学習データ
        """
        logger.debug(f"データ準備開始: target={target_column}")

        # 目的変数の選択
        if target_column not in targets.columns:
            logger.error(f"目的変数が見つかりません: {target_column}")
            raise ValueError(f"目的変数 '{target_column}' が見つかりません")

        y = targets[target_column].values

        # 特徴量の準備（数値列のみ）
        numeric_features = features.select_dtypes(include=[np.number])
        X = numeric_features.values
        self.feature_names = list(numeric_features.columns)

        # NaNの処理
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]

        logger.debug(f"データ準備完了: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(
            f"有効なサンプル数: {len(X)}/{len(features)} ({len(X)/len(features)*100:.1f}%)"
        )

        return X, y

    def train(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        モデルを訓練

        Args:
            X (np.ndarray): 特徴量
            y (np.ndarray): 目的変数
            test_size (float): テストデータの割合

        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"モデル訓練開始: {self.model_type}")

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logger.debug(f"データ分割: train={len(X_train)}, test={len(X_test)}")

        # 特徴量のスケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.debug("特徴量スケーリング完了")

        # モデル訓練
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        logger.debug("モデル訓練完了")

        # 予測と評価
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # 評価指標の計算
        results = {
            "model_type": self.model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
        }

        # 特徴量重要度（Random Forestのみ）
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_importance = dict(
                zip(self.feature_names, importances, strict=False)
            )
            results["feature_importance"] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            )  # 上位10個のみ

        logger.info(
            f"モデル評価完了 - テストR²: {results['test_r2']:.3f}, テストMAE: {results['test_mae']:.3f}"
        )

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を実行

        Args:
            X (np.ndarray): 特徴量データ

        Returns:
            np.ndarray: 予測結果
        """
        if not self.is_fitted:
            raise ValueError(
                "モデルが訓練されていません。先にtrain()を実行してください。"
            )

        logger.debug(f"予測実行: サンプル数={len(X)}")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        logger.debug("予測完了")

        return predictions

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Any]:
        """
        交差検証を実行

        Args:
            X (np.ndarray): 特徴量
            y (np.ndarray): 目的変数
            cv (int): 分割数

        Returns:
            Dict[str, Any]: 交差検証結果
        """
        logger.info(f"交差検証開始: {cv}分割")

        # スケーリング
        X_scaled = self.scaler.fit_transform(X)

        # 交差検証（時系列を考慮）
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring="r2")

        results = {
            "cv_scores": scores.tolist(),
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "cv_folds": cv,
        }

        logger.info(
            f"交差検証完了 - 平均R²: {results['mean_score']:.3f} (±{results['std_score']:.3f})"
        )

        return results

    def save_model(self, filepath: str) -> None:
        """
        モデルを保存

        Args:
            filepath (str): 保存先パス
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません。")

        logger.debug(f"モデル保存: {filepath}")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"モデル保存完了: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        モデルを読み込み

        Args:
            filepath (str): モデルファイルパス
        """
        logger.debug(f"モデル読み込み: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data["model_type"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = True

        logger.info(f"モデル読み込み完了: {filepath}")


class StockDirectionClassifier:
    """
    株価の方向性（上昇/下降）を予測する分類モデル（学習用）
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        株価方向性予測モデルを初期化

        Args:
            model_type (str): 使用するモデル ('logistic', 'random_forest', 'svc')
        """
        logger.info(f"株価方向性予測モデル初期化: {model_type}")

        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

        # モデルの初期化
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            logger.debug("ロジスティック回帰モデルを選択")
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            logger.debug("ランダムフォレスト分類モデルを選択")
        elif model_type == "svc":
            self.model = SVC(kernel="rbf", C=100, gamma="scale", random_state=42)
            logger.debug("サポートベクター分類モデルを選択")
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")

    def prepare_data(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_column: str = "direction_5d",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分類用にデータを準備

        Args:
            features (pd.DataFrame): 特徴量データ
            targets (pd.DataFrame): 目的変数データ
            target_column (str): 予測対象の列名

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) 形式の学習データ
        """
        logger.debug(f"分類データ準備開始: target={target_column}")

        # 目的変数の選択
        if target_column not in targets.columns:
            logger.error(f"目的変数が見つかりません: {target_column}")
            raise ValueError(f"目的変数 '{target_column}' が見つかりません")

        y = targets[target_column].values

        # 特徴量の準備（数値列のみ）
        numeric_features = features.select_dtypes(include=[np.number])
        X = numeric_features.values
        self.feature_names = list(numeric_features.columns)

        # NaNの処理
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]

        logger.debug(f"分類データ準備完了: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"クラス分布: 上昇={sum(y)}件, 下降={len(y)-sum(y)}件")

        return X, y

    def train(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        分類モデルを訓練

        Args:
            X (np.ndarray): 特徴量
            y (np.ndarray): 目的変数（0 or 1）
            test_size (float): テストデータの割合

        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"分類モデル訓練開始: {self.model_type}")

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.debug(f"データ分割: train={len(X_train)}, test={len(X_test)}")

        # 特徴量のスケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.debug("特徴量スケーリング完了")

        # モデル訓練
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        logger.debug("分類モデル訓練完了")

        # 予測と評価
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # 評価指標の計算
        results = {
            "model_type": self.model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "classification_report": classification_report(
                y_test, y_pred_test, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        }

        # 特徴量重要度（Random Forestのみ）
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_importance = dict(
                zip(self.feature_names, importances, strict=False)
            )
            results["feature_importance"] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            )  # 上位10個のみ

        logger.info(f"分類モデル評価完了 - テスト精度: {results['test_accuracy']:.3f}")

        return results


def compare_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_column: str = "return_5d",
    task_type: str = "regression",
) -> Dict[str, Any]:
    """
    複数のモデルを比較評価

    Args:
        features (pd.DataFrame): 特徴量データ
        targets (pd.DataFrame): 目的変数データ
        target_column (str): 予測対象の列名
        task_type (str): タスクタイプ ('regression' or 'classification')

    Returns:
        Dict[str, Any]: 各モデルの比較結果
    """
    logger.info(f"モデル比較開始: {task_type}, target={target_column}")

    results = {"task_type": task_type, "target_column": target_column, "models": {}}

    if task_type == "regression":
        model_types = ["linear", "random_forest", "svr"]

        for model_type in model_types:
            logger.debug(f"モデル評価中: {model_type}")

            try:
                # モデル作成と訓練
                predictor = StockPricePredictor(model_type)
                X, y = predictor.prepare_data(features, targets, target_column)

                # 交差検証
                cv_results = predictor.cross_validate(X, y, cv=5)

                # 通常の訓練・評価
                train_results = predictor.train(X, y)

                # 結果統合
                results["models"][model_type] = {
                    **train_results,
                    "cross_validation": cv_results,
                }

                logger.debug(f"{model_type} 評価完了")

            except Exception as e:
                logger.error(f"{model_type} でエラー: {e}")
                results["models"][model_type] = {"error": str(e)}

    elif task_type == "classification":
        model_types = ["logistic", "random_forest", "svc"]

        for model_type in model_types:
            logger.debug(f"分類モデル評価中: {model_type}")

            try:
                # モデル作成と訓練
                classifier = StockDirectionClassifier(model_type)
                X, y = classifier.prepare_data(features, targets, target_column)

                # 通常の訓練・評価
                train_results = classifier.train(X, y)

                results["models"][model_type] = train_results

                logger.debug(f"{model_type} 分類評価完了")

            except Exception as e:
                logger.error(f"{model_type} でエラー: {e}")
                results["models"][model_type] = {"error": str(e)}

    logger.info(f"モデル比較完了: {len(results['models'])}モデル")

    return results


# 使用例（このファイルを直接実行した時のみ動作）
if __name__ == "__main__":
    # ロギング設定
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    print("=== scikit-learn機械学習テスト ===")
    try:
        from ..analysis.features import clean_features, create_all_features
        from ..data.fetchers import get_stock_data

        # 1年分のデータで十分な履歴を確保
        data = get_stock_data("AAPL", "1y")
        print(
            f"データ期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}"
        )

        # 特徴量作成
        features, targets = create_all_features(data)
        features, targets = clean_features(features, targets)

        print("\n📊 データサマリー:")
        print(f"特徴量: {features.shape}")
        print(f"目的変数: {targets.shape}")

        # 回帰タスクのテスト
        print("\n🤖 回帰モデル比較:")
        regression_results = compare_models(
            features, targets, "return_5d", "regression"
        )

        for model_name, result in regression_results["models"].items():
            if "error" not in result:
                print(
                    f"{model_name}: R²={result['test_r2']:.3f}, MAE={result['test_mae']:.3f}"
                )
                if "cross_validation" in result:
                    cv = result["cross_validation"]
                    print(
                        f"  交差検証R²: {cv['mean_score']:.3f} (±{cv['std_score']:.3f})"
                    )
            else:
                print(f"{model_name}: エラー - {result['error']}")

        # 分類タスクのテスト
        print("\n🎯 分類モデル比較:")
        classification_results = compare_models(
            features, targets, "direction_5d", "classification"
        )

        for model_name, result in classification_results["models"].items():
            if "error" not in result:
                print(f"{model_name}: 精度={result['test_accuracy']:.3f}")
            else:
                print(f"{model_name}: エラー - {result['error']}")

        print("\n✅ 機械学習テスト完了")

    except ImportError:
        print("必要なモジュールが見つかりません。")
        print("以下のコマンドで実行してください:")
        print("uv run python -m stock_analyzer.ml.basic_models")
    except Exception as e:
        print(f"エラー: {e}")
