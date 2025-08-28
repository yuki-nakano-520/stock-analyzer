"""Portfolio analysis and multiple stock comparison."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from ..types import AnalysisResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__, module="portfolio")


@dataclass
class PortfolioConfig:
    """Configuration for portfolio analysis."""

    max_stocks: int = 10
    investment_amount: float = 100000.0  # Total investment amount in USD
    risk_tolerance: float = 0.3  # Risk tolerance (0-1)
    rebalance_period: int = 30  # Days between rebalancing
    min_correlation: float = -0.5  # Minimum correlation for diversification
    max_correlation: float = 0.7  # Maximum correlation for diversification


@dataclass
class PortfolioStock:
    """Individual stock in portfolio."""

    symbol: str
    weight: float  # Portfolio weight (0-1)
    expected_return: float  # Expected daily return
    risk_score: float  # Risk score (0-100)
    recommendation: str  # Buy/Hold/Sell
    investment_score: float  # Investment score (0-100)
    allocation_amount: float  # Dollar amount to invest


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""

    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    risk_score: float
    correlation_matrix: pd.DataFrame


class PortfolioAnalyzer:
    """Portfolio analysis and optimization."""

    def __init__(self, config: PortfolioConfig):
        """Initialize portfolio analyzer.

        Parameters
        ----------
        config : PortfolioConfig
            Portfolio configuration
        """
        logger.debug(
            "Initializing PortfolioAnalyzer",
            max_stocks=config.max_stocks,
            investment_amount=config.investment_amount,
            risk_tolerance=config.risk_tolerance,
        )
        self.config = config

    def analyze_multiple_stocks(
        self,
        symbols: List[str],
        analysis_results: Dict[str, AnalysisResult],
        period: str = "1y",
    ) -> Dict[str, Any]:
        """Analyze multiple stocks for portfolio construction.

        Parameters
        ----------
        symbols : List[str]
            List of stock symbols to analyze
        analysis_results : Dict[str, AnalysisResult]
            Individual stock analysis results
        period : str
            Analysis period

        Returns
        -------
        Dict[str, Any]
            Portfolio analysis results
        """
        logger.info(
            "Starting multiple stock analysis",
            stock_count=len(symbols),
            symbols=symbols[:5],  # Log first 5 symbols
            period=period,
        )

        if len(symbols) > self.config.max_stocks:
            logger.warning(
                "Too many stocks provided",
                provided_count=len(symbols),
                max_stocks=self.config.max_stocks,
                action="limiting to max_stocks",
            )
            symbols = symbols[: self.config.max_stocks]

        # Get historical data for correlation analysis
        logger.debug("Fetching historical data for correlation analysis")
        price_data = self._fetch_price_data(symbols, period)

        # Calculate correlation matrix
        logger.debug("Calculating correlation matrix")
        correlation_matrix = self._calculate_correlation_matrix(price_data)

        # Filter stocks by correlation criteria for diversification
        logger.debug("Filtering stocks for diversification")
        diversified_symbols = self._filter_for_diversification(
            symbols, correlation_matrix
        )

        # Create portfolio stocks
        logger.debug(
            "Creating portfolio stocks", selected_count=len(diversified_symbols)
        )
        portfolio_stocks = self._create_portfolio_stocks(
            diversified_symbols, analysis_results
        )

        # Calculate optimal weights
        logger.debug("Calculating optimal portfolio weights")
        optimized_weights = self._optimize_portfolio_weights(
            portfolio_stocks, correlation_matrix
        )

        # Update portfolio stocks with optimized weights
        self._update_portfolio_weights(portfolio_stocks, optimized_weights)

        # Calculate portfolio metrics
        logger.debug("Calculating portfolio metrics")
        portfolio_metrics = self._calculate_portfolio_metrics(
            portfolio_stocks, correlation_matrix
        )

        # Generate recommendations
        logger.debug("Generating portfolio recommendations")
        recommendations = self._generate_portfolio_recommendations(
            portfolio_stocks, portfolio_metrics
        )

        result = {
            "portfolio_stocks": portfolio_stocks,
            "portfolio_metrics": portfolio_metrics,
            "recommendations": recommendations,
            "correlation_matrix": correlation_matrix,
            "analysis_summary": self._create_analysis_summary(
                portfolio_stocks, portfolio_metrics
            ),
        }

        logger.info(
            "Portfolio analysis completed",
            selected_stocks=len(portfolio_stocks),
            total_allocation=sum(stock.allocation_amount for stock in portfolio_stocks),
            portfolio_risk=portfolio_metrics.risk_score,
        )

        return result

    def _fetch_price_data(self, symbols: List[str], period: str) -> pd.DataFrame:
        """Fetch historical price data for multiple stocks."""
        logger.debug("Fetching price data", symbols=symbols, period=period)

        price_data = pd.DataFrame()

        for symbol in symbols:
            try:
                logger.debug(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if not hist.empty:
                    price_data[symbol] = hist["Close"]
                    logger.debug(
                        f"Fetched data for {symbol}",
                        data_points=len(hist),
                        date_range=f"{hist.index[0].date()} to {hist.index[-1].date()}",
                    )
                else:
                    logger.warning(f"No data available for {symbol}")

            except Exception as e:
                logger.error(
                    f"Failed to fetch data for {symbol}",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )

        logger.debug(
            "Price data fetch completed",
            symbols_with_data=len(price_data.columns),
            total_data_points=len(price_data),
        )

        return price_data

    def _calculate_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix from price data."""
        if price_data.empty:
            logger.warning("Empty price data for correlation calculation")
            return pd.DataFrame()

        # Calculate daily returns
        returns = price_data.pct_change().dropna()

        if returns.empty:
            logger.warning("No returns data for correlation calculation")
            return pd.DataFrame()

        # Calculate correlation matrix
        correlation_matrix = returns.corr()

        logger.debug(
            "Correlation matrix calculated",
            matrix_size=correlation_matrix.shape,
            avg_correlation=correlation_matrix.values[
                np.triu_indices_from(correlation_matrix.values, k=1)
            ].mean()
            if correlation_matrix.size > 1
            else 0,
        )

        return correlation_matrix

    def _filter_for_diversification(
        self, symbols: List[str], correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Filter stocks for portfolio diversification."""
        if correlation_matrix.empty:
            logger.warning("Empty correlation matrix, returning all symbols")
            return symbols

        selected_symbols = []

        # Sort symbols by some criteria (could be investment score)
        # For now, just use the original order
        candidate_symbols = symbols.copy()

        for symbol in candidate_symbols:
            if symbol not in correlation_matrix.columns:
                logger.warning(f"Symbol {symbol} not in correlation matrix")
                continue

            # Check correlation with already selected symbols
            add_to_portfolio = True

            for selected_symbol in selected_symbols:
                if selected_symbol in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[symbol, selected_symbol]

                    if (
                        correlation < self.config.min_correlation
                        or correlation > self.config.max_correlation
                    ):
                        logger.debug(
                            f"Excluding {symbol} due to correlation",
                            correlation_with=selected_symbol,
                            correlation_value=correlation,
                        )
                        add_to_portfolio = False
                        break

            if add_to_portfolio:
                selected_symbols.append(symbol)
                logger.debug(f"Added {symbol} to diversified portfolio")

        logger.info(
            "Diversification filtering completed",
            original_count=len(symbols),
            filtered_count=len(selected_symbols),
            selected_symbols=selected_symbols,
        )

        return selected_symbols

    def _create_portfolio_stocks(
        self, symbols: List[str], analysis_results: Dict[str, AnalysisResult]
    ) -> List[PortfolioStock]:
        """Create portfolio stocks from analysis results."""
        portfolio_stocks = []

        for symbol in symbols:
            if symbol not in analysis_results:
                logger.warning(f"No analysis result for {symbol}")
                continue

            result = analysis_results[symbol]

            # Extract relevant metrics
            expected_return = (
                result.predictions.get("return_5d", 0.0) / 5.0
            )  # Daily return
            risk_score = result.risk_score
            investment_score = result.investment_score
            recommendation = result.recommendation

            portfolio_stock = PortfolioStock(
                symbol=symbol,
                weight=0.0,  # Will be set during optimization
                expected_return=expected_return,
                risk_score=risk_score,
                recommendation=recommendation,
                investment_score=investment_score,
                allocation_amount=0.0,  # Will be calculated from weight
            )

            portfolio_stocks.append(portfolio_stock)
            logger.debug(
                f"Created portfolio stock for {symbol}",
                expected_return=expected_return,
                risk_score=risk_score,
                investment_score=investment_score,
            )

        return portfolio_stocks

    def _optimize_portfolio_weights(
        self, portfolio_stocks: List[PortfolioStock], correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Optimize portfolio weights using simple equal weight approach."""
        # For now, use equal weights as a simple approach
        # In the future, this could be enhanced with Modern Portfolio Theory

        if not portfolio_stocks:
            return {}

        equal_weight = 1.0 / len(portfolio_stocks)
        weights = {stock.symbol: equal_weight for stock in portfolio_stocks}

        # Adjust weights based on investment scores
        total_score = sum(stock.investment_score for stock in portfolio_stocks)

        if total_score > 0:
            for stock in portfolio_stocks:
                score_weight = stock.investment_score / total_score
                # Blend equal weight with score-based weight
                weights[stock.symbol] = 0.5 * equal_weight + 0.5 * score_weight

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {
                symbol: weight / total_weight for symbol, weight in weights.items()
            }

        logger.debug(
            "Portfolio weights optimized",
            weights=weights,
            weight_distribution="score-based equal weighting",
        )

        return weights

    def _update_portfolio_weights(
        self,
        portfolio_stocks: List[PortfolioStock],
        optimized_weights: Dict[str, float],
    ) -> None:
        """Update portfolio stocks with optimized weights."""
        for stock in portfolio_stocks:
            if stock.symbol in optimized_weights:
                stock.weight = optimized_weights[stock.symbol]
                stock.allocation_amount = stock.weight * self.config.investment_amount

                logger.debug(
                    f"Updated weights for {stock.symbol}",
                    weight=stock.weight,
                    allocation_amount=stock.allocation_amount,
                )

    def _calculate_portfolio_metrics(
        self, portfolio_stocks: List[PortfolioStock], correlation_matrix: pd.DataFrame
    ) -> PortfolioMetrics:
        """Calculate portfolio performance metrics."""
        if not portfolio_stocks:
            logger.warning("No portfolio stocks for metrics calculation")
            return PortfolioMetrics(
                total_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                diversification_ratio=0.0,
                risk_score=0.0,
                correlation_matrix=pd.DataFrame(),
            )

        # Calculate portfolio expected return
        portfolio_return = sum(
            stock.weight * stock.expected_return for stock in portfolio_stocks
        )

        # Calculate portfolio risk (simplified)
        portfolio_risk = sum(
            stock.weight * stock.risk_score for stock in portfolio_stocks
        )

        # Calculate basic metrics
        volatility = portfolio_risk / 100.0  # Convert to decimal
        sharpe_ratio = portfolio_return / volatility if volatility > 0 else 0.0

        # Diversification ratio (simplified)
        diversification_ratio = len(portfolio_stocks) / 10.0  # Normalize by max stocks

        metrics = PortfolioMetrics(
            total_return=portfolio_return * 252,  # Annualized
            volatility=volatility * math.sqrt(252),  # Annualized
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.15,  # Placeholder
            diversification_ratio=min(diversification_ratio, 1.0),
            risk_score=portfolio_risk,
            correlation_matrix=correlation_matrix,
        )

        logger.debug(
            "Portfolio metrics calculated",
            total_return=metrics.total_return,
            volatility=metrics.volatility,
            sharpe_ratio=metrics.sharpe_ratio,
            risk_score=metrics.risk_score,
        )

        return metrics

    def _generate_portfolio_recommendations(
        self,
        portfolio_stocks: List[PortfolioStock],
        portfolio_metrics: PortfolioMetrics,
    ) -> Dict[str, Any]:
        """Generate portfolio recommendations."""
        recommendations = {
            "action": "ホールド",
            "reasoning": [],
            "risk_assessment": "中リスク",
            "rebalancing_needed": False,
            "next_review_date": None,
        }

        # Determine overall action based on portfolio metrics
        if portfolio_metrics.sharpe_ratio > 1.0:
            recommendations["action"] = "強い買い"
            recommendations["reasoning"].append("優れたシャープレシオ")
        elif portfolio_metrics.sharpe_ratio > 0.5:
            recommendations["action"] = "買い"
            recommendations["reasoning"].append("良好なシャープレシオ")

        # Risk assessment
        if portfolio_metrics.risk_score < 30:
            recommendations["risk_assessment"] = "低リスク"
        elif portfolio_metrics.risk_score > 70:
            recommendations["risk_assessment"] = "高リスク"

        # Check for rebalancing need
        weight_deviation = max(stock.weight for stock in portfolio_stocks) - min(
            stock.weight for stock in portfolio_stocks
        )
        if weight_deviation > 0.3:  # If weights are too uneven
            recommendations["rebalancing_needed"] = True
            recommendations["reasoning"].append("ポートフォリオのリバランスが必要")

        # Diversification check
        if portfolio_metrics.diversification_ratio < 0.5:
            recommendations["reasoning"].append("分散投資の改善を推奨")

        logger.debug(
            "Portfolio recommendations generated",
            action=recommendations["action"],
            risk_assessment=recommendations["risk_assessment"],
            reasoning_count=len(recommendations["reasoning"]),
        )

        return recommendations

    def _create_analysis_summary(
        self,
        portfolio_stocks: List[PortfolioStock],
        portfolio_metrics: PortfolioMetrics,
    ) -> Dict[str, Any]:
        """Create portfolio analysis summary."""
        summary = {
            "総銘柄数": len(portfolio_stocks),
            "総投資金額": self.config.investment_amount,
            "期待年間リターン": f"{portfolio_metrics.total_return:.2%}",
            "年間ボラティリティ": f"{portfolio_metrics.volatility:.2%}",
            "シャープレシオ": f"{portfolio_metrics.sharpe_ratio:.2f}",
            "ポートフォリオリスクスコア": f"{portfolio_metrics.risk_score:.1f}",
            "分散度": f"{portfolio_metrics.diversification_ratio:.2f}",
            "最大投資銘柄": max(portfolio_stocks, key=lambda x: x.weight).symbol
            if portfolio_stocks
            else "N/A",
            "平均投資スコア": sum(stock.investment_score for stock in portfolio_stocks)
            / len(portfolio_stocks)
            if portfolio_stocks
            else 0,
            "買い推奨数": sum(
                1 for stock in portfolio_stocks if stock.recommendation == "買い"
            ),
            "強い買い推奨数": sum(
                1 for stock in portfolio_stocks if stock.recommendation == "強い買い"
            ),
        }

        logger.info(
            "Portfolio analysis summary created",
            total_stocks=summary["総銘柄数"],
            expected_return=summary["期待年間リターン"],
            portfolio_risk=summary["ポートフォリオリスクスコア"],
        )

        return summary


def compare_stocks(
    analysis_results: Dict[str, AnalysisResult],
    sort_by: str = "investment_score",
    ascending: bool = False,
) -> pd.DataFrame:
    """Compare multiple stocks in a table format.

    Parameters
    ----------
    analysis_results : Dict[str, AnalysisResult]
        Analysis results for multiple stocks
    sort_by : str
        Column to sort by
    ascending : bool
        Sort order

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    logger.debug(
        "Creating stock comparison table",
        stock_count=len(analysis_results),
        sort_by=sort_by,
        ascending=ascending,
    )

    comparison_data = []

    for symbol, result in analysis_results.items():
        row = {
            "銘柄": symbol,
            "現在価格": result.current_price,
            "投資スコア": result.investment_score,
            "リスクスコア": result.risk_score,
            "5日後リターン予測": result.predictions.get("return_5d", 0),
            "30日後リターン予測": result.predictions.get("return_30d", 0),
            "推奨度": result.recommendation,
            "RSI": result.technical_indicators.get("rsi", 0),
            "MACD": result.technical_indicators.get("macd", 0),
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
        logger.debug(f"Sorted comparison table by {sort_by}")

    logger.info(
        "Stock comparison table created",
        rows=len(df),
        top_stock=df.iloc[0]["銘柄"] if not df.empty else "None",
    )

    return df
