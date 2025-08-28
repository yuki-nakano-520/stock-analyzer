"""Configuration management system."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
import yaml

from ..utils.logging_config import get_logger

logger = get_logger(__name__, module="config")


class ConfigManager:
    """Configuration manager for Stock Analyzer."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.

        Parameters
        ----------
        config_dir : Optional[Path]
            Custom configuration directory. If None, uses default locations.
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_file = self.config_dir / "config.toml"
        self._config_data: Dict[str, Any] = {}
        self._load_config()

        logger.info(
            "ConfigManager initialized",
            config_dir=str(self.config_dir),
            config_file=str(self.config_file),
        )

    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory."""
        # プロジェクトルートの設定を優先、なければホームディレクトリ
        project_config = Path.cwd() / ".stock-analyzer"
        if project_config.exists() or Path.cwd().name == "stock-analyzer":
            return project_config

        # ホームディレクトリの設定
        home_config = Path.home() / ".stock-analyzer"
        return home_config

    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            logger.info("Configuration file not found, creating default")
            self._create_default_config()
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                self._config_data = toml.load(f)
            logger.info(
                "Configuration loaded successfully",
                sections=list(self._config_data.keys()),
            )
        except Exception as e:
            logger.error(
                "Failed to load configuration",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        logger.info("Creating default configuration")

        self.config_dir.mkdir(parents=True, exist_ok=True)

        default_config = {
            "general": {
                "default_period": "1y",
                "default_investment_amount": 100000.0,
                "default_max_stocks": 10,
                "default_risk_tolerance": 0.3,
                "auto_export_csv": False,
                "log_level": "INFO",
            },
            "api": {
                "alpha_vantage_api_key": "",
                "request_timeout": 30,
                "max_retries": 3,
            },
            "analysis": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2.0,
                "volume_period": 20,
            },
            "portfolio": {
                "min_correlation": -0.5,
                "max_correlation": 0.7,
                "rebalance_period": 30,
            },
            "output": {
                "csv_encoding": "utf-8-sig",
                "decimal_places": 2,
                "include_timestamp": True,
                "output_directory": "output",
            },
            "default_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "watchlists": {
                "default": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
                "tech": [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "META",
                    "NVDA",
                    "NFLX",
                    "CRM",
                    "ADBE",
                    "INTC",
                ],
                "finance": [
                    "JPM",
                    "BAC",
                    "WFC",
                    "C",
                    "GS",
                    "MS",
                    "AXP",
                    "BK",
                    "USB",
                    "PNC",
                ],
                "healthcare": [
                    "JNJ",
                    "UNH",
                    "PFE",
                    "MRK",
                    "ABBV",
                    "TMO",
                    "DHR",
                    "BMY",
                    "LLY",
                    "MDT",
                ],
            },
        }

        self._config_data = default_config
        self.save_config()

        logger.info("Default configuration created successfully")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Parameters
        ----------
        key : str
            Configuration key in dot notation (e.g., 'general.default_period')
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split(".")
        value = self._config_data

        try:
            for k in keys:
                value = value[k]
            logger.debug(f"Retrieved config value: {key} = {value}")
            return value
        except (KeyError, TypeError):
            logger.debug(f"Config key not found: {key}, using default: {default}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Parameters
        ----------
        key : str
            Configuration key in dot notation
        value : Any
            Value to set
        """
        keys = key.split(".")
        config = self._config_data

        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

        logger.debug(f"Set config value: {key} = {value}")

    def get_watchlist(self, name: str) -> List[str]:
        """Get symbols from named watchlist.

        Parameters
        ----------
        name : str
            Watchlist name

        Returns
        -------
        List[str]
            List of symbols
        """
        watchlists = self.get("watchlists", {})
        symbols = watchlists.get(name, [])

        logger.debug(
            f"Retrieved watchlist '{name}'",
            symbol_count=len(symbols),
            symbols=symbols[:5],  # Log first 5 symbols
        )

        return symbols

    def add_watchlist(self, name: str, symbols: List[str]) -> None:
        """Add or update a watchlist.

        Parameters
        ----------
        name : str
            Watchlist name
        symbols : List[str]
            List of symbols
        """
        if "watchlists" not in self._config_data:
            self._config_data["watchlists"] = {}

        self._config_data["watchlists"][name] = symbols

        logger.info(f"Added/updated watchlist '{name}'", symbol_count=len(symbols))

    def list_watchlists(self) -> List[str]:
        """List available watchlist names.

        Returns
        -------
        List[str]
            List of watchlist names
        """
        watchlists = self.get("watchlists", {})
        return list(watchlists.keys())

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                toml.dump(self._config_data, f)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(
                "Failed to save configuration",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def load_symbols_from_file(self, file_path: str) -> List[str]:
        """Load symbols from various file formats.

        Parameters
        ----------
        file_path : str
            Path to symbols file (.txt, .csv, .yaml, .json)

        Returns
        -------
        List[str]
            List of symbols
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Symbols file not found: {file_path}")

        logger.debug(f"Loading symbols from file: {file_path}")

        try:
            if path.suffix.lower() == ".txt":
                symbols = self._load_symbols_from_txt(path)
            elif path.suffix.lower() == ".csv":
                symbols = self._load_symbols_from_csv(path)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                symbols = self._load_symbols_from_yaml(path)
            elif path.suffix.lower() == ".json":
                symbols = self._load_symbols_from_json(path)
            else:
                # Default to text format
                symbols = self._load_symbols_from_txt(path)

            # Clean and validate symbols
            symbols = [s.upper().strip() for s in symbols if s.strip()]
            symbols = list(
                dict.fromkeys(symbols)
            )  # Remove duplicates while preserving order

            logger.info(
                "Loaded symbols from file",
                file_path=file_path,
                symbol_count=len(symbols),
                symbols=symbols[:10],  # Log first 10 symbols
            )

            return symbols

        except Exception as e:
            logger.error(
                "Failed to load symbols from file",
                file_path=file_path,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def _load_symbols_from_txt(self, path: Path) -> List[str]:
        """Load symbols from text file (one per line)."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        symbols = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                # Handle comma-separated values in single line
                if "," in line:
                    symbols.extend([s.strip() for s in line.split(",")])
                else:
                    symbols.append(line)

        return symbols

    def _load_symbols_from_csv(self, path: Path) -> List[str]:
        """Load symbols from CSV file."""
        import csv

        with open(path, "r", encoding="utf-8") as f:
            # Try to detect if first line is header
            sample = f.read(1024)
            f.seek(0)

            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample)

            reader = csv.reader(f)
            if has_header:
                next(reader)  # Skip header

            symbols = []
            for row in reader:
                if row:  # Skip empty rows
                    # Take first column as symbol
                    symbol = row[0].strip()
                    if symbol and not symbol.startswith("#"):
                        symbols.append(symbol)

        return symbols

    def _load_symbols_from_yaml(self, path: Path) -> List[str]:
        """Load symbols from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Look for common keys
            for key in ["symbols", "stocks", "tickers"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Return first list value found
            for value in data.values():
                if isinstance(value, list):
                    return value

        raise ValueError("Could not find symbol list in YAML file")

    def _load_symbols_from_json(self, path: Path) -> List[str]:
        """Load symbols from JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Look for common keys
            for key in ["symbols", "stocks", "tickers"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Return first list value found
            for value in data.values():
                if isinstance(value, list):
                    return value

        raise ValueError("Could not find symbol list in JSON file")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance.

    Returns
    -------
    ConfigManager
        Global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
