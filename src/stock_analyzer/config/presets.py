"""Predefined symbol presets for common investment themes."""

from typing import Any, Dict, List

from ..utils.logging_config import get_logger

logger = get_logger(__name__, module="presets")


class PresetManager:
    """Manager for predefined symbol groups."""

    def __init__(self):
        """Initialize preset manager with built-in presets."""
        self._presets: Dict[str, List[str]] = {
            # Technology Giants
            "tech-giants": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
                "NVDA",
                "NFLX",
            ],
            # FAANG stocks
            "faang": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
            # Magnificent Seven
            "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"],
            # S&P 500 Top 20 (by market cap)
            "sp500-top20": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "NVDA",
                "TSLA",
                "META",
                "BRK-B",
                "UNH",
                "JNJ",
                "AVGO",
                "XOM",
                "LLY",
                "JPM",
                "V",
                "PG",
                "MA",
                "HD",
                "CVX",
                "COST",
            ],
            # Financial Sector
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
                "TFC",
                "COF",
                "SCHW",
                "BLK",
                "SPGI",
            ],
            # Healthcare & Biotech
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
                "GILD",
                "AMGN",
                "CVS",
                "CI",
                "HCA",
            ],
            # Consumer Goods
            "consumer": [
                "PG",
                "KO",
                "PEP",
                "WMT",
                "TGT",
                "COST",
                "HD",
                "LOW",
                "NKE",
                "SBUX",
                "MCD",
                "DIS",
                "NIKE",
                "LULU",
                "TJX",
            ],
            # Energy Sector
            "energy": [
                "XOM",
                "CVX",
                "COP",
                "EOG",
                "SLB",
                "PSX",
                "VLO",
                "MPC",
                "KMI",
                "OKE",
                "WMB",
                "EPD",
                "BKR",
                "HAL",
                "DVN",
            ],
            # AI & Semiconductor
            "ai-semicon": [
                "NVDA",
                "AMD",
                "INTC",
                "TSM",
                "AVGO",
                "QCOM",
                "TXN",
                "ADI",
                "MCHP",
                "MU",
                "LRCX",
                "AMAT",
                "KLAC",
                "NXPI",
                "MRVL",
            ],
            # Cloud & Software
            "cloud-software": [
                "MSFT",
                "GOOGL",
                "AMZN",
                "CRM",
                "ORCL",
                "ADBE",
                "NOW",
                "INTU",
                "WDAY",
                "ZM",
                "DDOG",
                "SNOW",
                "PLTR",
                "CRWD",
                "ZS",
            ],
            # E-commerce & Retail
            "ecommerce": [
                "AMZN",
                "WMT",
                "TGT",
                "COST",
                "HD",
                "LOW",
                "SHOP",
                "EBAY",
                "ETSY",
                "W",
                "CHWY",
                "BABA",
                "JD",
                "PDD",
                "SE",
            ],
            # Electric Vehicles & Clean Energy
            "ev-clean": [
                "TSLA",
                "NIO",
                "XPEV",
                "LI",
                "RIVN",
                "LCID",
                "FSR",
                "ENPH",
                "SEDG",
                "FSLR",
                "NEE",
                "DUK",
                "SO",
                "EXC",
                "AEP",
            ],
            # REITs
            "reits": [
                "AMT",
                "CCI",
                "PLD",
                "EQIX",
                "PSA",
                "EXR",
                "AVB",
                "EQR",
                "DLR",
                "SPG",
                "O",
                "VTR",
                "WELL",
                "UDR",
                "ESS",
            ],
            # Dividend Aristocrats (Sample)
            "dividend-aristocrats": [
                "KO",
                "PG",
                "JNJ",
                "MMM",
                "CAT",
                "MCD",
                "WMT",
                "CVX",
                "XOM",
                "T",
                "VZ",
                "IBM",
                "GD",
                "LMT",
                "RTX",
            ],
            # Growth Stocks
            "growth": [
                "TSLA",
                "NVDA",
                "AMZN",
                "NFLX",
                "CRM",
                "SHOP",
                "ZM",
                "PTON",
                "ROKU",
                "SQ",
                "PYPL",
                "TWLO",
                "OKTA",
                "DDOG",
                "SNOW",
            ],
            # Value Stocks
            "value": [
                "BRK-B",
                "JPM",
                "JNJ",
                "PG",
                "KO",
                "XOM",
                "CVX",
                "WMT",
                "VZ",
                "T",
                "IBM",
                "INTC",
                "F",
                "GM",
                "BAC",
            ],
            # Defensive/Recession-Resistant
            "defensive": [
                "JNJ",
                "PG",
                "KO",
                "WMT",
                "COST",
                "UNH",
                "MRK",
                "PFE",
                "VZ",
                "T",
                "NEE",
                "DUK",
                "SO",
                "AWK",
                "ATO",
            ],
            # International/ADR
            "international": [
                "TSM",
                "ASML",
                "NVO",
                "UL",
                "NESN",
                "RHHBY",
                "TM",
                "SNY",
                "DEO",
                "BUD",
                "BABA",
                "TCEHY",
                "NTT",
                "SONY",
                "SAP",
            ],
            # Small-Mid Cap Growth
            "small-mid-growth": [
                "ROKU",
                "PTON",
                "ZM",
                "SHOP",
                "SQ",
                "TWLO",
                "OKTA",
                "DDOG",
                "SNOW",
                "CRWD",
                "ZS",
                "NET",
                "FSLY",
                "MDB",
                "TEAM",
            ],
        }

        logger.info(
            "PresetManager initialized",
            preset_count=len(self._presets),
            presets=list(self._presets.keys()),
        )

    def get_preset(self, name: str) -> List[str]:
        """Get symbols from a preset.

        Parameters
        ----------
        name : str
            Preset name

        Returns
        -------
        List[str]
            List of symbols

        Raises
        ------
        KeyError
            If preset name not found
        """
        if name not in self._presets:
            available = list(self._presets.keys())
            logger.error(
                f"Preset '{name}' not found",
                available_presets=available[:10],  # Log first 10 available presets
            )
            raise KeyError(
                f"Preset '{name}' not found. Available presets: {', '.join(available)}"
            )

        symbols = self._presets[name]
        logger.debug(
            f"Retrieved preset '{name}'",
            symbol_count=len(symbols),
            symbols=symbols[:5],  # Log first 5 symbols
        )

        return symbols.copy()  # Return a copy to prevent external modification

    def list_presets(self) -> List[str]:
        """Get list of available preset names.

        Returns
        -------
        List[str]
            List of preset names
        """
        return list(self._presets.keys())

    def get_preset_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a preset.

        Parameters
        ----------
        name : str
            Preset name

        Returns
        -------
        Dict[str, any]
            Preset information including symbol count and description
        """
        if name not in self._presets:
            raise KeyError(f"Preset '{name}' not found")

        symbols = self._presets[name]

        # Generate description based on preset name
        descriptions = {
            "tech-giants": "Major technology companies (Apple, Microsoft, Google, etc.)",
            "faang": "Facebook, Apple, Amazon, Netflix, Google stocks",
            "mag7": "The Magnificent Seven mega-cap tech stocks",
            "sp500-top20": "Top 20 companies in S&P 500 by market capitalization",
            "finance": "Major banks and financial services companies",
            "healthcare": "Healthcare, pharmaceutical, and biotech companies",
            "consumer": "Consumer goods and retail companies",
            "energy": "Oil, gas, and energy sector companies",
            "ai-semicon": "AI-focused and semiconductor companies",
            "cloud-software": "Cloud computing and software companies",
            "ecommerce": "E-commerce and online retail companies",
            "ev-clean": "Electric vehicle and clean energy companies",
            "reits": "Real Estate Investment Trusts",
            "dividend-aristocrats": "Companies with 25+ years of dividend increases",
            "growth": "High-growth technology and emerging companies",
            "value": "Undervalued companies with strong fundamentals",
            "defensive": "Recession-resistant, stable companies",
            "international": "International companies via ADRs",
            "small-mid-growth": "Small to mid-cap growth companies",
        }

        return {
            "name": name,
            "symbol_count": len(symbols),
            "symbols": symbols,
            "description": descriptions.get(
                name, f"Preset containing {len(symbols)} symbols"
            ),
        }

    def search_presets(self, query: str) -> List[str]:
        """Search presets by name or theme.

        Parameters
        ----------
        query : str
            Search query

        Returns
        -------
        List[str]
            List of matching preset names
        """
        query_lower = query.lower()
        matching_presets = []

        for preset_name in self._presets:
            if query_lower in preset_name.lower():
                matching_presets.append(preset_name)

        logger.debug(
            f"Preset search for '{query}'",
            matches_found=len(matching_presets),
            matches=matching_presets,
        )

        return matching_presets

    def add_custom_preset(self, name: str, symbols: List[str]) -> None:
        """Add a custom preset.

        Parameters
        ----------
        name : str
            Preset name
        symbols : List[str]
            List of symbols
        """
        # Validate and clean symbols
        clean_symbols = [s.upper().strip() for s in symbols if s.strip()]
        clean_symbols = list(dict.fromkeys(clean_symbols))  # Remove duplicates

        self._presets[name] = clean_symbols

        logger.info(f"Added custom preset '{name}'", symbol_count=len(clean_symbols))

    def remove_preset(self, name: str) -> None:
        """Remove a preset.

        Parameters
        ----------
        name : str
            Preset name to remove
        """
        if name not in self._presets:
            raise KeyError(f"Preset '{name}' not found")

        del self._presets[name]
        logger.info(f"Removed preset '{name}'")


# Global preset manager instance
_preset_manager: PresetManager | None = None


def get_preset_symbols(preset_name: str) -> List[str]:
    """Get symbols from a preset (convenience function).

    Parameters
    ----------
    preset_name : str
        Name of the preset

    Returns
    -------
    List[str]
        List of symbols
    """
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()

    return _preset_manager.get_preset(preset_name)


def list_available_presets() -> List[str]:
    """List all available presets (convenience function).

    Returns
    -------
    List[str]
        List of available preset names
    """
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()

    return _preset_manager.list_presets()
