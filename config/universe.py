"""Peer groups and sector ETF mapping. Extend freely; unknown tickers fall back to sector ETFs."""
from __future__ import annotations

PEERS = {
    "MU":   ["SNDK", "WDC", "STX", "NVDA", "AMD"],
    "SNDK": ["MU", "WDC", "STX", "AMD"],
    "WDC":  ["SNDK", "STX", "MU"],
    "STX":  ["WDC", "SNDK", "MU"],
    "AMD":  ["NVDA", "INTC", "AVGO", "MU", "MRVL"],
    "NVDA": ["AMD", "AVGO", "TSM", "MRVL"],
    "AVGO": ["NVDA", "AMD", "MRVL", "QCOM"],
    "INTC": ["AMD", "NVDA", "QCOM", "TXN"],
    "AMAT": ["LRCX", "KLAC", "ASML"],
    "LRCX": ["AMAT", "KLAC", "ASML"],
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "AMZN": ["MSFT", "GOOGL", "WMT"],
    "TSLA": ["RIVN", "GM", "F"],
    "RBLX": ["EA", "TTWO", "U"],
    "NFLX": ["DIS", "WBD", "SPOT"],
}

SECTOR_ETF_BY_TICKER = {
    "MU": "SMH", "SNDK": "SMH", "WDC": "SMH", "STX": "SMH", "AMD": "SMH", "NVDA": "SMH",
    "AVGO": "SMH", "INTC": "SMH", "AMAT": "SMH", "LRCX": "SMH", "MRVL": "SMH", "QCOM": "SMH",
    "AAPL": "XLK", "MSFT": "XLK", "ORCL": "XLK", "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "RBLX": "XLC",
}

SECTOR_TO_ETF = {
    "Technology": "XLK", "Communication Services": "XLC", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Healthcare": "XLV", "Financial Services": "XLF",
    "Industrials": "XLI", "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}

INDUSTRY_TO_ETF = {
    "Semiconductors": "SMH", "Semiconductor Equipment & Materials": "SMH",
    "Software - Application": "IGV", "Software - Infrastructure": "IGV",
    "Biotechnology": "XBI", "Banks - Regional": "KRE", "Oil & Gas E&P": "XOP",
    "Computer Hardware": "SMH",
}


def get_peers(ticker: str, info: dict | None = None) -> tuple[list[str], str]:
    """Return (peer_tickers, sector_etf)."""
    t = ticker.upper()
    info = info or {}
    peers = PEERS.get(t, [])
    etf = SECTOR_ETF_BY_TICKER.get(t)
    if etf is None:
        etf = INDUSTRY_TO_ETF.get(info.get("industry") or "", None) or SECTOR_TO_ETF.get(info.get("sector") or "", "SPY")
    return peers, etf
