from .base import Dimension, DimOutput, EventContext
from .d01_technical import TechnicalDimension
from .d02_microstructure import MicrostructureDimension
from .d03_event_history import EventHistoryDimension
from .d04_expectation import ExpectationDimension
from .d05_options_rnd import OptionsRNDDimension
from .d06_options_flow import OptionsFlowDimension
from .d07_peer_sector import PeerSectorDimension
from .d08_macro_regime import MacroRegimeDimension
from .d09_sentiment_nlp import SentimentDimension
from .d10_positioning import PositioningDimension

ALL_DIMENSIONS = [
    OptionsRNDDimension(), EventHistoryDimension(), ExpectationDimension(), PeerSectorDimension(),
    TechnicalDimension(), PositioningDimension(), OptionsFlowDimension(), MacroRegimeDimension(),
    SentimentDimension(), MicrostructureDimension(),
]


def get_dimensions(names: list[str] | None = None, backtest_only: bool = False) -> list[Dimension]:
    dims = ALL_DIMENSIONS
    if backtest_only:
        dims = [d for d in dims if d.supports_backtest]
    if names:
        dims = [d for d in dims if d.name in names]
    return dims
