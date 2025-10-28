import pandas as pd

from interface.Analyzer_interface import IColumnAnalyzer
from core.Analyzers.categorical_analyzer import CategoricalAnalyzer
from core.Analyzers.date_time_analyzer import DatetimeAnalyzer
from core.Analyzers.numerical_analyzer import NumericAnalyzer


class AnalyzerFactory:
    """Factory for creating appropriate analyzers"""
    
    def __init__(self):
        self._analyzers = [
            DatetimeAnalyzer(),
            NumericAnalyzer(),
            CategoricalAnalyzer(),
        ]
    
    def get_analyzer(self, series: pd.Series) -> IColumnAnalyzer:
        """Get the appropriate analyzer for a series"""
        for analyzer in self._analyzers:
            if analyzer.can_analyze(series):
                return analyzer
        return CategoricalAnalyzer()  # Fallback
