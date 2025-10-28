from abc import ABC, abstractmethod
import os
import sys
from typing import Any, Dict

import pandas as pd

# Adjust system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classes.data_classes import ColumnAnalysis


class IColumnAnalyzer(ABC):
    """Interface for column analysis strategies"""
    
    @abstractmethod
    def can_analyze(self, series: pd.Series) -> bool:
        """Check if this analyzer can handle the series"""
        pass
    
    @abstractmethod
    def analyze(self, series: pd.Series) -> ColumnAnalysis:
        """Perform analysis on the series"""
        pass
    
    @abstractmethod
    def get_visualization_data(self, series: pd.Series) -> Dict[str, Any]:
        """Get data needed for visualization"""
        pass
