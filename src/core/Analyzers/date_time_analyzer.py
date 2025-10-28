import os
import sys
from typing import Any, Dict
import pandas as pd
# Adjust system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classes.data_classes import ColumnAnalysis, ColumnType
from interface.Analyzer_interface import IColumnAnalyzer


class DatetimeAnalyzer(IColumnAnalyzer):
    """Analyzer for datetime columns"""
    
    def can_analyze(self, series: pd.Series) -> bool:
        return pd.api.types.is_datetime64_any_dtype(series)
    
    def analyze(self, series: pd.Series) -> ColumnAnalysis:
        insights = []
        stats = {}
        
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return ColumnAnalysis(
                name=series.name,
                column_type=ColumnType.DATETIME,
                unique_count=0,
                missing_count=series.isnull().sum(),
                missing_percentage=100.0,
                statistics={},
                insights=["All values are missing"]
            )
        
        stats['min_date'] = str(clean_series.min())
        stats['max_date'] = str(clean_series.max())
        stats['date_range_days'] = (clean_series.max() - clean_series.min()).days
        
        # Extract components
        stats['year_range'] = f"{clean_series.dt.year.min()} - {clean_series.dt.year.max()}"
        
        # Generate insights
        if stats['date_range_days'] > 365:
            insights.append(f"Spans {stats['date_range_days']} days ({stats['date_range_days']//365} years)")
        
        return ColumnAnalysis(
            name=series.name,
            column_type=ColumnType.DATETIME,
            unique_count=series.nunique(),
            missing_count=series.isnull().sum(),
            missing_percentage=round(series.isnull().sum() / len(series) * 100, 2),
            statistics=stats,
            insights=insights
        )
    
    def get_visualization_data(self, series: pd.Series) -> Dict[str, Any]:
        return {
            'type': 'datetime',
            'values': series.dropna().dt.strftime('%Y-%m-%d').tolist()
        }
