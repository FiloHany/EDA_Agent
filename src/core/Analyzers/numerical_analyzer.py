import os
import sys
from typing import Any, Dict
import pandas as pd

# Adjust system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classes.data_classes import ColumnAnalysis, ColumnType
from interface.Analyzer_interface import IColumnAnalyzer


class NumericAnalyzer(IColumnAnalyzer):
    """Analyzer for numeric columns"""
    
    def can_analyze(self, series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series) and not pd.api.types.is_timedelta64_dtype(series) and not pd.api.types.is_bool_dtype(series)
    
    def analyze(self, series: pd.Series) -> ColumnAnalysis:
        insights = []
        stats = {}
        
        # Basic statistics
        stats['mean'] = float(series.mean())
        stats['median'] = float(series.median())
        stats['std'] = float(series.std())
        stats['min'] = float(series.min())
        stats['max'] = float(series.max())
        stats['q25'] = float(series.quantile(0.25))
        stats['q75'] = float(series.quantile(0.75))
        
        # Skewness & Kurtosis
        stats['skewness'] = float(series.skew())
        stats['kurtosis'] = float(series.kurtosis())
        
        # Outlier detection (IQR method)
        q1, q3 = stats['q25'], stats['q75']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        stats['outlier_count'] = len(outliers)
        stats['outlier_percentage'] = round(len(outliers) / len(series) * 100, 2)
        
        # Generate insights
        if abs(stats['skewness']) > 1:
            direction = "right" if stats['skewness'] > 0 else "left"
            insights.append(f"Highly {direction}-skewed distribution (skew={stats['skewness']:.2f})")
        
        if stats['outlier_count'] > 0:
            insights.append(f"Contains {stats['outlier_count']} outliers ({stats['outlier_percentage']:.1f}%)")
        
        if stats['std'] > stats['mean']:
            insights.append("High variability (std > mean)")
        
        return ColumnAnalysis(
            name=series.name,
            column_type=ColumnType.NUMERIC,
            unique_count=series.nunique(),
            missing_count=series.isnull().sum(),
            missing_percentage=round(series.isnull().sum() / len(series) * 100, 2),
            statistics=stats,
            insights=insights
        )
    
    def get_visualization_data(self, series: pd.Series) -> Dict[str, Any]:
        return {
            'type': 'numeric',
            'values': series.dropna().tolist(),
            'bins': min(50, series.nunique())
        }

