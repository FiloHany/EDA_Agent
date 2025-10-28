import os
import sys
from typing import Any, Dict
import pandas as pd
# Adjust system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classes.data_classes import ColumnAnalysis, ColumnType
from interface.Analyzer_interface import IColumnAnalyzer

class CategoricalAnalyzer(IColumnAnalyzer):
    """Analyzer for categorical columns"""
    
    def can_analyze(self, series: pd.Series) -> bool:
        return (pd.api.types.is_object_dtype(series) or 
                pd.api.types.is_categorical_dtype(series) or
                series.nunique() < 20)  # Low cardinality numeric
    
    def analyze(self, series: pd.Series) -> ColumnAnalysis:
        insights = []
        stats = {}
        
        # Value counts
        value_counts = series.value_counts()
        stats['top_values'] = value_counts.head(10).to_dict()
        stats['unique_count'] = series.nunique()
        stats['mode'] = str(series.mode()[0]) if not series.mode().empty else None
        stats['mode_frequency'] = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        stats['mode_percentage'] = round(stats['mode_frequency'] / len(series) * 100, 2)
        
        # Cardinality analysis
        cardinality_ratio = stats['unique_count'] / len(series)
        stats['cardinality_ratio'] = round(cardinality_ratio, 4)
        
        # Generate insights
        if cardinality_ratio > 0.95:
            insights.append(f"Very high cardinality ({stats['unique_count']} unique values)")
        elif cardinality_ratio < 0.01:
            insights.append(f"Very low cardinality ({stats['unique_count']} unique values)")
        
        if stats['mode_percentage'] > 50:
            insights.append(f"Dominated by '{stats['mode']}' ({stats['mode_percentage']:.1f}%)")
        
        # Check for imbalance
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
            if imbalance_ratio > 10:
                insights.append(f"Highly imbalanced (top category is {imbalance_ratio:.1f}x more frequent)")
        
        return ColumnAnalysis(
            name=series.name,
            column_type=ColumnType.CATEGORICAL,
            unique_count=stats['unique_count'],
            missing_count=series.isnull().sum(),
            missing_percentage=round(series.isnull().sum() / len(series) * 100, 2),
            statistics=stats,
            insights=insights
        )
    
    def get_visualization_data(self, series: pd.Series) -> Dict[str, Any]:
        value_counts = series.value_counts().head(20)
        return {
            'type': 'categorical',
            'labels': value_counts.index.tolist(),
            'values': value_counts.values.tolist()
        }
