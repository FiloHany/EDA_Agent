from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from classes.data_classes import ColumnAnalysis, ColumnType, DataQualityReport, DatasetMetadata
from core.Analyzers.factory_analyzer import AnalyzerFactory


class EDAService:
    """Main EDA service coordinating all analyses"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.factory = AnalyzerFactory()
        self._metadata: Optional[DatasetMetadata] = None
        self._quality_report: Optional[DataQualityReport] = None
        self._column_analyses: Optional[List[ColumnAnalysis]] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
    
    def get_metadata(self) -> DatasetMetadata:
        """Extract dataset metadata"""
        if self._metadata is None:
            memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
            self._metadata = DatasetMetadata(
                shape=self.df.shape,
                memory_usage_mb=round(memory_mb, 2),
                column_count=len(self.df.columns),
                row_count=len(self.df),
                columns=self.df.columns.tolist(),
                dtypes={col: str(dtype) for col, dtype in self.df.dtypes.items()}
            )
        return self._metadata
    
    def get_quality_report(self) -> DataQualityReport:
        """Generate data quality report"""
        if self._quality_report is None:
            missing = self.df.isnull().sum()
            total_cells = self.df.shape[0] * self.df.shape[1]
            missing_cells = missing.sum()
            duplicates = self.df.duplicated().sum()
            
            self._quality_report = DataQualityReport(
                missing_values={col: int(count) for col, count in missing.items() if count > 0},
                missing_percentages={col: round(count/len(self.df)*100, 2) 
                                   for col, count in missing.items() if count > 0},
                duplicate_rows=int(duplicates),
                duplicate_percentage=round(duplicates/len(self.df)*100, 2),
                total_cells=total_cells,
                missing_cells=int(missing_cells)
            )
        return self._quality_report
    
    def analyze_columns(self) -> List[ColumnAnalysis]:
        """Analyze all columns"""
        if self._column_analyses is None:
            self._column_analyses = []
            for col in self.df.columns:
                analyzer = self.factory.get_analyzer(self.df[col])
                analysis = analyzer.analyze(self.df[col])
                self._column_analyses.append(analysis)
        return self._column_analyses
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for numeric columns"""
        if self._correlation_matrix is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                self._correlation_matrix = self.df[numeric_cols].corr()
        return self._correlation_matrix
    
    def get_high_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs"""
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None:
            return []
        
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        round(corr_val, 3)
                    ))
        return sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True)
    
    def generate_insights(self) -> Dict[str, Any]:
        """Generate automated insights"""
        insights = {
            'data_quality': [],
            'distributions': [],
            'correlations': [],
            'recommendations': []
        }
        
        # Quality insights
        quality = self.get_quality_report()
        if quality.missing_cells > 0:
            pct = round(quality.missing_cells / quality.total_cells * 100, 2)
            insights['data_quality'].append(f"Dataset has {pct}% missing values")
        
        if quality.duplicate_rows > 0:
            insights['data_quality'].append(
                f"Found {quality.duplicate_rows} duplicate rows ({quality.duplicate_percentage:.1f}%)"
            )
        
        # Distribution insights
        analyses = self.analyze_columns()
        for analysis in analyses:
            if analysis.insights:
                insights['distributions'].extend([
                    f"{analysis.name}: {insight}" for insight in analysis.insights
                ])
        
        # Correlation insights
        high_corrs = self.get_high_correlations()
        for col1, col2, corr in high_corrs[:5]:  # Top 5
            insights['correlations'].append(
                f"Strong correlation between '{col1}' and '{col2}' (r={corr})"
            )
        
        # Recommendations
        if quality.missing_percentages:
            high_missing = [col for col, pct in quality.missing_percentages.items() if pct > 30]
            if high_missing:
                insights['recommendations'].append(
                    f"Consider dropping columns with >30% missing: {', '.join(high_missing)}"
                )
        
        skewed_features = [a.name for a in analyses 
                          if a.column_type == ColumnType.NUMERIC 
                          and abs(a.statistics.get('skewness', 0)) > 1]
        if skewed_features:
            insights['recommendations'].append(
                f"Apply log/box-cox transformation to skewed features: {', '.join(skewed_features[:5])}"
            )
        
        if high_corrs:
            insights['recommendations'].append(
                "Consider removing redundant features or using PCA for highly correlated variables"
            )
        
        return insights