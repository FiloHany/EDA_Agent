import json
from typing import List
from langchain_core.tools import tool

from classes.data_classes import ColumnType
from core.EDA.eda_service import EDAService
from core.report.report_generator import ReportBuilder

class EDATools:
    """Tools for LLM interaction with EDA service"""
    
    def __init__(self, eda_service: EDAService):
        self.eda = eda_service
    
    @staticmethod
    def create_tools(eda_service: EDAService) -> List:
        """Factory method to create all tools"""
        tools_instance = EDATools(eda_service)
        
        @tool
        def get_dataset_overview(query: str = "") -> str:
            """Get basic dataset information: shape, columns, memory usage.
            Use this first to understand the dataset structure."""
            metadata = tools_instance.eda.get_metadata()
            return f"""Dataset Overview:
- Rows: {metadata.row_count:,}
- Columns: {metadata.column_count}
- Memory: {metadata.memory_usage_mb:.2f} MB
- Column Names: {', '.join(metadata.columns)}
- Data Types: {json.dumps(metadata.dtypes, indent=2)}"""
        
        @tool
        def get_data_quality(query: str = "") -> str:
            """Get data quality metrics: missing values, duplicates, completeness.
            Use this to identify data quality issues."""
            quality = tools_instance.eda.get_quality_report()
            result = f"""Data Quality Report:
- Total Cells: {quality.total_cells:,}
- Missing Cells: {quality.missing_cells:,} ({quality.missing_cells/quality.total_cells*100:.2f}%)
- Duplicate Rows: {quality.duplicate_rows:,} ({quality.duplicate_percentage:.2f}%)
"""
            if quality.missing_values:
                result += "\nColumns with Missing Values:\n"
                for col, count in quality.missing_values.items():
                    pct = quality.missing_percentages[col]
                    result += f"  - {col}: {count:,} missing ({pct:.1f}%)\n"
            return result
        
        @tool
        def analyze_column(column_name: str) -> str:
            """Analyze a specific column in detail.
            Input: Column name (e.g., 'age', 'sex', 'fare')
            Returns: Statistics, insights, and distribution info for that column."""
            if not column_name or column_name.strip() == "":
                return "Error: Please provide a column name"
            
            col = column_name.strip()
            if col not in tools_instance.eda.df.columns:
                available = ', '.join(tools_instance.eda.df.columns.tolist()[:10])
                return f"Error: Column '{col}' not found. Available: {available}..."
            
            analyzer = tools_instance.eda.factory.get_analyzer(tools_instance.eda.df[col])
            analysis = analyzer.analyze(tools_instance.eda.df[col])
            
            result = f"""Column Analysis: {analysis.name}
- Type: {analysis.column_type.value}
- Unique Values: {analysis.unique_count:,}
- Missing: {analysis.missing_count:,} ({analysis.missing_percentage:.1f}%)

Statistics:
{json.dumps(analysis.statistics, indent=2, default=str)}

Insights:
"""
            if analysis.insights:
                result += '\n'.join(f"  - {insight}" for insight in analysis.insights)
            else:
                result += "  - No significant insights detected"
            
            return result
        
        @tool
        def get_correlations(threshold: str = "0.7") -> str:
            """Get highly correlated feature pairs.
            Input: Correlation threshold (default: 0.7)
            Returns: Pairs of features with correlation above threshold."""
            try:
                thresh = float(threshold) if threshold else 0.7
            except:
                thresh = 0.7
            
            high_corrs = tools_instance.eda.get_high_correlations(threshold=thresh)
            
            if not high_corrs:
                return f"No correlations found above threshold {thresh}"
            
            result = f"Highly Correlated Features (|r| > {thresh}):\n\n"
            for col1, col2, corr in high_corrs:
                result += f"  - {col1} ↔ {col2}: r = {corr:.3f}\n"
            
            return result
        
        @tool
        def get_automated_insights(query: str = "") -> str:
            """Get automated insights and recommendations for the dataset.
            Returns: Key findings, patterns, and suggested next steps."""
            insights = tools_instance.eda.generate_insights()
            
            result = "Automated Insights:\n\n"
            for category, items in insights.items():
                if items:
                    result += f"{category.replace('_', ' ').title()}:\n"
                    for item in items:
                        result += f"  • {item}\n"
                    result += "\n"
            
            return result
        
        @tool
        def generate_full_report(format_type: str = "markdown") -> str:
            """Generate a complete EDA report.
            Input: Format type ('markdown' or 'json')
            Returns: Full EDA report with all sections."""
            builder = ReportBuilder(tools_instance.eda)
            builder.add_metadata_section()\
                   .add_quality_section()\
                   .add_column_analysis_section()\
                   .add_insights_section()
            
            if format_type.lower() == "json":
                return builder.build_json()
            return builder.build_markdown()
        
        @tool
        def compare_columns(columns: str) -> str:
            """Compare statistics between multiple columns.
            Input: Comma-separated column names (e.g., 'age,fare,pclass')
            Returns: Comparative statistics for the specified columns."""
            if not columns or columns.strip() == "":
                return "Error: Please provide comma-separated column names"
            
            cols = [c.strip() for c in columns.split(",")]
            invalid = [c for c in cols if c not in tools_instance.eda.df.columns]
            
            if invalid:
                return f"Error: Invalid columns: {', '.join(invalid)}"
            
            result = "Column Comparison:\n\n"
            for col in cols:
                analyzer = tools_instance.eda.factory.get_analyzer(tools_instance.eda.df[col])
                analysis = analyzer.analyze(tools_instance.eda.df[col])
                result += f"{col} ({analysis.column_type.value}):\n"
                result += f"  - Unique: {analysis.unique_count:,}\n"
                result += f"  - Missing: {analysis.missing_percentage:.1f}%\n"
                if analysis.column_type == ColumnType.NUMERIC:
                    result += f"  - Mean: {analysis.statistics.get('mean', 'N/A'):.2f}\n"
                    result += f"  - Std: {analysis.statistics.get('std', 'N/A'):.2f}\n"
                result += "\n"
            
            return result
        
        return [
            get_dataset_overview,
            get_data_quality,
            analyze_column,
            get_correlations,
            get_automated_insights,
            generate_full_report,
            compare_columns
        ]