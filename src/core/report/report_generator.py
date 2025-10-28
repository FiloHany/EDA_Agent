from dataclasses import asdict
import json
import os
import sys

# Adjust system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from EDA.eda_service import EDAService


class ReportBuilder:
    """Builds EDA reports in various formats"""
    
    def __init__(self, eda_service: EDAService):
        self.eda = eda_service
        self.sections = []
    
    def add_metadata_section(self) -> 'ReportBuilder':
        """Add metadata section"""
        metadata = self.eda.get_metadata()
        section = f"""
## 📊 Dataset Metadata

- **Shape**: {metadata.row_count:,} rows × {metadata.column_count} columns
- **Memory Usage**: {metadata.memory_usage_mb:.2f} MB
- **Columns**: {', '.join(metadata.columns[:10])}{'...' if len(metadata.columns) > 10 else ''}
"""
        self.sections.append(section)
        return self
    
    def add_quality_section(self) -> 'ReportBuilder':
        """Add data quality section"""
        quality = self.eda.get_quality_report()
        section = f"""
## 🔍 Data Quality Report

- **Missing Values**: {quality.missing_cells:,} cells ({quality.missing_cells/quality.total_cells*100:.2f}%)
- **Duplicate Rows**: {quality.duplicate_rows:,} ({quality.duplicate_percentage:.2f}%)

"""
        if quality.missing_values:
            section += "**Columns with Missing Values:**\n"
            for col, count in list(quality.missing_values.items())[:10]:
                pct = quality.missing_percentages[col]
                section += f"- {col}: {count:,} ({pct:.1f}%)\n"
        
        self.sections.append(section)
        return self
    
    def add_column_analysis_section(self) -> 'ReportBuilder':
        """Add column analysis section"""
        analyses = self.eda.analyze_columns()
        section = "\n## 📈 Column Analysis\n\n"
        
        for analysis in analyses[:10]:  # Limit for brevity
            section += f"### {analysis.name} ({analysis.column_type.value})\n"
            section += f"- **Unique Values**: {analysis.unique_count:,}\n"
            section += f"- **Missing**: {analysis.missing_count:,} ({analysis.missing_percentage:.1f}%)\n"
            
            if analysis.insights:
                section += f"- **Insights**: {'; '.join(analysis.insights)}\n"
            
            section += "\n"
        
        self.sections.append(section)
        return self
    
    def add_insights_section(self) -> 'ReportBuilder':
        """Add automated insights section"""
        insights = self.eda.generate_insights()
        section = "\n## 💡 Automated Insights\n\n"
        
        for category, items in insights.items():
            if items:
                section += f"**{category.replace('_', ' ').title()}:**\n"
                for item in items[:5]:  # Limit per category
                    section += f"- {item}\n"
                section += "\n"
        
        self.sections.append(section)
        return self
    
    def build_markdown(self) -> str:
        """Build markdown report"""
        return "\n".join(self.sections)
    
    def build_json(self) -> str:
        """Build JSON report"""
        report = {
            'metadata': asdict(self.eda.get_metadata()),
            'quality': asdict(self.eda.get_quality_report()),
            'columns': [asdict(a) for a in self.eda.analyze_columns()],
            'insights': self.eda.generate_insights()
        }
        return json.dumps(report, indent=2, default=str)
