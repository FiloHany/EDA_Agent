from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple


class ColumnType(Enum):
    """Column type enumeration"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class DatasetMetadata:
    """Dataset metadata container"""
    shape: Tuple[int, int]
    memory_usage_mb: float
    column_count: int
    row_count: int
    columns: List[str]
    dtypes: Dict[str, str]


@dataclass
class DataQualityReport:
    """Data quality metrics container"""
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    duplicate_rows: int
    duplicate_percentage: float
    total_cells: int
    missing_cells: int


@dataclass
class ColumnAnalysis:
    """Individual column analysis result"""
    name: str
    column_type: ColumnType
    unique_count: int
    missing_count: int
    missing_percentage: float
    statistics: Dict[str, Any]
    insights: List[str]
