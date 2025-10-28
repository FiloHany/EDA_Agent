# 🤖 Intelligent EDA Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Qwen3:4b-purple.svg)

**AI-Powered Exploratory Data Analysis Made Effortless**

[🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🎯 Features](#-features) • [🔧 Installation](#-installation)

</div>

---

## 📋 Table of Contents

- [🤖 Intelligent EDA Agent](#-intelligent-eda-agent)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Overview](#-overview)
  - [🎯 Key Features](#-key-features)
  - [🏗️ Architecture](#️-architecture)
  - [🔧 Installation](#-installation)
  - [🚀 Quick Start](#-quick-start)
  - [💻 Usage](#-usage)
  - [📚 API Reference](#-api-reference)
  - [🎨 Examples](#-examples)
  - [📊 Supported Data Types](#-supported-data-types)
  - [🔍 Analysis Capabilities](#-analysis-capabilities)
  - [📈 Report Generation](#-report-generation)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Overview

The **Intelligent EDA Agent** is a cutting-edge AI-powered platform that revolutionizes Exploratory Data Analysis (EDA) for CSV datasets. Built with modern machine learning frameworks and designed for data scientists, analysts, and business users, it transforms complex data analysis into natural language conversations.

**What makes it special:**
- 🤖 **Conversational AI**: Ask questions in plain English, get instant insights
- 🎨 **Multiple Interfaces**: CLI, beautiful Streamlit web app, and Jupyter notebook integration
- 🧠 **Smart Analysis**: Specialized analyzers for numeric, categorical, and datetime data
- 📊 **Rich Visualizations**: Interactive Plotly charts with one-click generation
- 📋 **Automated Reports**: Comprehensive Markdown reports with actionable insights
- 🚀 **Production Ready**: Supports both OpenAI GPT models and local Ollama inference

---

## 🎯 Key Features

### 🤖 AI-Powered Analysis
- **Natural Language Queries**: "What's the correlation between age and income?" or "Show me outliers in sales data"
- **Contextual Responses**: AI understands data context and provides actionable insights
- **Multi-step Analysis**: Complex workflows handled automatically
- **Intelligent Recommendations**: Suggests next steps and potential issues

### 🎨 Multiple User Interfaces
- **Streamlit Web App**: Modern, responsive interface with drag-and-drop uploads
- **Command Line Interface**: Perfect for automation and scripting
- **Jupyter Integration**: Seamless notebook experience with interactive widgets

### 📊 Advanced Analytics
- **Automated Data Profiling**: Comprehensive dataset overview in seconds
- **Quality Assessment**: Missing values, duplicates, data type validation
- **Statistical Analysis**: Distribution analysis, correlation studies, outlier detection
- **Pattern Recognition**: Trend identification, seasonality detection, anomaly detection

### 📈 Rich Visualizations
- **Interactive Charts**: Plotly-powered visualizations with hover details
- **One-Click Generation**: Create histograms, scatter plots, box plots, and heatmaps instantly
- **Export Capabilities**: Save as HTML, PNG, or embed in reports
- **Customizable Styling**: Dark theme optimized for data analysis

### 📋 Report Generation
- **Automated Reports**: Complete EDA reports generated without manual effort
- **Structured Format**: Well-organized sections with insights and recommendations
- **Export Options**: Markdown, JSON, and HTML formats
- **Customizable Templates**: Adapt reports for different audiences

---

## 🏗️ Architecture

The system follows a **layered, hexagonal architecture** designed for scalability and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACES                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Streamlit │  │ Command Line│  │   Jupyter   │          │
│  │   Web App   │  │  Interface  │  │ Integration │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   EDA Agent │  │   EDA Cmd   │  │ Report Gen. │          │
│  │ (LangGraph) │  │    Line     │  │   Builder   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    DOMAIN LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ EDA Service │  │  EDA Tools  │  │ Analyzer    │          │
│  │             │  │ (LangChain) │  │  Factory    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Numeric    │  │ Categorical │  │  Datetime   │          │
│  │  Analyzer   │  │  Analyzer   │  │  Analyzer   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **EDA Agent**: LangGraph-powered conversational AI orchestrator
- **EDA Service**: Central service coordinating all analysis operations
- **Analyzer Factory**: Intelligent analyzer selection based on data types
- **Specialized Analyzers**: Domain-specific analysis for different data types

---

## 🔧 Installation

### Prerequisites

- **Python 3.8+**
- **For OpenAI users**: OpenAI API key
- **For local inference**: [Ollama](https://ollama.ai/) with Qwen3:4b model

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/intelligent-eda-agent.git
cd intelligent-eda-agent

# Install dependencies
pip install -r requirements.txt

# For Ollama users (recommended for privacy)
ollama pull qwen3:4b
```

### Optional Dependencies

```bash
# For enhanced visualizations
pip install matplotlib seaborn plotly

# For development
pip install pytest black mypy
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

```python
import pandas as pd
from core.EDA.eda_agent import EDAAgent

# Load your dataset
df = pd.read_csv('your_data.csv')

# Initialize the agent
agent = EDAAgent(df, use_openai=False)  # Set True for OpenAI
```

### 2. Start Analyzing

```python
# Get comprehensive overview
overview = agent.chat("Give me a dataset overview")
print(overview)

# Check data quality
quality = agent.chat("Check for data quality issues")
print(quality)

# Generate insights
insights = agent.chat("What are the key insights?")
print(insights)
```

### 3. Launch Web Interface

```bash
streamlit run src/streamlit_app.py
```

### 4. Generate Report

```python
# Generate comprehensive EDA report
report = agent.generate_automatic_eda(save_path="eda_report.md")
```

---

## 💻 Usage

### Command Line Interface

```bash
# Interactive mode
python src/main.py

# Direct query
python src/main.py "What are the correlations in my data?"
```

### Streamlit Web App

```bash
streamlit run src/streamlit_app.py
```

**Features:**
- Drag-and-drop CSV upload
- Real-time chat interface
- Interactive visualizations
- One-click report generation
- Export capabilities

### Jupyter Notebook

```python
from core.EDA.eda_agent import EDAAgent
import pandas as pd

df = pd.read_csv('data.csv')
agent = EDAAgent(df)

# Use in notebook cells
response = agent.chat("Analyze the age column")
print(response)
```

---

## 📚 API Reference

### EDAAgent Class

#### Initialization
```python
EDAAgent(df: pd.DataFrame, use_openai: bool = False)
```

**Parameters:**
- `df`: Pandas DataFrame to analyze
- `use_openai`: Use OpenAI GPT models (default: False, uses Ollama)

#### Methods

##### `chat(query: str, verbose: bool = False) -> str`
Send a natural language query to the agent.

**Example:**
```python
response = agent.chat("What's the average age by category?")
```

##### `generate_automatic_eda(save_path: Optional[str] = None) -> str`
Generate comprehensive EDA report.

**Example:**
```python
report = agent.generate_automatic_eda("report.md")
```

### EDAService Class

#### Methods

##### `get_dataset_overview() -> dict`
Returns comprehensive dataset statistics.

##### `get_data_quality() -> dict`
Analyzes data quality issues.

##### `analyze_column(column_name: str) -> dict`
Deep analysis of specific column.

##### `get_correlations() -> pd.DataFrame`
Computes correlation matrix for numeric columns.

---

## 🎨 Examples

### Basic Analysis

```python
from core.EDA.eda_agent import EDAAgent
import pandas as pd

# Load Titanic dataset
df = pd.read_csv('src/titanic.csv')
agent = EDAAgent(df)

# Dataset overview
print(agent.chat("Give me an overview of the dataset"))

# Quality check
print(agent.chat("Check data quality"))

# Specific analysis
print(agent.chat("Analyze the Age column"))

# Correlations
print(agent.chat("Show me correlations between numeric features"))
```

### Advanced Queries

```python
# Complex analysis
agent.chat("What's the survival rate by passenger class and gender?")

# Statistical insights
agent.chat("Find outliers in the Fare column")

# Pattern recognition
agent.chat("Are there any seasonal patterns in the data?")

# Custom analysis
agent.chat("Compare the distribution of Age for survivors vs non-survivors")
```

### Visualization Generation

```python
import plotly.express as px

# Create visualization based on agent insights
fig = px.histogram(df, x='Age', color='Survived',
                  title='Age Distribution by Survival')
fig.show()
```

---

## 📊 Supported Data Types

| Data Type | Analysis Features | Visualizations |
|-----------|------------------|----------------|
| **Numeric** | Mean, median, std, quartiles, skewness, kurtosis, outliers | Histograms, box plots, scatter plots |
| **Categorical** | Frequency distribution, mode, cardinality | Bar charts, pie charts |
| **Datetime** | Trends, seasonality, time patterns | Time series, seasonal decomposition |
| **Text** | Word frequency, sentiment (future) | Word clouds, text statistics |
| **Boolean** | True/false ratios, patterns | Pie charts, stacked bars |

---

## 🔍 Analysis Capabilities

### Automated Analysis Pipeline

1. **Data Loading & Validation**
   - CSV parsing with error handling
   - Data type inference
   - Memory optimization

2. **Quality Assessment**
   - Missing value detection
   - Duplicate identification
   - Data type validation
   - Outlier detection

3. **Statistical Analysis**
   - Descriptive statistics
   - Distribution analysis
   - Correlation studies
   - Hypothesis testing

4. **Pattern Recognition**
   - Trend identification
   - Seasonal patterns
   - Anomaly detection
   - Feature relationships

### Specialized Analyzers

#### Numeric Analyzer
- Central tendency measures
- Dispersion statistics
- Distribution shape analysis
- Outlier detection (IQR, Z-score)
- Normality tests

#### Categorical Analyzer
- Frequency distributions
- Mode analysis
- Cardinality assessment
- Category relationships
- Chi-square tests

#### Datetime Analyzer
- Temporal patterns
- Seasonality detection
- Trend analysis
- Time series decomposition
- Calendar effects

---

## 📈 Report Generation

### Automated Report Structure

```
# Dataset Overview
├── Basic Statistics
├── Data Types
└── Memory Usage

# Data Quality Report
├── Missing Values Analysis
├── Duplicate Detection
├── Outlier Summary
└── Data Type Issues

# Column Analysis
├── [Column 1] Analysis
├── [Column 2] Analysis
└── ...

# Insights & Recommendations
├── Key Findings
├── Actionable Insights
└── Next Steps
```

### Report Formats

- **Markdown**: Human-readable with formatting
- **JSON**: Structured data for programmatic use
- **HTML**: Web-ready interactive reports

### Customization Options

```python
from core.report.report_generator import ReportBuilder

builder = ReportBuilder(eda_service)
report = (builder
    .add_metadata_section()
    .add_quality_section()
    .add_column_analysis_section()
    .add_insights_section()
    .build_markdown())
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/intelligent-eda-agent.git
cd intelligent-eda-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black src/
```

### Areas for Contribution

- **New Analyzers**: Support for additional data types
- **Enhanced Visualizations**: More chart types and interactivity
- **Performance Optimization**: Faster analysis for large datasets
- **Additional LLM Support**: Integration with other AI models
- **Plugin Architecture**: Extensible analyzer system

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** & **LangGraph** for the powerful AI orchestration framework
- **Streamlit** for the amazing web app framework
- **Plotly** for interactive visualizations
- **Pandas** for robust data manipulation
- **Ollama** for accessible local AI inference
- **OpenAI** for GPT model access

### Special Thanks

- The data science community for inspiration and feedback
- Contributors who help improve the project
- Users who provide valuable use cases and bug reports

---

<div align="center">

**Made with ❤️ for the data science community**

[⭐ Star us on GitHub](https://github.com/yourusername/intelligent-eda-agent) • [🐛 Report Issues](https://github.com/yourusername/intelligent-eda-agent/issues) • [💬 Join Discussions](https://github.com/yourusername/intelligent-eda-agent/discussions)

</div>
