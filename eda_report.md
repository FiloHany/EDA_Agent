
## 📊 Dataset Metadata

- **Shape**: 891 rows × 15 columns
- **Memory Usage**: 0.35 MB
- **Columns**: survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who...


## 🔍 Data Quality Report

- **Missing Values**: 869 cells (6.50%)
- **Duplicate Rows**: 107 (12.01%)

**Columns with Missing Values:**
- age: 177 (19.9%)
- embarked: 2 (0.2%)
- deck: 688 (77.2%)
- embark_town: 2 (0.2%)


## 📈 Column Analysis

### survived (numeric)
- **Unique Values**: 2
- **Missing**: 0 (0.0%)
- **Insights**: High variability (std > mean)

### pclass (numeric)
- **Unique Values**: 3
- **Missing**: 0 (0.0%)

### sex (categorical)
- **Unique Values**: 2
- **Missing**: 0 (0.0%)
- **Insights**: Very low cardinality (2 unique values); Dominated by 'male' (64.8%)

### age (numeric)
- **Unique Values**: 88
- **Missing**: 177 (19.9%)
- **Insights**: Contains 11 outliers (1.2%)

### sibsp (numeric)
- **Unique Values**: 7
- **Missing**: 0 (0.0%)
- **Insights**: Highly right-skewed distribution (skew=3.70); Contains 46 outliers (5.2%); High variability (std > mean)

### parch (numeric)
- **Unique Values**: 7
- **Missing**: 0 (0.0%)
- **Insights**: Highly right-skewed distribution (skew=2.75); Contains 213 outliers (23.9%); High variability (std > mean)

### fare (numeric)
- **Unique Values**: 248
- **Missing**: 0 (0.0%)
- **Insights**: Highly right-skewed distribution (skew=4.79); Contains 116 outliers (13.0%); High variability (std > mean)

### embarked (categorical)
- **Unique Values**: 3
- **Missing**: 2 (0.2%)
- **Insights**: Very low cardinality (3 unique values); Dominated by 'S' (72.3%)

### class (categorical)
- **Unique Values**: 3
- **Missing**: 0 (0.0%)
- **Insights**: Very low cardinality (3 unique values); Dominated by 'Third' (55.1%)

### who (categorical)
- **Unique Values**: 3
- **Missing**: 0 (0.0%)
- **Insights**: Very low cardinality (3 unique values); Dominated by 'man' (60.3%)



## 💡 Automated Insights

**Data Quality:**
- Dataset has 6.5% missing values
- Found 107 duplicate rows (12.0%)

**Distributions:**
- survived: High variability (std > mean)
- sex: Very low cardinality (2 unique values)
- sex: Dominated by 'male' (64.8%)
- age: Contains 11 outliers (1.2%)
- sibsp: Highly right-skewed distribution (skew=3.70)

**Recommendations:**
- Consider dropping columns with >30% missing: deck
- Apply log/box-cox transformation to skewed features: sibsp, parch, fare

