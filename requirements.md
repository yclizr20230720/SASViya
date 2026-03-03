# Requirements Document: ML-Stats Migration

## Introduction

This document specifies requirements for migrating traditional statistical Python code to ML-powered analysis for semiconductor wafer manufacturing engineering data analysis. The system will enhance root cause identification capability for factors affecting Yield, CP (Chip Probing), WAT (Wafer Acceptance Test), and Metrology results by replacing traditional statistical methods (Pearson correlation, ANOVA, Chi-Square) with modern ML approaches (SHAP, XGBoost, Mutual Information, LASSO).

## Glossary

- **ML_System**: The new machine learning-powered analysis system
- **Traditional_System**: The existing statistical analysis system (22 Python files)
- **Input_Data**: CSV files containing wafer manufacturing data (dtgb__detail.csv, dtgb__row.csv, dtrw*_0_all.csv, x_all_*.csv, x_maplabelvar_*.csv)
- **Output_Data**: Analysis results in CSV and JSON formats
- **Factor**: A manufacturing variable that may affect yield (e.g., EQPID, CHAMBERID, RECIPENAME)
- **Ranking_Metric**: A statistical or ML-based score used to rank factors by importance
- **SHAP**: SHapley Additive exPlanations - ML interpretability method
- **Mutual_Information**: ML metric measuring dependency between variables
- **Ensemble_Voting**: Combining results from multiple ML algorithms
- **Backward_Compatible**: Output format matches original system format

## Requirements

### Requirement 1: Input Data Compatibility

**User Story:** As a manufacturing engineer, I want the ML system to process the same input data files as the traditional system, so that I can migrate without changing upstream data pipelines.

#### Acceptance Criteria

1. WHEN Input_Data files are provided from E:\DAP\2026\POC\StatsToML_code&data\Sample_Data\RTM_input_data_from_veda2, THE ML_System SHALL read and parse all file types (dtgb__detail.csv, dtgb__row.csv, dtrw1_0_all.csv through dtrw9_0_all.csv, x_all_*.csv, x_maplabelvar_*.csv)
2. WHEN Input_Data contains missing values or edge cases, THE ML_System SHALL handle them using the same logic as Traditional_System
3. WHEN Input_Data is at LOT level or WAFER level, THE ML_System SHALL process both granularities correctly
4. WHEN Input_Data contains multi-factor combinations, THE ML_System SHALL support analysis across all factor combinations
5. THE ML_System SHALL validate Input_Data schema before processing and report clear errors for invalid formats

### Requirement 2: ML Model Implementation

**User Story:** As a data scientist, I want the system to implement modern ML algorithms that replace traditional statistical methods, so that I can leverage advanced techniques for better root cause identification.

#### Acceptance Criteria

1. WHEN calculating factor importance for continuous data, THE ML_System SHALL use SHAP values and Mutual Information instead of Pearson correlation (R)
2. WHEN calculating weighted correlation (RW), THE ML_System SHALL use LASSO or Elastic Net regularization
3. WHEN calculating significance metrics (Sigma), THE ML_System SHALL use Permutation Importance
4. WHEN calculating statistical significance (P-Value), THE ML_System SHALL use Boruta or RFECV feature selection
5. WHEN calculating directional association (P+/P-), THE ML_System SHALL use XGBoost Gain and SHAP values
6. WHEN analyzing discrete data (Chi-Square), THE ML_System SHALL use Mutual Information
7. WHEN analyzing variance (F-Score), THE ML_System SHALL use XGBoost or SHAP-based importance
8. THE ML_System SHALL implement ensemble voting across SHAP, XGBoost, Permutation Importance, Mutual Information, and LASSO algorithms

### Requirement 3: Output Format Compatibility

**User Story:** As a manufacturing engineer, I want the ML system to generate output files in the same CSV format as the traditional system, so that existing downstream tools and reports continue to work without modification.

#### Acceptance Criteria

1. WHEN ML_System completes analysis, THE ML_System SHALL write Output_Data CSV files to the same directory as Input_Data
2. WHEN generating CSV output, THE ML_System SHALL maintain column names, data types, and structure identical to Traditional_System output
3. WHEN generating ranking tables, THE ML_System SHALL include all columns present in Traditional_System output (Factor name, Ranking_Metric values, sample sizes, effect sizes)
4. THE ML_System SHALL preserve row ordering conventions from Traditional_System (sorted by primary ranking metric)
5. THE ML_System SHALL use the same numeric precision and formatting as Traditional_System for Backward_Compatible output

### Requirement 4: Enhanced JSON Output

**User Story:** As a web application developer, I want the system to generate comprehensive JSON output with rich ML insights, so that I can build modern dashboards and reporting interfaces.

#### Acceptance Criteria

1. WHEN ML_System completes analysis, THE ML_System SHALL generate JSON Output_Data files alongside CSV files
2. WHEN generating JSON output, THE ML_System SHALL include all information from CSV output plus additional ML-specific metrics
3. WHEN generating JSON output, THE ML_System SHALL include model performance metrics (accuracy, R-squared, cross-validation scores)
4. WHEN generating JSON output, THE ML_System SHALL include feature importance rankings from each individual ML algorithm (SHAP, XGBoost, Permutation Importance, Mutual Information, LASSO)
5. WHEN generating JSON output, THE ML_System SHALL include ensemble voting results with confidence scores
6. WHEN generating JSON output, THE ML_System SHALL include metadata (timestamp, input file paths, model hyperparameters, processing time)
7. THE ML_System SHALL structure JSON output with clear hierarchical organization suitable for web application consumption

### Requirement 5: Code Modularization and Naming

**User Story:** As a developer, I want the new ML code to be well-modularized with clear naming conventions, so that I can maintain and extend the system easily.

#### Acceptance Criteria

1. WHEN creating new Python files, THE ML_System SHALL use naming convention Original_name + _ML suffix (e.g., CommonTool_ML.py, SortRanking_ML.py)
2. THE ML_System SHALL organize code into modular components with clear separation of concerns (data loading, preprocessing, ML model training, ranking calculation, output generation)
3. THE ML_System SHALL implement each major statistical method replacement as a separate module (SHAP_module, XGBoost_module, MutualInformation_module, LASSO_module, PermutationImportance_module)
4. THE ML_System SHALL provide a main orchestration module that coordinates the data processing pipeline
5. THE ML_System SHALL include configuration files for ML model hyperparameters separate from code logic

### Requirement 6: Statistical Method Migration Mapping

**User Story:** As a manufacturing engineer, I want clear traceability between traditional statistical methods and their ML replacements, so that I can understand and validate the migration.

#### Acceptance Criteria

1. WHEN replacing Pearson correlation (R) from Index_PearsonCorr.py, THE ML_System SHALL implement SHAP + Mutual Information in corresponding _ML module
2. WHEN replacing weighted correlation (RW) from FCN_RW.py, THE ML_System SHALL implement LASSO or Elastic Net in corresponding _ML module
3. WHEN replacing P-Value from StatIndexCal.py, THE ML_System SHALL implement Boruta or RFECV in corresponding _ML module
4. WHEN replacing P+/P- from SummaryIndex.py, THE ML_System SHALL implement XGBoost Gain + SHAP in corresponding _ML module
5. WHEN replacing Chi-Square from CommonTool.py, THE ML_System SHALL implement Mutual Information in corresponding _ML module
6. WHEN replacing F-Score from StatIndexCal.py, THE ML_System SHALL implement XGBoost or SHAP in corresponding _ML module
7. WHEN replacing Spearman correlation from Index_SpearmanCorr.py, THE ML_System SHALL implement rank-based Mutual Information in corresponding _ML module
8. THE ML_System SHALL document the mapping between each traditional method and its ML replacement in code comments and documentation

### Requirement 7: Model Performance and Validation

**User Story:** As a data scientist, I want robust metrics demonstrating ML model capability, so that I can validate the migration improves upon traditional methods.

#### Acceptance Criteria

1. WHEN ML_System trains models, THE ML_System SHALL perform cross-validation with minimum 5 folds
2. WHEN ML_System completes training, THE ML_System SHALL calculate and report model performance metrics (R-squared for regression, accuracy/F1 for classification)
3. WHEN ML_System generates rankings, THE ML_System SHALL calculate confidence intervals or uncertainty estimates for each Ranking_Metric
4. WHEN ML_System uses ensemble voting, THE ML_System SHALL report agreement scores across individual algorithms
5. THE ML_System SHALL compare ML-based rankings against Traditional_System rankings and report correlation metrics
6. THE ML_System SHALL detect and flag cases where ML model performance is below acceptable thresholds (R-squared < 0.3 for regression)
7. THE ML_System SHALL log all model training parameters, performance metrics, and validation results

### Requirement 8: Root Cause Factor Analysis

**User Story:** As a manufacturing engineer, I want the system to analyze all relevant manufacturing factors, so that I can identify root causes affecting yield and quality metrics.

#### Acceptance Criteria

1. WHEN analyzing manufacturing data, THE ML_System SHALL process equipment factors (EQPID, CHAMBERID, CHAMBER_GROUP)
2. WHEN analyzing manufacturing data, THE ML_System SHALL process process factors (Process Stage, Operation Number, RECIPENAME, EQPRECIPEID)
3. WHEN analyzing manufacturing data, THE ML_System SHALL process test factors (Test Program, Tester, Probe Card)
4. WHEN analyzing manufacturing data, THE ML_System SHALL process manufacturing parameters (MANUF_NO, CUST_CODE, SUPPLIER, RESIST_VALUE)
5. WHEN analyzing manufacturing data, THE ML_System SHALL process metrology-induced factors
6. THE ML_System SHALL support analysis of individual factors and multi-factor interactions
7. THE ML_System SHALL rank all factors by their impact on target variables (Yield, CP, WAT, Metrology results)

### Requirement 9: Data Processing Pipeline

**User Story:** As a developer, I want a well-defined data processing pipeline, so that the system processes data consistently and reliably.

#### Acceptance Criteria

1. THE ML_System SHALL implement data processing stages in sequence: Input validation → Data preparation → Feature engineering → Model training → Ranking calculation → Output generation
2. WHEN processing Input_Data, THE ML_System SHALL perform data cleaning (handle missing values, remove duplicates, validate ranges)
3. WHEN processing Input_Data, THE ML_System SHALL perform feature engineering (encode categorical variables, normalize continuous variables, create interaction terms)
4. WHEN processing Input_Data, THE ML_System SHALL split data into training and validation sets
5. WHEN processing Input_Data, THE ML_System SHALL detect and handle outliers using robust methods
6. WHEN processing Input_Data, THE ML_System SHALL perform time-based filtering when temporal constraints are specified
7. THE ML_System SHALL log each pipeline stage completion with timing and data quality metrics

### Requirement 10: Error Handling and Robustness

**User Story:** As a manufacturing engineer, I want the system to handle errors gracefully and provide clear diagnostics, so that I can troubleshoot issues quickly.

#### Acceptance Criteria

1. WHEN Input_Data files are missing or corrupted, THE ML_System SHALL report specific error messages identifying the problem file and expected format
2. WHEN ML model training fails, THE ML_System SHALL report the failure reason and suggest corrective actions
3. WHEN insufficient data is available for reliable analysis, THE ML_System SHALL warn the user and report minimum sample size requirements
4. WHEN ML_System encounters unexpected data patterns, THE ML_System SHALL log warnings without stopping execution
5. IF ML model performance is below acceptable thresholds, THEN THE ML_System SHALL flag the results as low confidence
6. THE ML_System SHALL implement comprehensive logging at DEBUG, INFO, WARNING, and ERROR levels
7. THE ML_System SHALL create a summary report of all errors and warnings encountered during execution

### Requirement 11: Performance and Scalability

**User Story:** As a manufacturing engineer, I want the system to process large datasets efficiently, so that I can analyze production data in reasonable time.

#### Acceptance Criteria

1. WHEN processing datasets with up to 100,000 rows, THE ML_System SHALL complete analysis within 30 minutes on standard hardware
2. WHEN processing multiple factors (>50 factors), THE ML_System SHALL use efficient algorithms to avoid combinatorial explosion
3. THE ML_System SHALL implement parallel processing for independent ML model training tasks
4. THE ML_System SHALL use memory-efficient data structures to handle large Input_Data files
5. WHEN processing very large datasets, THE ML_System SHALL support batch processing or sampling strategies
6. THE ML_System SHALL report processing time for each major pipeline stage

### Requirement 12: Dependency Management

**User Story:** As a developer, I want clear dependency specifications, so that I can set up the development and production environments correctly.

#### Acceptance Criteria

1. THE ML_System SHALL document all required Python packages (pandas, numpy, scipy, sklearn, xgboost, shap, statsmodels, pandasql) with version constraints
2. THE ML_System SHALL use scikit-learn for LASSO, Elastic Net, Permutation Importance, and Mutual Information
3. THE ML_System SHALL use xgboost library for XGBoost-based importance metrics
4. THE ML_System SHALL use shap library for SHAP value calculations
5. THE ML_System SHALL use Boruta-py or similar for Boruta feature selection
6. THE ML_System SHALL provide a requirements.txt or environment.yml file for reproducible environment setup
7. THE ML_System SHALL specify minimum Python version (3.8 or higher recommended)
