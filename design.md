# Design Document: ML-Stats Migration

## Overview

This design specifies the architecture for migrating traditional statistical analysis code to ML-powered analysis for semiconductor wafer manufacturing data. The system replaces 22 Python files implementing traditional statistical methods (Pearson correlation, ANOVA, Chi-Square) with modern ML approaches (SHAP, XGBoost, Mutual Information, LASSO, Permutation Importance).

### Design Goals

1. **Backward Compatibility**: Maintain identical CSV output format for existing downstream tools
2. **Enhanced Insights**: Provide rich JSON output with ML-specific metrics for modern dashboards
3. **Modular Architecture**: Clean separation between data processing, ML models, and output generation
4. **Ensemble Approach**: Combine multiple ML algorithms for robust factor ranking
5. **Traceability**: Clear mapping between traditional statistical methods and ML replacements

### Key Design Decisions

- **Ensemble Voting Strategy**: Use weighted voting across 5 ML algorithms (SHAP, XGBoost, Permutation Importance, Mutual Information, LASSO) to produce robust rankings
- **Pipeline Architecture**: Implement modular pipeline with clear stages (validation → preprocessing → feature engineering → model training → ranking → output)
- **Dual Output Format**: Generate both CSV (backward compatible) and JSON (enhanced) outputs
- **Configuration-Driven**: Externalize ML hyperparameters and pipeline settings to configuration files
- **Incremental Migration**: Each traditional Python file maps to a corresponding _ML module for clear traceability

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ML Stats System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Input      │      │   Pipeline   │      │   Output     │  │
│  │   Layer      │─────▶│   Layer      │─────▶│   Layer      │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │                      │                      │          │
│    ┌────▼────┐          ┌─────▼─────┐          ┌────▼────┐    │
│    │ Data    │          │ ML Model  │          │ CSV     │    │
│    │ Loader  │          │ Ensemble  │          │ Writer  │    │
│    └─────────┘          └───────────┘          └─────────┘    │
│                               │                       │          │
│                         ┌─────▼─────┐          ┌────▼────┐    │
│                         │ SHAP      │          │ JSON    │    │
│                         │ XGBoost   │          │ Writer  │    │
│                         │ Perm Imp  │          └─────────┘    │
│                         │ Mutual I  │                          │
│                         │ LASSO     │                          │
│                         └───────────┘                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Input Validation Stage**: Validate file existence, schema, data types
2. **Data Preprocessing Stage**: Handle missing values, outliers, data cleaning
3. **Feature Engineering Stage**: Encode categoricals, normalize, create interactions
4. **Model Training Stage**: Train 5 ML models in parallel
5. **Ranking Calculation Stage**: Compute ensemble rankings via weighted voting
6. **Output Generation Stage**: Write CSV and JSON outputs

### Module Organization

```
ml_stats_migration/
├── config/
│   ├── pipeline_config.yaml       # Pipeline settings
│   └── model_config.yaml          # ML hyperparameters
├── core/
│   ├── pipeline_orchestrator.py   # Main pipeline coordinator
│   ├── data_loader.py             # Input data loading
│   ├── data_validator.py          # Schema and data validation
│   └── logger.py                  # Logging utilities
├── preprocessing/
│   ├── data_cleaner.py            # Missing values, outliers
│   ├── feature_engineer.py        # Feature transformations
│   └── data_splitter.py           # Train/validation split
├── models/
│   ├── base_model.py              # Abstract base class
│   ├── shap_model.py              # SHAP-based importance
│   ├── xgboost_model.py           # XGBoost gain/importance
│   ├── permutation_model.py       # Permutation importance
│   ├── mutual_info_model.py       # Mutual information
│   ├── lasso_model.py             # LASSO/Elastic Net
│   └── ensemble_voter.py          # Ensemble voting logic
├── ranking/
│   ├── ranking_calculator.py      # Compute final rankings
│   └── confidence_estimator.py    # Uncertainty quantification
├── output/
│   ├── csv_writer.py              # Backward-compatible CSV
│   └── json_writer.py             # Enhanced JSON output
└── legacy_mapping/
    ├── CommonTool_ML.py           # Replaces CommonTool.py
    ├── SortRanking_ML.py          # Replaces SortRanking.py
    ├── StatIndexCal_ML.py         # Replaces StatIndexCal.py
    ├── SummaryIndex_ML.py         # Replaces SummaryIndex.py
    ├── HitCountCal_ML.py          # Replaces HitCountCal.py
    ├── FCN_RW_ML.py               # Replaces FCN_RW.py
    ├── Index_PearsonCorr_ML.py    # Replaces Index_PearsonCorr.py
    └── [... 15 more _ML modules]
```

## Components and Interfaces

### Core Components

#### PipelineOrchestrator

**Responsibility**: Coordinate execution of all pipeline stages

**Interface**:
```python
class PipelineOrchestrator:
    def __init__(self, config_path: str)
    def run(self, input_dir: str, output_dir: str) -> PipelineResult
    def validate_inputs(self, input_dir: str) -> ValidationResult
    def execute_stage(self, stage: PipelineStage) -> StageResult
    def handle_error(self, error: Exception, stage: PipelineStage) -> None
```

**Key Methods**:
- `run()`: Execute complete pipeline from input to output
- `validate_inputs()`: Check input files exist and are valid
- `execute_stage()`: Run individual pipeline stage with error handling
- `handle_error()`: Log errors and determine if pipeline should continue

#### DataLoader

**Responsibility**: Load and parse input CSV files

**Interface**:
```python
class DataLoader:
    def load_dtgb_detail(self, path: str) -> pd.DataFrame
    def load_dtgb_row(self, path: str) -> pd.DataFrame
    def load_dtrw_files(self, dir_path: str) -> Dict[str, pd.DataFrame]
    def load_x_all_files(self, dir_path: str) -> Dict[str, pd.DataFrame]
    def load_x_maplabelvar_files(self, dir_path: str) -> Dict[str, pd.DataFrame]
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame
```

**Key Methods**:
- `load_*()`: Load specific file types with appropriate parsing
- `merge_datasets()`: Combine multiple datasets into unified DataFrame

#### DataValidator

**Responsibility**: Validate input data schema and quality

**Interface**:
```python
class DataValidator:
    def validate_schema(self, df: pd.DataFrame, expected_schema: Schema) -> ValidationResult
    def validate_data_types(self, df: pd.DataFrame) -> ValidationResult
    def validate_ranges(self, df: pd.DataFrame, range_config: Dict) -> ValidationResult
    def check_missing_values(self, df: pd.DataFrame) -> MissingValueReport
    def check_duplicates(self, df: pd.DataFrame) -> DuplicateReport
```

**Key Methods**:
- `validate_schema()`: Check required columns exist
- `validate_data_types()`: Verify column types match expectations
- `validate_ranges()`: Check numeric values within valid ranges
- `check_missing_values()`: Report missing data statistics
- `check_duplicates()`: Identify duplicate rows

### Preprocessing Components

#### DataCleaner

**Responsibility**: Handle missing values, outliers, data quality issues

**Interface**:
```python
class DataCleaner:
    def handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame
    def detect_outliers(self, df: pd.DataFrame, method: str) -> pd.Series
    def remove_outliers(self, df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame
    def cap_outliers(self, df: pd.DataFrame, lower_pct: float, upper_pct: float) -> pd.DataFrame
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame
```

**Key Methods**:
- `handle_missing_values()`: Impute or remove missing data (mean, median, forward-fill)
- `detect_outliers()`: Identify outliers using IQR or Z-score methods
- `remove_outliers()`: Remove rows with outliers
- `cap_outliers()`: Cap outliers at percentile thresholds

#### FeatureEngineer

**Responsibility**: Transform features for ML model consumption

**Interface**:
```python
class FeatureEngineer:
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame
    def normalize_continuous(self, df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame
    def create_interactions(self, df: pd.DataFrame, factor_pairs: List[Tuple[str, str]]) -> pd.DataFrame
    def extract_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame
    def bin_continuous_variables(self, df: pd.DataFrame, bin_config: Dict) -> pd.DataFrame
```

**Key Methods**:
- `encode_categorical()`: One-hot or label encoding for categorical variables
- `normalize_continuous()`: StandardScaler or MinMaxScaler normalization
- `create_interactions()`: Generate interaction terms between factor pairs
- `extract_temporal_features()`: Extract day-of-week, month, etc. from dates

### ML Model Components

#### BaseModel (Abstract)

**Responsibility**: Define common interface for all ML models

**Interface**:
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    
    @abstractmethod
    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]
```

**Key Methods**:
- `fit()`: Train model on data
- `compute_importance()`: Calculate feature importance scores
- `get_performance_metrics()`: Return model performance (R², accuracy, etc.)
- `cross_validate()`: Perform k-fold cross-validation

#### SHAPModel

**Responsibility**: Compute SHAP-based feature importance

**Interface**:
```python
class SHAPModel(BaseModel):
    def __init__(self, base_estimator: str = 'tree')
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray
    def get_performance_metrics(self) -> Dict[str, float]
```

**Implementation Details**:
- Use `shap.TreeExplainer` for tree-based models or `shap.KernelExplainer` for general models
- Compute mean absolute SHAP values as importance scores
- Support both regression and classification tasks

#### XGBoostModel

**Responsibility**: Compute XGBoost gain-based importance

**Interface**:
```python
class XGBoostModel(BaseModel):
    def __init__(self, objective: str = 'reg:squarederror', n_estimators: int = 100)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def get_gain_importance(self) -> pd.Series
    def get_performance_metrics(self) -> Dict[str, float]
```

**Implementation Details**:
- Use `xgboost.XGBRegressor` or `xgboost.XGBClassifier`
- Extract `gain` importance from trained model
- Hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=100

#### PermutationImportanceModel

**Responsibility**: Compute permutation-based feature importance

**Interface**:
```python
class PermutationImportanceModel(BaseModel):
    def __init__(self, base_estimator: Any, n_repeats: int = 10)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def get_performance_metrics(self) -> Dict[str, float]
```

**Implementation Details**:
- Use `sklearn.inspection.permutation_importance`
- Base estimator: RandomForestRegressor or RandomForestClassifier
- Repeat permutations 10 times and average results

#### MutualInformationModel

**Responsibility**: Compute mutual information scores

**Interface**:
```python
class MutualInformationModel(BaseModel):
    def __init__(self, discrete_features: List[bool] = None)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def get_performance_metrics(self) -> Dict[str, float]
```

**Implementation Details**:
- Use `sklearn.feature_selection.mutual_info_regression` or `mutual_info_classif`
- Handle both continuous and discrete features
- Normalize scores to [0, 1] range

#### LASSOModel

**Responsibility**: Compute LASSO/Elastic Net coefficient-based importance

**Interface**:
```python
class LASSOModel(BaseModel):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 1.0)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def get_coefficients(self) -> pd.Series
    def get_performance_metrics(self) -> Dict[str, float]
```

**Implementation Details**:
- Use `sklearn.linear_model.ElasticNet` (l1_ratio=1.0 for pure LASSO)
- Use absolute coefficient values as importance scores
- Perform cross-validated alpha selection with `ElasticNetCV`

#### EnsembleVoter

**Responsibility**: Combine rankings from multiple ML models

**Interface**:
```python
class EnsembleVoter:
    def __init__(self, weights: Dict[str, float] = None)
    def vote(self, model_rankings: Dict[str, pd.Series]) -> pd.Series
    def compute_agreement_score(self, model_rankings: Dict[str, pd.Series]) -> float
    def compute_confidence_intervals(self, model_rankings: Dict[str, pd.Series]) -> pd.DataFrame
```

**Key Methods**:
- `vote()`: Compute weighted average of normalized rankings
- `compute_agreement_score()`: Measure consensus across models (Kendall's tau)
- `compute_confidence_intervals()`: Calculate uncertainty bounds for rankings

**Voting Algorithm**:
1. Normalize each model's importance scores to [0, 1]
2. Apply model-specific weights (default: equal weights)
3. Compute weighted average: `final_score = Σ(weight_i × normalized_score_i)`
4. Rank factors by final_score in descending order

### Ranking Components

#### RankingCalculator

**Responsibility**: Compute final factor rankings from ensemble results

**Interface**:
```python
class RankingCalculator:
    def calculate_rankings(self, ensemble_scores: pd.Series) -> pd.DataFrame
    def add_legacy_metrics(self, rankings: pd.DataFrame, legacy_data: Dict) -> pd.DataFrame
    def sort_by_metric(self, rankings: pd.DataFrame, metric: str) -> pd.DataFrame
```

**Key Methods**:
- `calculate_rankings()`: Convert ensemble scores to ranked table
- `add_legacy_metrics()`: Include traditional metrics for comparison
- `sort_by_metric()`: Sort rankings by specified column

#### ConfidenceEstimator

**Responsibility**: Quantify uncertainty in rankings

**Interface**:
```python
class ConfidenceEstimator:
    def estimate_confidence(self, model_rankings: Dict[str, pd.Series]) -> pd.Series
    def compute_bootstrap_ci(self, X: pd.DataFrame, y: pd.Series, model: BaseModel, n_bootstrap: int = 100) -> pd.DataFrame
    def flag_low_confidence(self, rankings: pd.DataFrame, threshold: float = 0.5) -> pd.Series
```

**Key Methods**:
- `estimate_confidence()`: Compute confidence based on model agreement
- `compute_bootstrap_ci()`: Bootstrap confidence intervals for importance scores
- `flag_low_confidence()`: Identify factors with uncertain rankings

### Output Components

#### CSVWriter

**Responsibility**: Generate backward-compatible CSV output

**Interface**:
```python
class CSVWriter:
    def write_ranking_table(self, rankings: pd.DataFrame, output_path: str) -> None
    def format_legacy_columns(self, rankings: pd.DataFrame) -> pd.DataFrame
    def apply_legacy_precision(self, rankings: pd.DataFrame) -> pd.DataFrame
```

**Key Methods**:
- `write_ranking_table()`: Write rankings to CSV with legacy format
- `format_legacy_columns()`: Ensure column names match traditional system
- `apply_legacy_precision()`: Apply same numeric formatting as original

**Output Format** (matches Traditional_System):
```
Factor,R,RW,Sigma,PValue,PPlus,PMinus,ChiSquare,FScore,SampleSize,EffectSize
EQPID,0.856,0.823,2.45,0.001,0.92,0.08,45.2,38.7,1250,0.65
CHAMBERID,0.742,0.698,1.89,0.012,0.85,0.15,32.1,28.4,1250,0.52
...
```

#### JSONWriter

**Responsibility**: Generate enhanced JSON output with ML insights

**Interface**:
```python
class JSONWriter:
    def write_enhanced_output(self, rankings: pd.DataFrame, metadata: Dict, output_path: str) -> None
    def build_json_structure(self, rankings: pd.DataFrame, metadata: Dict) -> Dict
    def add_model_metrics(self, json_data: Dict, model_results: Dict) -> Dict
    def add_ensemble_details(self, json_data: Dict, ensemble_data: Dict) -> Dict
```

**Key Methods**:
- `write_enhanced_output()`: Write complete JSON output
- `build_json_structure()`: Create hierarchical JSON structure
- `add_model_metrics()`: Include individual model performance
- `add_ensemble_details()`: Include voting results and confidence scores

**JSON Structure**:
```json
{
  "metadata": {
    "timestamp": "2026-01-15T10:30:00Z",
    "input_files": ["dtgb__detail.csv", "dtrw1_0_all.csv", ...],
    "processing_time_seconds": 245.3,
    "pipeline_version": "1.0.0"
  },
  "model_performance": {
    "shap": {"r_squared": 0.78, "cv_score": 0.75},
    "xgboost": {"r_squared": 0.82, "cv_score": 0.79},
    "permutation": {"r_squared": 0.76, "cv_score": 0.73},
    "mutual_info": {"mi_score": 0.68},
    "lasso": {"r_squared": 0.71, "cv_score": 0.69}
  },
  "ensemble": {
    "agreement_score": 0.85,
    "weights": {"shap": 0.25, "xgboost": 0.25, "permutation": 0.2, "mutual_info": 0.15, "lasso": 0.15}
  },
  "rankings": [
    {
      "factor": "EQPID",
      "rank": 1,
      "ensemble_score": 0.856,
      "confidence": 0.92,
      "individual_scores": {
        "shap": 0.89,
        "xgboost": 0.91,
        "permutation": 0.84,
        "mutual_info": 0.82,
        "lasso": 0.78
      },
      "legacy_metrics": {
        "R": 0.856,
        "RW": 0.823,
        "PValue": 0.001
      },
      "sample_size": 1250,
      "effect_size": 0.65
    },
    ...
  ]
}
```

### Legacy Mapping Modules

Each traditional Python file maps to a corresponding _ML module that implements the ML replacement:

#### CommonTool_ML

**Replaces**: CommonTool.py (statistical library with T-tests, F-tests, Chi-Square)

**ML Replacement**: Mutual Information for Chi-Square, Permutation Importance for T-tests/F-tests

**Interface**:
```python
class CommonToolML:
    def chi_square_ml(self, X: pd.DataFrame, y: pd.Series) -> float
    def t_test_ml(self, X: pd.DataFrame, y: pd.Series) -> float
    def f_test_ml(self, X: pd.DataFrame, y: pd.Series) -> float
```

#### Index_PearsonCorr_ML

**Replaces**: Index_PearsonCorr.py (Pearson correlation coefficient)

**ML Replacement**: SHAP + Mutual Information

**Interface**:
```python
class PearsonCorrML:
    def compute_correlation_ml(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
```

#### FCN_RW_ML

**Replaces**: FCN_RW.py (weighted correlation with outlier detection)

**ML Replacement**: LASSO / Elastic Net

**Interface**:
```python
class FCNRWML:
    def compute_weighted_correlation_ml(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
```

#### StatIndexCal_ML

**Replaces**: StatIndexCal.py (P-Value using ANOVA F-test, Kruskal-Wallis)

**ML Replacement**: Boruta / RFECV for feature selection

**Interface**:
```python
class StatIndexCalML:
    def compute_pvalue_ml(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def compute_fscore_ml(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
```

#### SummaryIndex_ML

**Replaces**: SummaryIndex.py (PPlus calculation combining PHit, PRate, PHotRate, PConti)

**ML Replacement**: XGBoost Gain + SHAP

**Interface**:
```python
class SummaryIndexML:
    def compute_pplus_ml(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
    def compute_pminus_ml(self, X: pd.DataFrame, y: pd.Series) -> pd.Series
```

## Data Models

### Input Data Models

#### DTGBDetail

**Source**: dtgb__detail.csv

**Schema**:
```python
@dataclass
class DTGBDetail:
    lot_id: str
    wafer_id: str
    eqpid: str
    chamberid: str
    chamber_group: str
    process_stage: str
    operation_number: str
    recipename: str
    eqprecipeid: str
    timestamp: datetime
    # ... additional manufacturing parameters
```

#### DTGBRow

**Source**: dtgb__row.csv

**Schema**:
```python
@dataclass
class DTGBRow:
    lot_id: str
    wafer_id: str
    row_number: int
    # ... row-level data
```

#### DTRWData

**Source**: dtrw1_0_all.csv through dtrw9_0_all.csv

**Schema**:
```python
@dataclass
class DTRWData:
    lot_id: str
    wafer_id: str
    parameter_name: str
    parameter_value: float
    # ... parameter-specific data
```

#### XAllData

**Source**: x_all_*.csv

**Schema**:
```python
@dataclass
class XAllData:
    lot_id: str
    wafer_id: str
    factor_name: str
    factor_value: Union[str, float]
    factor_type: str  # 'continuous' or 'discrete'
```

#### XMapLabelVar

**Source**: x_maplabelvar_*.csv

**Schema**:
```python
@dataclass
class XMapLabelVar:
    factor_name: str
    label: str
    variable_type: str
    description: str
```

### Intermediate Data Models

#### ProcessedDataset

**Description**: Unified dataset after loading and merging all input files

**Schema**:
```python
@dataclass
class ProcessedDataset:
    data: pd.DataFrame  # Merged data with all factors and targets
    factor_columns: List[str]  # List of factor column names
    target_columns: List[str]  # List of target column names (Yield, CP, WAT, etc.)
    continuous_factors: List[str]  # Continuous factor names
    discrete_factors: List[str]  # Discrete factor names
    metadata: Dict[str, Any]  # Additional metadata
```

#### FeatureSet

**Description**: Engineered features ready for ML model training

**Schema**:
```python
@dataclass
class FeatureSet:
    X: pd.DataFrame  # Feature matrix
    y: pd.Series  # Target variable
    feature_names: List[str]  # Original feature names
    feature_types: Dict[str, str]  # Feature type mapping
    encoding_info: Dict[str, Any]  # Encoding metadata for categorical features
```

### Output Data Models

#### FactorRanking

**Description**: Single factor ranking result

**Schema**:
```python
@dataclass
class FactorRanking:
    factor_name: str
    rank: int
    ensemble_score: float
    confidence: float
    individual_scores: Dict[str, float]  # Scores from each ML model
    legacy_metrics: Dict[str, float]  # Traditional metrics for comparison
    sample_size: int
    effect_size: float
    p_value: Optional[float]
```

#### RankingTable

**Description**: Complete ranking table for all factors

**Schema**:
```python
@dataclass
class RankingTable:
    rankings: List[FactorRanking]
    target_variable: str  # Which target was analyzed (Yield, CP, WAT, etc.)
    timestamp: datetime
    metadata: Dict[str, Any]
```

#### ModelPerformance

**Description**: Performance metrics for a single ML model

**Schema**:
```python
@dataclass
class ModelPerformance:
    model_name: str
    r_squared: Optional[float]  # For regression tasks
    accuracy: Optional[float]  # For classification tasks
    f1_score: Optional[float]  # For classification tasks
    cv_scores: List[float]  # Cross-validation scores
    training_time_seconds: float
    hyperparameters: Dict[str, Any]
```

#### PipelineResult

**Description**: Complete pipeline execution result

**Schema**:
```python
@dataclass
class PipelineResult:
    success: bool
    ranking_table: Optional[RankingTable]
    model_performances: Dict[str, ModelPerformance]
    ensemble_agreement_score: float
    processing_time_seconds: float
    errors: List[str]
    warnings: List[str]
    output_files: Dict[str, str]  # File type -> file path mapping
```

### Configuration Models

#### PipelineConfig

**Description**: Pipeline execution configuration

**Schema**:
```python
@dataclass
class PipelineConfig:
    input_directory: str
    output_directory: str
    target_variable: str  # 'Yield', 'CP', 'WAT', 'Metrology'
    analysis_level: str  # 'LOT' or 'WAFER'
    enable_parallel: bool
    n_jobs: int
    log_level: str
    enable_json_output: bool
    enable_csv_output: bool
```

#### ModelConfig

**Description**: ML model hyperparameters

**Schema**:
```python
@dataclass
class ModelConfig:
    shap_config: Dict[str, Any]  # {'base_estimator': 'tree', 'n_samples': 100}
    xgboost_config: Dict[str, Any]  # {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    permutation_config: Dict[str, Any]  # {'n_repeats': 10, 'base_estimator': 'random_forest'}
    mutual_info_config: Dict[str, Any]  # {'n_neighbors': 3}
    lasso_config: Dict[str, Any]  # {'alpha': 1.0, 'l1_ratio': 1.0, 'cv_folds': 5}
    ensemble_weights: Dict[str, float]  # {'shap': 0.25, 'xgboost': 0.25, ...}
    cross_validation_folds: int
    random_state: int
```

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Input File Type Compatibility

*For any* valid input directory containing the expected CSV file types (dtgb__detail.csv, dtgb__row.csv, dtrw*_0_all.csv, x_all_*.csv, x_maplabelvar_*.csv), the ML_System should successfully load and parse all files without errors.

**Validates: Requirements 1.1**

### Property 2: Multi-Granularity Processing

*For any* input dataset at LOT level or WAFER level, the ML_System should process the data and generate rankings without errors.

**Validates: Requirements 1.3**

### Property 3: Multi-Factor Analysis Support

*For any* input dataset containing multiple factors (1 to N factors), the ML_System should analyze all factors and generate rankings for each.

**Validates: Requirements 1.4**

### Property 4: Schema Validation Error Reporting

*For any* input file with invalid schema (missing required columns, wrong data types), the ML_System should raise a clear error message identifying the specific validation failure.

**Validates: Requirements 1.5**

### Property 5: Ensemble Voting Completeness

*For any* valid input dataset, the ensemble voting mechanism should combine results from all 5 ML algorithms (SHAP, XGBoost, Permutation Importance, Mutual Information, LASSO) to produce final rankings.

**Validates: Requirements 2.8**

### Property 6: Output File Location

*For any* input directory path, the ML_System should write output CSV and JSON files to the same directory.

**Validates: Requirements 3.1, 4.1**

### Property 7: CSV Backward Compatibility

*For any* generated CSV output, the file should have column names, data types, row ordering, and numeric precision matching the Traditional_System output format.

**Validates: Requirements 3.2, 3.3, 3.4, 3.5**

### Property 8: JSON Output Completeness

*For any* generated JSON output, the file should contain all CSV data plus model performance metrics, individual algorithm rankings, ensemble voting results, confidence scores, and metadata.

**Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**

### Property 9: Module Naming Convention

*For any* new Python module created to replace a traditional statistical module, the filename should follow the pattern `{OriginalName}_ML.py`.

**Validates: Requirements 5.1**

### Property 10: Statistical Method Module Existence

*For all* major statistical method replacements (SHAP, XGBoost, Permutation Importance, Mutual Information, LASSO), a corresponding module should exist in the codebase.

**Validates: Requirements 5.3**

### Property 11: Cross-Validation Execution

*For any* ML model training, the system should perform cross-validation with at least 5 folds.

**Validates: Requirements 7.1**

### Property 12: Model Performance Metrics Reporting

*For any* completed model training, the system should calculate and include performance metrics (R², accuracy, F1, CV scores) in the output.

**Validates: Requirements 7.2**

### Property 13: Ranking Confidence Estimation

*For any* generated factor ranking, the system should calculate and include confidence intervals or uncertainty estimates.

**Validates: Requirements 7.3**

### Property 14: Ensemble Agreement Scoring

*For any* ensemble voting result, the system should calculate and report agreement scores across individual algorithms.

**Validates: Requirements 7.4**

### Property 15: Legacy Comparison Metrics

*For any* ML-based ranking output, the system should calculate and report correlation metrics comparing ML rankings to Traditional_System rankings.

**Validates: Requirements 7.5**

### Property 16: Low Performance Flagging

*For any* trained model with performance below acceptable thresholds (R² < 0.3 for regression), the system should flag the results as low confidence.

**Validates: Requirements 7.6, 10.5**

### Property 17: Training Metadata Logging

*For any* model training execution, the system should log all training parameters, performance metrics, and validation results.

**Validates: Requirements 7.7**

### Property 18: Comprehensive Factor Type Processing

*For any* manufacturing dataset, the ML_System should process all factor types (equipment, process, test, manufacturing parameters, metrology) and include them in the analysis.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

### Property 19: Multi-Factor Interaction Analysis

*For any* input dataset, the ML_System should support analysis of both individual factors and multi-factor interactions.

**Validates: Requirements 8.6**

### Property 20: Complete Factor Ranking

*For any* input dataset with N factors, the ML_System should generate rankings for all N factors by their impact on target variables.

**Validates: Requirements 8.7**

### Property 21: Pipeline Stage Sequencing

*For any* pipeline execution, the stages should execute in the correct sequence: Input validation → Data preparation → Feature engineering → Model training → Ranking calculation → Output generation.

**Validates: Requirements 9.1**

### Property 22: Pipeline Stage Completeness

*For any* input dataset, the ML_System should execute all required pipeline stages (data cleaning, feature engineering, train/val split, outlier detection) before model training.

**Validates: Requirements 9.2, 9.3, 9.4, 9.5**

### Property 23: Conditional Time Filtering

*For any* input dataset with temporal constraints specified, the ML_System should perform time-based filtering before analysis.

**Validates: Requirements 9.6**

### Property 24: Pipeline Stage Logging

*For any* pipeline execution, the system should log completion of each stage with timing and data quality metrics.

**Validates: Requirements 9.7**

### Property 25: Missing File Error Reporting

*For any* execution where required input files are missing or corrupted, the ML_System should report specific error messages identifying the problem file and expected format.

**Validates: Requirements 10.1**

### Property 26: Model Training Failure Reporting

*For any* model training failure, the ML_System should report the failure reason and suggest corrective actions.

**Validates: Requirements 10.2**

### Property 27: Insufficient Data Warning

*For any* dataset with sample size below minimum requirements, the ML_System should warn the user and report minimum sample size requirements.

**Validates: Requirements 10.3**

### Property 28: Unexpected Pattern Warning Without Halt

*For any* unexpected data pattern encountered, the ML_System should log warnings but continue execution without stopping.

**Validates: Requirements 10.4**

### Property 29: Multi-Level Logging Implementation

*For any* pipeline execution, the ML_System should generate log entries at all levels (DEBUG, INFO, WARNING, ERROR) as appropriate.

**Validates: Requirements 10.6**

### Property 30: Error Summary Report Generation

*For any* pipeline execution, the ML_System should create a summary report containing all errors and warnings encountered.

**Validates: Requirements 10.7**

### Property 31: Parallel Model Training

*For any* pipeline execution with multiple independent ML models, the system should train models in parallel when possible.

**Validates: Requirements 11.3**

### Property 32: Batch Processing Support

*For any* very large dataset, the ML_System should support batch processing or sampling strategies to handle the data.

**Validates: Requirements 11.5**

### Property 33: Pipeline Stage Timing Reporting

*For any* pipeline execution, the system should report processing time for each major pipeline stage.

**Validates: Requirements 11.6**

## Error Handling

### Error Categories

1. **Input Validation Errors**: Missing files, invalid schema, corrupted data
2. **Data Quality Errors**: Insufficient samples, excessive missing values, invalid ranges
3. **Model Training Errors**: Convergence failures, numerical instability, insufficient features
4. **Output Generation Errors**: File write failures, permission issues, disk space

### Error Handling Strategy

#### Input Validation Errors

**Detection**: During input validation stage before any processing

**Handling**:
- Raise `InputValidationError` with specific details about the problem
- Include expected schema/format in error message
- Log error at ERROR level
- Terminate pipeline execution (cannot proceed without valid input)

**Example**:
```python
if not os.path.exists(dtgb_detail_path):
    raise InputValidationError(
        f"Required file not found: {dtgb_detail_path}. "
        f"Expected dtgb__detail.csv in input directory."
    )
```

#### Data Quality Errors

**Detection**: During data preprocessing stage

**Handling**:
- For insufficient samples: Raise `InsufficientDataError` with minimum sample size requirement
- For excessive missing values: Log WARNING and attempt imputation
- For invalid ranges: Log WARNING and cap/remove outliers
- Include data quality metrics in output metadata

**Example**:
```python
if len(df) < MIN_SAMPLE_SIZE:
    raise InsufficientDataError(
        f"Insufficient samples: {len(df)} rows. "
        f"Minimum required: {MIN_SAMPLE_SIZE} rows for reliable analysis."
    )
```

#### Model Training Errors

**Detection**: During model training stage

**Handling**:
- Catch training exceptions and log detailed error information
- Attempt fallback strategies (e.g., simpler model, different hyperparameters)
- If all models fail: Raise `ModelTrainingError`
- If some models fail: Log WARNING and proceed with successful models
- Flag results as low confidence if model performance is poor

**Example**:
```python
try:
    model.fit(X_train, y_train)
except Exception as e:
    logger.warning(f"Model {model_name} training failed: {e}. Trying fallback...")
    try:
        fallback_model.fit(X_train, y_train)
    except Exception as e2:
        raise ModelTrainingError(f"All training attempts failed for {model_name}: {e2}")
```

#### Output Generation Errors

**Detection**: During output generation stage

**Handling**:
- Catch file I/O exceptions
- Check disk space before writing
- Verify write permissions
- If CSV write fails: Raise `OutputGenerationError` (critical for backward compatibility)
- If JSON write fails: Log ERROR but continue (JSON is enhancement, not critical)

**Example**:
```python
try:
    csv_writer.write_ranking_table(rankings, csv_path)
except IOError as e:
    raise OutputGenerationError(f"Failed to write CSV output: {e}")

try:
    json_writer.write_enhanced_output(rankings, metadata, json_path)
except IOError as e:
    logger.error(f"Failed to write JSON output: {e}. Continuing with CSV only.")
```

### Error Recovery Strategies

1. **Graceful Degradation**: If optional features fail (e.g., JSON output), continue with core functionality
2. **Fallback Models**: If primary ML model fails, attempt simpler alternatives
3. **Partial Results**: If some factors fail analysis, report results for successful factors
4. **Retry Logic**: For transient failures (e.g., file locks), retry with exponential backoff

### Logging Strategy

**Log Levels**:
- **DEBUG**: Detailed diagnostic information (data shapes, intermediate values)
- **INFO**: Pipeline stage completion, model training progress
- **WARNING**: Data quality issues, unexpected patterns, non-critical failures
- **ERROR**: Critical failures that prevent completion

**Log Format**:
```
[TIMESTAMP] [LEVEL] [MODULE] [FUNCTION] - MESSAGE
[2026-01-15 10:30:45] [INFO] [pipeline_orchestrator] [run] - Starting pipeline execution
[2026-01-15 10:30:46] [DEBUG] [data_loader] [load_dtgb_detail] - Loaded 1250 rows from dtgb__detail.csv
[2026-01-15 10:31:02] [WARNING] [data_cleaner] [handle_missing_values] - 5% missing values in CHAMBERID column, applying median imputation
[2026-01-15 10:35:12] [ERROR] [xgboost_model] [fit] - Model training failed: Insufficient features after encoding
```

## Testing Strategy

### Dual Testing Approach

The ML-Stats Migration system requires both unit testing and property-based testing for comprehensive coverage:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Specific input file format examples
- Edge cases (empty files, single-row datasets, all-missing columns)
- Error conditions (missing files, invalid schemas)
- Integration between pipeline stages
- Output format validation against known examples

**Property-Based Tests**: Focus on universal properties across all inputs
- Input file loading for any valid CSV structure
- Pipeline execution for any valid dataset
- Ensemble voting for any combination of model results
- Output generation for any ranking table
- Error handling for any invalid input pattern

### Property-Based Testing Configuration

**Library**: Use `hypothesis` for Python property-based testing

**Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with feature name and property number
- Tag format: `# Feature: ml-stats-migration, Property {N}: {property_text}`

**Example Property Test**:
```python
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as pdst

# Feature: ml-stats-migration, Property 1: Input File Type Compatibility
@given(
    dtgb_detail=pdst.data_frames(
        columns=[
            pdst.column('lot_id', dtype=str),
            pdst.column('wafer_id', dtype=str),
            pdst.column('eqpid', dtype=str),
            # ... more columns
        ],
        rows=st.tuples(st.text(), st.text(), st.text())
    )
)
@settings(max_examples=100)
def test_load_dtgb_detail_any_valid_csv(dtgb_detail):
    """For any valid dtgb_detail CSV structure, loading should succeed."""
    # Write DataFrame to temp CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        dtgb_detail.to_csv(f.name, index=False)
        temp_path = f.name
    
    # Test loading
    loader = DataLoader()
    result = loader.load_dtgb_detail(temp_path)
    
    # Verify successful load
    assert result is not None
    assert len(result) == len(dtgb_detail)
    
    # Cleanup
    os.unlink(temp_path)
```

### Unit Testing Strategy

**Test Organization**:
```
tests/
├── unit/
│   ├── test_data_loader.py
│   ├── test_data_validator.py
│   ├── test_data_cleaner.py
│   ├── test_feature_engineer.py
│   ├── test_shap_model.py
│   ├── test_xgboost_model.py
│   ├── test_permutation_model.py
│   ├── test_mutual_info_model.py
│   ├── test_lasso_model.py
│   ├── test_ensemble_voter.py
│   ├── test_ranking_calculator.py
│   ├── test_csv_writer.py
│   └── test_json_writer.py
├── integration/
│   ├── test_pipeline_end_to_end.py
│   └── test_legacy_compatibility.py
└── property/
    ├── test_properties_input.py
    ├── test_properties_pipeline.py
    ├── test_properties_models.py
    └── test_properties_output.py
```

**Key Unit Test Examples**:

1. **Data Loader Tests**:
   - Test loading each file type with known example
   - Test merging multiple datasets
   - Test handling of missing files (error case)

2. **Model Tests**:
   - Test each ML model with simple synthetic dataset
   - Test model performance metrics calculation
   - Test cross-validation execution

3. **Ensemble Voter Tests**:
   - Test voting with known model rankings
   - Test agreement score calculation
   - Test confidence interval computation

4. **Output Writer Tests**:
   - Test CSV output matches expected format
   - Test JSON output contains all required fields
   - Test numeric precision in CSV output

5. **Integration Tests**:
   - Test complete pipeline with sample manufacturing dataset
   - Test output compatibility with Traditional_System format
   - Test error handling across pipeline stages

### Test Data Strategy

**Synthetic Test Data**:
- Generate small synthetic datasets for unit tests
- Use `hypothesis` strategies for property tests
- Create fixtures for common test scenarios

**Sample Real Data**:
- Use anonymized sample from actual manufacturing data
- Store in `tests/fixtures/sample_data/`
- Use for integration tests and validation

**Edge Case Data**:
- Empty datasets
- Single-row datasets
- All-missing columns
- Extreme outliers
- Invalid data types

### Testing Coverage Goals

- **Line Coverage**: Minimum 80% for all modules
- **Branch Coverage**: Minimum 70% for conditional logic
- **Property Coverage**: All 33 correctness properties implemented as property tests
- **Integration Coverage**: End-to-end pipeline execution with multiple scenarios

### Continuous Testing

**Pre-commit Hooks**:
- Run unit tests before commit
- Run linting and type checking

**CI/CD Pipeline**:
- Run full test suite on every pull request
- Run property tests with 100 iterations
- Generate coverage reports
- Fail build if coverage drops below thresholds

**Performance Benchmarks**:
- Track pipeline execution time on standard dataset
- Alert if performance degrades by >20%
- Monitor memory usage during large dataset processing
