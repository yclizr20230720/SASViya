# Implementation Plan: ML-Stats Migration

## Overview

This implementation plan breaks down the migration of traditional statistical Python code to ML-powered analysis into discrete, incremental coding tasks. The plan follows a modular approach: core infrastructure → data processing → ML models → ensemble logic → output generation → legacy module mapping → testing and validation.

## Tasks

- [ ] 1. Set up project structure and core infrastructure
  - Create directory structure (core/, preprocessing/, models/, ranking/, output/, legacy_mapping/, config/, tests/)
  - Create configuration files (pipeline_config.yaml, model_config.yaml)
  - Implement logging utility (logger.py) with DEBUG, INFO, WARNING, ERROR levels
  - Create requirements.txt with all dependencies (pandas, numpy, scipy, sklearn, xgboost, shap, statsmodels, pandasql, boruta, hypothesis)
  - Set up pytest configuration and test directory structure
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

- [ ] 2. Implement data loading and validation components
  - [ ] 2.1 Implement DataLoader class
    - Write methods to load all CSV file types (dtgb__detail, dtgb__row, dtrw files, x_all files, x_maplabelvar files)
    - Implement merge_datasets() to combine multiple DataFrames
    - Handle file not found errors with clear error messages
    - _Requirements: 1.1, 10.1_
  
  - [ ]* 2.2 Write property test for DataLoader
    - **Property 1: Input File Type Compatibility**
    - **Validates: Requirements 1.1**
  
  - [ ] 2.3 Implement DataValidator class
    - Write schema validation (check required columns exist)
    - Write data type validation
    - Write range validation for numeric columns
    - Implement missing value and duplicate detection reporting
    - _Requirements: 1.5, 10.1_
  
  - [ ]* 2.4 Write property test for DataValidator
    - **Property 4: Schema Validation Error Reporting**
    - **Validates: Requirements 1.5**
  
  - [ ]* 2.5 Write unit tests for data loading edge cases
    - Test empty files, single-row datasets, missing columns
    - Test error messages for corrupted files

- [ ] 3. Implement data preprocessing components
  - [ ] 3.1 Implement DataCleaner class
    - Write handle_missing_values() with multiple strategies (mean, median, forward-fill)
    - Write detect_outliers() using IQR and Z-score methods
    - Write remove_outliers() and cap_outliers() methods
    - Write remove_duplicates() method
    - _Requirements: 9.2, 9.5_
  
  - [ ] 3.2 Implement FeatureEngineer class
    - Write encode_categorical() with one-hot and label encoding
    - Write normalize_continuous() with StandardScaler and MinMaxScaler
    - Write create_interactions() for factor pairs
    - Write extract_temporal_features() for date columns
    - _Requirements: 9.3_
  
  - [ ] 3.3 Implement DataSplitter class
    - Write train/validation split logic with stratification support
    - Support LOT and WAFER level splitting
    - _Requirements: 9.4_
  
  - [ ]* 3.4 Write property test for preprocessing pipeline
    - **Property 22: Pipeline Stage Completeness**
    - **Validates: Requirements 9.2, 9.3, 9.4, 9.5**
  
  - [ ]* 3.5 Write unit tests for preprocessing edge cases
    - Test all-missing columns, extreme outliers, single-category variables

- [ ] 4. Checkpoint - Ensure data pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement base ML model infrastructure
  - [ ] 5.1 Implement BaseModel abstract class
    - Define abstract methods: fit(), compute_importance(), get_performance_metrics()
    - Implement cross_validate() with k-fold cross-validation
    - _Requirements: 7.1_
  
  - [ ]* 5.2 Write property test for cross-validation
    - **Property 11: Cross-Validation Execution**
    - **Validates: Requirements 7.1**

- [ ] 6. Implement individual ML model classes
  - [ ] 6.1 Implement SHAPModel class
    - Use shap.TreeExplainer or shap.KernelExplainer
    - Compute mean absolute SHAP values as importance scores
    - Support both regression and classification
    - Calculate R² or accuracy metrics
    - _Requirements: 2.1, 2.5, 2.7, 6.1, 6.4, 6.6_
  
  - [ ] 6.2 Implement XGBoostModel class
    - Use xgboost.XGBRegressor or xgboost.XGBClassifier
    - Extract gain importance from trained model
    - Set hyperparameters (max_depth=6, learning_rate=0.1, n_estimators=100)
    - Calculate performance metrics
    - _Requirements: 2.5, 2.7, 6.4, 6.6_
  
  - [ ] 6.3 Implement PermutationImportanceModel class
    - Use sklearn.inspection.permutation_importance
    - Use RandomForest as base estimator
    - Set n_repeats=10 and average results
    - _Requirements: 2.3_
  
  - [ ] 6.4 Implement MutualInformationModel class
    - Use sklearn.feature_selection.mutual_info_regression or mutual_info_classif
    - Handle both continuous and discrete features
    - Normalize scores to [0, 1] range
    - Support rank-based MI for Spearman replacement
    - _Requirements: 2.1, 2.6, 6.1, 6.5, 6.7_
  
  - [ ] 6.5 Implement LASSOModel class
    - Use sklearn.linear_model.ElasticNet with l1_ratio=1.0 for pure LASSO
    - Use ElasticNetCV for cross-validated alpha selection
    - Use absolute coefficient values as importance scores
    - _Requirements: 2.2, 6.2_
  
  - [ ]* 6.6 Write property test for model performance metrics
    - **Property 12: Model Performance Metrics Reporting**
    - **Validates: Requirements 7.2**
  
  - [ ]* 6.7 Write unit tests for each ML model
    - Test each model with simple synthetic dataset
    - Test performance metrics calculation
    - Test error handling for training failures

- [ ] 7. Implement ensemble voting and ranking components
  - [ ] 7.1 Implement EnsembleVoter class
    - Write vote() method with weighted average of normalized rankings
    - Write compute_agreement_score() using Kendall's tau
    - Write compute_confidence_intervals() for uncertainty quantification
    - Support configurable model weights from config file
    - _Requirements: 2.8, 7.4_
  
  - [ ]* 7.2 Write property test for ensemble voting
    - **Property 5: Ensemble Voting Completeness**
    - **Validates: Requirements 2.8**
  
  - [ ]* 7.3 Write property test for ensemble agreement
    - **Property 14: Ensemble Agreement Scoring**
    - **Validates: Requirements 7.4**
  
  - [ ] 7.4 Implement RankingCalculator class
    - Write calculate_rankings() to convert ensemble scores to ranked table
    - Write add_legacy_metrics() to include traditional metrics for comparison
    - Write sort_by_metric() for sorting by specified column
    - _Requirements: 7.5, 8.7_
  
  - [ ] 7.5 Implement ConfidenceEstimator class
    - Write estimate_confidence() based on model agreement
    - Write compute_bootstrap_ci() for bootstrap confidence intervals
    - Write flag_low_confidence() to identify uncertain rankings (threshold < 0.5)
    - _Requirements: 7.3, 7.6, 10.5_
  
  - [ ]* 7.6 Write property test for ranking confidence
    - **Property 13: Ranking Confidence Estimation**
    - **Validates: Requirements 7.3**
  
  - [ ]* 7.7 Write property test for low performance flagging
    - **Property 16: Low Performance Flagging**
    - **Validates: Requirements 7.6, 10.5**

- [ ] 8. Checkpoint - Ensure ML model tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement output generation components
  - [ ] 9.1 Implement CSVWriter class
    - Write write_ranking_table() to generate CSV with legacy format
    - Write format_legacy_columns() to match Traditional_System column names
    - Write apply_legacy_precision() for numeric formatting
    - Ensure output includes: Factor, R, RW, Sigma, PValue, PPlus, PMinus, ChiSquare, FScore, SampleSize, EffectSize
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ]* 9.2 Write property test for CSV backward compatibility
    - **Property 7: CSV Backward Compatibility**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5**
  
  - [ ] 9.3 Implement JSONWriter class
    - Write write_enhanced_output() to generate comprehensive JSON
    - Write build_json_structure() with hierarchical organization (metadata, model_performance, ensemble, rankings)
    - Write add_model_metrics() to include individual model performance
    - Write add_ensemble_details() to include voting results and confidence scores
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  
  - [ ]* 9.4 Write property test for JSON output completeness
    - **Property 8: JSON Output Completeness**
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**
  
  - [ ]* 9.5 Write property test for output file location
    - **Property 6: Output File Location**
    - **Validates: Requirements 3.1, 4.1**
  
  - [ ]* 9.6 Write unit tests for output formatting
    - Test CSV numeric precision matches legacy system
    - Test JSON structure is valid and complete
    - Test file write error handling

- [ ] 10. Implement pipeline orchestration
  - [ ] 10.1 Implement PipelineOrchestrator class
    - Write run() method to execute complete pipeline
    - Write validate_inputs() to check file existence and validity
    - Write execute_stage() with error handling for each stage
    - Write handle_error() to log errors and determine continuation
    - Implement stage sequencing: validation → preprocessing → feature engineering → model training → ranking → output
    - Add timing and logging for each stage
    - _Requirements: 9.1, 9.7, 10.1, 10.2, 10.4, 11.6_
  
  - [ ]* 10.2 Write property test for pipeline stage sequencing
    - **Property 21: Pipeline Stage Sequencing**
    - **Validates: Requirements 9.1**
  
  - [ ]* 10.3 Write property test for pipeline stage logging
    - **Property 24: Pipeline Stage Logging**
    - **Validates: Requirements 9.7**
  
  - [ ]* 10.4 Write property test for error reporting
    - **Property 25: Missing File Error Reporting**
    - **Property 26: Model Training Failure Reporting**
    - **Validates: Requirements 10.1, 10.2**

- [ ] 11. Implement parallel processing and performance optimizations
  - [ ] 11.1 Add parallel model training support
    - Use joblib or multiprocessing to train models in parallel
    - Make parallelization configurable via pipeline_config.yaml
    - _Requirements: 11.3_
  
  - [ ] 11.2 Implement batch processing support
    - Add batch processing mode for very large datasets
    - Implement sampling strategies for datasets exceeding memory limits
    - _Requirements: 11.5_
  
  - [ ]* 11.3 Write property test for parallel processing
    - **Property 31: Parallel Model Training**
    - **Validates: Requirements 11.3**

- [ ] 12. Checkpoint - Ensure pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement legacy mapping modules
  - [ ] 13.1 Implement CommonTool_ML module
    - Write chi_square_ml() using Mutual Information
    - Write t_test_ml() using Permutation Importance
    - Write f_test_ml() using Permutation Importance
    - _Requirements: 6.5_
  
  - [ ] 13.2 Implement Index_PearsonCorr_ML module
    - Write compute_correlation_ml() using SHAP + Mutual Information
    - _Requirements: 6.1_
  
  - [ ] 13.3 Implement FCN_RW_ML module
    - Write compute_weighted_correlation_ml() using LASSO/Elastic Net
    - _Requirements: 6.2_
  
  - [ ] 13.4 Implement StatIndexCal_ML module
    - Write compute_pvalue_ml() using Boruta/RFECV
    - Write compute_fscore_ml() using XGBoost/SHAP
    - _Requirements: 6.3, 6.6_
  
  - [ ] 13.5 Implement SummaryIndex_ML module
    - Write compute_pplus_ml() using XGBoost Gain + SHAP
    - Write compute_pminus_ml() using XGBoost Gain + SHAP
    - _Requirements: 6.4_
  
  - [ ] 13.6 Implement Index_SpearmanCorr_ML module
    - Write compute_spearman_ml() using rank-based Mutual Information
    - _Requirements: 6.7_
  
  - [ ]* 13.7 Write property test for module naming convention
    - **Property 9: Module Naming Convention**
    - **Validates: Requirements 5.1**
  
  - [ ]* 13.8 Write unit tests for legacy module compatibility
    - Test each _ML module produces output compatible with original module
    - Test algorithm mapping is correct (e.g., Pearson → SHAP+MI)

- [ ] 14. Implement comprehensive error handling and logging
  - [ ] 14.1 Add error handling for all error categories
    - Implement InputValidationError, InsufficientDataError, ModelTrainingError, OutputGenerationError
    - Add specific error messages with corrective action suggestions
    - Implement graceful degradation (continue with partial results when possible)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [ ] 14.2 Implement comprehensive logging
    - Add DEBUG logs for data shapes and intermediate values
    - Add INFO logs for pipeline stage completion
    - Add WARNING logs for data quality issues
    - Add ERROR logs for critical failures
    - _Requirements: 10.6, 10.7_
  
  - [ ] 14.3 Implement error summary report generation
    - Create summary report with all errors and warnings
    - Include error counts, timestamps, and affected components
    - _Requirements: 10.7_
  
  - [ ]* 14.4 Write property test for multi-level logging
    - **Property 29: Multi-Level Logging Implementation**
    - **Validates: Requirements 10.6**
  
  - [ ]* 14.5 Write property test for error summary report
    - **Property 30: Error Summary Report Generation**
    - **Validates: Requirements 10.7**
  
  - [ ]* 14.6 Write property test for insufficient data warning
    - **Property 27: Insufficient Data Warning**
    - **Validates: Requirements 10.3**
  
  - [ ]* 14.7 Write property test for unexpected pattern handling
    - **Property 28: Unexpected Pattern Warning Without Halt**
    - **Validates: Requirements 10.4**

- [ ] 15. Implement factor type processing and analysis
  - [ ] 15.1 Add factor type identification
    - Identify equipment factors (EQPID, CHAMBERID, CHAMBER_GROUP)
    - Identify process factors (Process Stage, Operation Number, RECIPENAME, EQPRECIPEID)
    - Identify test factors (Test Program, Tester, Probe Card)
    - Identify manufacturing parameters (MANUF_NO, CUST_CODE, SUPPLIER, RESIST_VALUE)
    - Identify metrology-induced factors
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ] 15.2 Add multi-factor interaction analysis
    - Implement interaction term generation for factor pairs
    - Add interaction analysis to feature engineering pipeline
    - _Requirements: 8.6_
  
  - [ ]* 15.3 Write property test for comprehensive factor processing
    - **Property 18: Comprehensive Factor Type Processing**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
  
  - [ ]* 15.4 Write property test for multi-factor interaction analysis
    - **Property 19: Multi-Factor Interaction Analysis**
    - **Validates: Requirements 8.6**
  
  - [ ]* 15.5 Write property test for complete factor ranking
    - **Property 20: Complete Factor Ranking**
    - **Validates: Requirements 8.7**

- [ ] 16. Implement multi-granularity and time-based processing
  - [ ] 16.1 Add LOT and WAFER level processing support
    - Implement granularity detection from input data
    - Add granularity-specific aggregation logic
    - _Requirements: 1.3_
  
  - [ ] 16.2 Add time-based filtering
    - Implement temporal constraint parsing from config
    - Add time-based filtering in preprocessing stage
    - _Requirements: 9.6_
  
  - [ ]* 16.3 Write property test for multi-granularity processing
    - **Property 2: Multi-Granularity Processing**
    - **Validates: Requirements 1.3**
  
  - [ ]* 16.4 Write property test for conditional time filtering
    - **Property 23: Conditional Time Filtering**
    - **Validates: Requirements 9.6**

- [ ] 17. Checkpoint - Ensure all feature tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. Implement integration tests and end-to-end validation
  - [ ]* 18.1 Write end-to-end integration test
    - Test complete pipeline with sample manufacturing dataset
    - Verify CSV and JSON outputs are generated correctly
    - Verify all 5 ML models are trained and contribute to ensemble
    - _Requirements: All requirements_
  
  - [ ]* 18.2 Write legacy compatibility integration test
    - Compare ML_System output format against Traditional_System output
    - Verify CSV schema matches exactly
    - Verify numeric precision matches
    - _Requirements: 3.2, 3.3, 3.4, 3.5_
  
  - [ ]* 18.3 Write property test for multi-factor analysis
    - **Property 3: Multi-Factor Analysis Support**
    - **Validates: Requirements 1.4**
  
  - [ ]* 18.4 Write property test for legacy comparison metrics
    - **Property 15: Legacy Comparison Metrics**
    - **Validates: Requirements 7.5**
  
  - [ ]* 18.5 Write property test for training metadata logging
    - **Property 17: Training Metadata Logging**
    - **Validates: Requirements 7.7**
  
  - [ ]* 18.6 Write property test for pipeline stage timing
    - **Property 33: Pipeline Stage Timing Reporting**
    - **Validates: Requirements 11.6**

- [ ] 19. Create sample data and documentation
  - [ ] 19.1 Create sample test data
    - Generate anonymized sample manufacturing dataset
    - Create fixtures for common test scenarios
    - Store in tests/fixtures/sample_data/
  
  - [ ] 19.2 Create configuration examples
    - Create example pipeline_config.yaml with comments
    - Create example model_config.yaml with hyperparameter explanations
  
  - [ ] 19.3 Create README documentation
    - Document installation instructions
    - Document usage examples
    - Document configuration options
    - Document ML algorithm mappings from traditional methods
  
  - [ ] 19.4 Create API documentation
    - Document all public classes and methods
    - Include usage examples for each component
    - Document error types and handling

- [ ] 20. Final checkpoint - Run complete test suite
  - Run all unit tests, property tests, and integration tests
  - Verify test coverage meets minimum thresholds (80% line coverage, 70% branch coverage)
  - Run end-to-end pipeline with sample data and verify outputs
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties with minimum 100 iterations
- Unit tests validate specific examples, edge cases, and error conditions
- The implementation follows a bottom-up approach: infrastructure → data processing → ML models → ensemble → output → integration
- All ML models inherit from BaseModel for consistent interface
- Configuration files externalize hyperparameters for easy tuning
- Legacy mapping modules provide clear traceability to original statistical methods
