# Requirements Document: ML Analysis Web Dashboard

## Introduction

The ML Analysis Web Dashboard is a professional, self-guided web application for managing and visualizing ML-powered factor ranking analysis jobs in semiconductor wafer manufacturing. The system enables engineers to create analysis jobs, monitor ML pipeline execution in real-time, and perform comprehensive root cause analysis through interactive dashboards and visualizations.

The application integrates with the ML-Stats-Migration backend (Python ML pipeline) to execute analysis jobs and retrieve results. It provides a modern React-based frontend with Flask backend, PostgreSQL database, and real-time communication capabilities.

## Glossary

- **Analysis_Job**: A user-created task that executes ML pipeline analysis on semiconductor manufacturing data
- **ML_Pipeline**: The backend Python system that processes data through Input Layer, Pipeline Layer, and Output Layer
- **Factor**: A manufacturing variable (equipment, chamber, process, recipe, metrology) that may influence wafer yield or quality
- **Ensemble_Score**: Weighted composite ranking from five ML models (SHAP, XGBoost, Permutation Importance, Mutual Information, LASSO)
- **Root_Cause**: A factor identified as having significant negative impact on yield or quality
- **SHAP**: SHapley Additive exPlanations - ML interpretability method measuring feature contribution
- **XGBoost**: Gradient boosting ML model providing feature importance metrics (Gain, Weight, Coverage)
- **Permutation_Importance**: ML metric measuring accuracy drop when a feature is shuffled
- **Mutual_Information**: Statistical measure of dependency between a factor and target variable
- **LASSO**: Least Absolute Shrinkage and Selection Operator - regularized regression with feature selection
- **Dashboard**: Web interface displaying job status, results, and visualizations
- **Data_Grid**: Interactive table component supporting sorting, filtering, searching, and export
- **WebSocket**: Real-time bidirectional communication protocol for live status updates
- **Flask_API**: RESTful backend service handling requests between frontend and ML pipeline
- **PostgreSQL_Database**: Relational database storing jobs, results, users, and configurations
- **Wafer**: Silicon disc used in semiconductor manufacturing
- **Lot**: Batch of wafers processed together
- **Yield**: Percentage of functional chips produced from wafers
- **Chamber**: Equipment compartment where wafer processing occurs
- **Recipe**: Set of process parameters for manufacturing operations
- **FDC**: Fault Detection and Classification system
- **CP**: Chip Probing test data
- **WAT**: Wafer Acceptance Test data
- **Q-time**: Queue time between manufacturing steps
- **U-Chart**: Statistical control chart for monitoring defect rates

## Requirements

### Requirement 1: Analysis Job Creation and Type Selection

**User Story:** As a manufacturing engineer, I want to create analysis jobs by selecting from predefined analysis types, so that I can investigate specific relationships between manufacturing factors and yield outcomes.

#### Acceptance Criteria

1. WHEN a user accesses the job creation interface, THE Dashboard SHALL display six analysis type options: "CP VS N-chamber", "WAT VS N-chamber", "CP/WAT VS Q-time (wafer level)", "CP VS FDC U-Chart", "WAT VS FDC U-Chart", and "CP_WAT (CP vs WAT)"
2. WHEN a user selects an analysis type, THE Dashboard SHALL display the specific data file requirements for that type
3. WHEN a user provides a job name and description, THE Dashboard SHALL validate that the name is unique and non-empty
4. WHEN a user submits job creation, THE Dashboard SHALL create an Analysis_Job record with status "Pending File Upload"
5. THE Dashboard SHALL assign a unique job identifier to each created Analysis_Job

### Requirement 2: Data File Upload and Validation

**User Story:** As a manufacturing engineer, I want to upload required data files for my analysis job, so that the ML pipeline has the necessary input data to execute.

#### Acceptance Criteria

1. WHEN a user views job details for a pending job, THE Dashboard SHALL display a checklist of required data files based on the analysis type
2. WHEN a user uploads a file, THE Dashboard SHALL validate the file name matches the expected pattern for that analysis type
3. WHEN a user uploads a file, THE Dashboard SHALL validate the file format is CSV and contains expected column headers
4. WHEN all required files are uploaded, THE Dashboard SHALL automatically change job status to "Ready to Start"
5. IF a file upload fails validation, THEN THE Dashboard SHALL display a descriptive error message and prevent job execution
6. WHEN a user uploads a file exceeding 500MB, THE Dashboard SHALL reject the upload and display a size limit error

### Requirement 3: Automatic Job Execution Trigger

**User Story:** As a manufacturing engineer, I want my analysis job to automatically start when all required files are uploaded, so that I don't need to manually trigger execution.

#### Acceptance Criteria

1. WHEN all required data files are validated and uploaded, THE Dashboard SHALL automatically submit the job to the ML_Pipeline
2. WHEN job execution starts, THE Dashboard SHALL update job status to "Running" and record the start timestamp
3. WHEN the ML_Pipeline accepts the job, THE Dashboard SHALL receive a job execution identifier for tracking
4. IF the ML_Pipeline rejects the job, THEN THE Dashboard SHALL update status to "Failed" and display the rejection reason
5. THE Dashboard SHALL persist job status changes to the PostgreSQL_Database immediately

### Requirement 4: Real-Time Pipeline Stage Monitoring

**User Story:** As a manufacturing engineer, I want to see real-time updates of my job's progress through the ML pipeline stages, so that I can monitor execution and estimate completion time.

#### Acceptance Criteria

1. WHEN a job is running, THE Dashboard SHALL display the current pipeline stage: "Input Layer", "Pipeline Layer", or "Output Layer"
2. WHEN the pipeline stage changes, THE Dashboard SHALL update the display within 2 seconds via WebSocket
3. WHILE a job is in "Input Layer", THE Dashboard SHALL show "Data Loader" sub-stage status
4. WHILE a job is in "Pipeline Layer", THE Dashboard SHALL show "ML Model Ensemble" sub-stage status with individual model progress
5. WHILE a job is in "Output Layer", THE Dashboard SHALL show "CSV Writer" and "JSON Writer" sub-stage status
6. THE Dashboard SHALL display progress percentage for each stage (0-100%)
7. THE Dashboard SHALL display elapsed time since job start
8. THE Dashboard SHALL display estimated remaining time based on historical job durations

### Requirement 5: ML Model Execution Monitoring

**User Story:** As a manufacturing engineer, I want to monitor individual ML model execution status and key metrics, so that I can understand which models are running and their performance.

#### Acceptance Criteria

1. WHEN a job enters "Pipeline Layer", THE Dashboard SHALL display status for five ML models: SHAP, XGBoost, Permutation_Importance, Mutual_Information, and LASSO
2. WHEN a model starts execution, THE Dashboard SHALL update its status to "Training" or "Computing Importance"
3. WHEN a model completes, THE Dashboard SHALL display key output metrics: R² score, accuracy, F1 score, cross-validation scores, training time, and number of features ranked
4. WHEN a model completes, THE Dashboard SHALL display the top 5 most important factors with their scores
5. IF a model fails, THEN THE Dashboard SHALL update status to "Failed" and display the error message
6. THE Dashboard SHALL update model status in real-time via WebSocket within 2 seconds of backend changes

### Requirement 6: Job History and Search

**User Story:** As a manufacturing engineer, I want to view all my past analysis jobs with filtering and search capabilities, so that I can find and review previous analyses.

#### Acceptance Criteria

1. WHEN a user accesses the job history page, THE Dashboard SHALL display all Analysis_Job records for that user sorted by creation date descending
2. WHEN a user enters a search query, THE Dashboard SHALL filter jobs by job name, job ID, or analysis type containing the query text
3. WHEN a user selects a date range filter, THE Dashboard SHALL display only jobs created within that range
4. WHEN a user selects a status filter, THE Dashboard SHALL display only jobs matching that status (Pending, Running, Complete, Failed)
5. WHEN a user clicks on a job row, THE Dashboard SHALL navigate to the job detail page
6. THE Dashboard SHALL paginate job history results with 50 jobs per page

### Requirement 7: Interactive Data Grid for Ranking Results

**User Story:** As a manufacturing engineer, I want to view factor ranking results in an interactive data grid with sorting, filtering, and export capabilities, so that I can analyze and share findings efficiently.

#### Acceptance Criteria

1. WHEN a job completes successfully, THE Dashboard SHALL display ranking results in a Data_Grid with columns: Rank, Factor Name, Factor Type, Ensemble_Score, Confidence Level, SHAP Score, XGBoost Score, Permutation Score, MI Score, LASSO Score, Sample Size, Effect Size
2. WHEN a user clicks a column header, THE Data_Grid SHALL sort results by that column in ascending order
3. WHEN a user clicks a sorted column header again, THE Data_Grid SHALL toggle to descending order
4. WHEN a user enters text in the search box, THE Data_Grid SHALL filter rows where any column contains the search text
5. WHEN a user selects a factor type filter, THE Data_Grid SHALL display only rows matching that factor type
6. WHEN a user selects a score range filter, THE Data_Grid SHALL display only rows where Ensemble_Score falls within that range
7. WHEN a user clicks the export button, THE Data_Grid SHALL generate a CSV file containing all filtered and sorted data
8. WHEN a user clicks the export button with Excel format selected, THE Data_Grid SHALL generate an XLSX file
9. WHEN a user marks a factor row, THE Data_Grid SHALL visually highlight that row and persist the mark to the database
10. WHEN the Data_Grid contains more than 100 rows, THE Data_Grid SHALL paginate results with configurable page size
11. WHEN a user reorders columns by drag-and-drop, THE Data_Grid SHALL persist the column order preference
12. WHEN a user hides a column, THE Data_Grid SHALL remove it from view and persist the visibility preference

### Requirement 8: Ensemble Bar Chart Visualization

**User Story:** As a manufacturing engineer, I want to see a stacked bar chart showing how each ML model contributes to the ensemble score, so that I can understand the consensus and disagreement between models.

#### Acceptance Criteria

1. WHEN a user views the results page, THE Dashboard SHALL display an ensemble bar chart for the top 20 factors
2. THE Dashboard SHALL render each bar as a stacked composition showing SHAP contribution (25%), XGBoost contribution (25%), Permutation contribution (20%), MI contribution (15%), and LASSO contribution (15%)
3. WHEN a user hovers over a bar segment, THE Dashboard SHALL display a tooltip showing the model name and exact score value
4. WHEN a user clicks on a bar, THE Dashboard SHALL navigate to the detailed factor analysis page for that factor
5. THE Dashboard SHALL color-code each model segment consistently across all charts (SHAP=blue, XGBoost=green, Permutation=orange, MI=purple, LASSO=red)
6. WHEN a user clicks the export button, THE Dashboard SHALL save the chart as PNG or SVG format

### Requirement 9: Advanced Chart Types for Factor Analysis

**User Story:** As a manufacturing engineer, I want to visualize factor importance using multiple chart types (trend, box plot, scatter, heatmap, waterfall, radar), so that I can gain different analytical perspectives on the data.

#### Acceptance Criteria

1. WHEN a user selects "Trend Chart" view, THE Dashboard SHALL display factor importance over time for historical job comparison
2. WHEN a user selects "Box Plot" view, THE Dashboard SHALL display distribution of factor scores across different analysis types
3. WHEN a user selects "Scatter Plot" view, THE Dashboard SHALL display correlation between two selected ML model scores
4. WHEN a user selects "Heatmap" view, THE Dashboard SHALL display a factor importance matrix across multiple jobs or time periods
5. WHEN a user selects "Waterfall Chart" view, THE Dashboard SHALL display contribution breakdown for ensemble scores
6. WHEN a user selects "Radar Chart" view, THE Dashboard SHALL display multi-dimensional factor comparison across all five ML models
7. WHEN a user interacts with any chart (zoom, pan, brush selection), THE Dashboard SHALL update the chart view responsively
8. WHEN a user hovers over any chart element, THE Dashboard SHALL display contextual tooltips with detailed values
9. THE Dashboard SHALL render all charts responsively to fit different screen sizes (desktop, tablet, mobile)

### Requirement 10: Executive Dashboard Summary

**User Story:** As a manufacturing manager, I want to see an at-a-glance executive dashboard summarizing job status, top root causes, model performance, and data quality, so that I can quickly assess analysis results without diving into details.

#### Acceptance Criteria

1. WHEN a user views a completed job, THE Dashboard SHALL display a Job Summary Card showing job ID, analysis type, status, start/end time, total processing time, and data size
2. THE Dashboard SHALL display a Top Killers Card showing the top 5 root cause factors with ensemble scores and severity indicators (red for critical, yellow for moderate, green for minor)
3. THE Dashboard SHALL display a Model Performance Card showing overall ensemble agreement score, individual model performance metrics, and a model consensus visualization
4. THE Dashboard SHALL display a Data Quality Card showing input data completeness percentage, missing value percentage, and outlier detection summary
5. WHEN historical jobs exist, THE Dashboard SHALL display a Historical Comparison Card comparing the current job with previous runs and showing trend indicators
6. WHEN a user clicks "View Details" on any card, THE Dashboard SHALL navigate to the corresponding detailed analysis page
7. WHEN a user clicks "Generate Report" on the Top Killers Card, THE Dashboard SHALL create a PDF report summarizing root causes

### Requirement 11: Comprehensive ML Model Explanation - SHAP

**User Story:** As a manufacturing engineer, I want to understand SHAP model results with clear explanations, visualizations, and interpretation guidance, so that I can correctly identify root causes.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display a SHAP section with the metric definition: "Mean |SHAP value| per factor"
2. THE Dashboard SHALL display interpretation guidance: "High positive SHAP → Factor increases yield/quality; High negative SHAP → Factor decreases yield/quality (ROOT CAUSE); Near-zero SHAP → Factor has minimal impact"
3. THE Dashboard SHALL display a SHAP waterfall chart showing top factors and their contribution direction
4. THE Dashboard SHALL display a SHAP force plot showing how factors push predictions up or down
5. THE Dashboard SHALL display a SHAP dependence plot showing factor value vs SHAP value relationship
6. THE Dashboard SHALL display the key insight: "SHAP reveals non-linear relationships that traditional correlation misses"
7. WHEN a user hovers over any SHAP visualization element, THE Dashboard SHALL display contextual tooltips explaining the values

### Requirement 12: Comprehensive ML Model Explanation - XGBoost

**User Story:** As a manufacturing engineer, I want to understand XGBoost model results with clear explanations of Gain, Weight, and Coverage metrics, so that I can interpret feature importance correctly.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display an XGBoost section with three metrics: Gain, Weight, and Coverage
2. THE Dashboard SHALL display metric definitions: "Gain = Information gain when splitting on this factor; Weight = How often this factor is used for splitting; Coverage = Fraction of samples affected by this factor"
3. THE Dashboard SHALL display interpretation guidance: "High Gain → Strong predictive power; High Weight → Consistently important; High Coverage → Affects many wafers/lots"
4. THE Dashboard SHALL display a horizontal bar chart showing Gain, Weight, and Coverage for top factors
5. THE Dashboard SHALL display a tree structure visualization showing factor split points
6. THE Dashboard SHALL display the key insight: "XGBoost captures feature interactions that linear methods cannot"

### Requirement 13: Comprehensive ML Model Explanation - Permutation Importance

**User Story:** As a manufacturing engineer, I want to understand Permutation Importance results with clear explanations of accuracy drop metrics, so that I can identify truly critical factors.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display a Permutation_Importance section with the metric: "Mean accuracy drop when factor is shuffled"
2. THE Dashboard SHALL display interpretation guidance: "Large drop → Factor is critical for prediction (ROOT CAUSE); Small drop → Factor is less important; Negative drop → Factor may be noise"
3. THE Dashboard SHALL display a bar chart with error bars (standard deviation) showing permutation importance scores
4. THE Dashboard SHALL display a comparison line showing baseline model performance
5. THE Dashboard SHALL display the key insight: "Permutation importance reflects true predictive value, not just correlation"

### Requirement 14: Comprehensive ML Model Explanation - Mutual Information

**User Story:** As a manufacturing engineer, I want to understand Mutual Information results with clear explanations of dependency detection, so that I can identify non-linear relationships.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display a Mutual_Information section with the metric: "MI score (0 to 1 normalized)"
2. THE Dashboard SHALL display interpretation guidance: "MI close to 1 → Strong dependency (linear or non-linear); MI close to 0 → Independent, no relationship"
3. THE Dashboard SHALL display a sorted bar chart of MI scores for all factors
4. THE Dashboard SHALL display an MI matrix heatmap showing factor interaction strengths
5. THE Dashboard SHALL display the key insight: "MI detects any type of dependency, including non-linear and non-monotonic"

### Requirement 15: Comprehensive ML Model Explanation - LASSO

**User Story:** As a manufacturing engineer, I want to understand LASSO model results with clear explanations of coefficient values and regularization, so that I can identify linear effects and automatic feature selection.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display a LASSO section with the metric: "Absolute coefficient value"
2. THE Dashboard SHALL display interpretation guidance: "Large coefficient → Strong linear effect; Coefficient = 0 → Factor eliminated by regularization (not important); Sign indicates direction (positive/negative impact)"
3. THE Dashboard SHALL display a regularization path plot showing how coefficients change with regularization strength
4. THE Dashboard SHALL display a coefficient bar chart showing final coefficient values
5. THE Dashboard SHALL display the key insight: "LASSO handles multicollinearity and performs automatic feature selection"

### Requirement 16: Ensemble Composite Index Explanation

**User Story:** As a manufacturing engineer, I want to understand how the ensemble score is calculated and what confidence levels mean, so that I can trust the composite rankings.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display the ensemble formula: "Ensemble Score = 0.25×SHAP + 0.25×XGBoost + 0.20×Perm + 0.15×MI + 0.15×LASSO"
2. THE Dashboard SHALL display interpretation guidance: "High ensemble score + high confidence → Definite ROOT CAUSE; High ensemble score + low confidence → Investigate further (models disagree); Low ensemble score → Likely not a root cause"
3. THE Dashboard SHALL display how confidence is calculated: "Based on agreement between models (Kendall's tau)"
4. THE Dashboard SHALL display an ensemble bar chart with color-coded segments showing each model's contribution
5. THE Dashboard SHALL display confidence interval error bars on the ensemble chart
6. THE Dashboard SHALL display an agreement score gauge showing model consensus level
7. THE Dashboard SHALL display the key insight: "Ensemble voting reduces false positives and provides robust rankings"

### Requirement 17: Root Cause Identification and Factor Type Breakdown

**User Story:** As a manufacturing engineer, I want to see root causes organized by factor type (equipment, chamber, process, recipe, metrology) with severity indicators, so that I can prioritize corrective actions.

#### Acceptance Criteria

1. WHEN a user views the Root Cause Analysis page, THE Dashboard SHALL display factors grouped by type: Equipment/Tool, Chamber, Process, Test, Manufacturing Parameters, and Metrology-induced
2. THE Dashboard SHALL display a Killer Factor Summary showing the top 10 factors ranked by Ensemble_Score
3. THE Dashboard SHALL display visual severity indicators: red for critical (score > 0.8), yellow for moderate (score 0.5-0.8), green for minor (score < 0.5)
4. THE Dashboard SHALL display impact quantification for each killer factor: "This factor accounts for X% of yield loss"
5. THE Dashboard SHALL display a Factor Interaction Analysis section showing which factors interact and interaction strength
6. THE Dashboard SHALL display Actionable Recommendations based on top factors, such as "Investigate Chamber C123 - shows 85% ensemble score"

### Requirement 18: Factor Drill-Down and Detailed View

**User Story:** As a manufacturing engineer, I want to click on any factor to see detailed analysis including all ML model scores, historical trends, and wafer-level breakdown, so that I can perform deep investigation.

#### Acceptance Criteria

1. WHEN a user clicks on a factor in any view, THE Dashboard SHALL navigate to a detailed factor analysis page
2. THE Dashboard SHALL display all five ML model scores side-by-side in a comparison table
3. THE Dashboard SHALL display a historical trend chart showing this factor's scores across previous jobs
4. THE Dashboard SHALL display a wafer/lot level breakdown showing which specific wafers or lots are affected
5. THE Dashboard SHALL provide a link to export raw data for this factor as CSV
6. THE Dashboard SHALL display related factors that show similar patterns or correlations
7. WHEN a user clicks "Export Root Cause Report", THE Dashboard SHALL generate a PDF or PowerPoint file summarizing the factor analysis

### Requirement 19: User Authentication and Role-Based Access

**User Story:** As a system administrator, I want to manage user accounts with role-based permissions (Engineer, Manager, Admin), so that I can control access to sensitive manufacturing data.

#### Acceptance Criteria

1. WHEN a user accesses the Dashboard, THE Dashboard SHALL require authentication via username and password
2. WHEN a user logs in successfully, THE Dashboard SHALL create a session token valid for 8 hours
3. THE Dashboard SHALL support three roles: Engineer (create/view own jobs), Manager (view all jobs, generate reports), Admin (all permissions plus user management)
4. WHEN an Engineer attempts to access another user's job, THE Dashboard SHALL deny access and display an authorization error
5. WHEN a Manager accesses the Dashboard, THE Dashboard SHALL display all jobs from all users
6. WHEN an Admin accesses user management, THE Dashboard SHALL display options to create, edit, disable, and delete user accounts
7. WHEN a user session expires, THE Dashboard SHALL redirect to the login page and display a session timeout message

### Requirement 20: Job Comparison Mode

**User Story:** As a manufacturing engineer, I want to compare results from multiple analysis jobs side-by-side, so that I can identify trends and changes in root causes over time.

#### Acceptance Criteria

1. WHEN a user selects multiple jobs from the job history page, THE Dashboard SHALL enable a "Compare Jobs" button
2. WHEN a user clicks "Compare Jobs", THE Dashboard SHALL display a comparison view with jobs arranged in columns
3. THE Dashboard SHALL display top 10 factors from each job aligned by factor name for easy comparison
4. THE Dashboard SHALL highlight factors that appear in multiple jobs with a visual indicator
5. THE Dashboard SHALL display a trend chart showing how ensemble scores for common factors change across jobs
6. THE Dashboard SHALL display a difference summary showing which factors increased or decreased in importance
7. WHEN a user exports the comparison, THE Dashboard SHALL generate a CSV or PDF report with side-by-side results

### Requirement 21: Alerts and Notifications

**User Story:** As a manufacturing engineer, I want to receive email or SMS notifications when my job completes or when critical root causes are detected, so that I can respond quickly to manufacturing issues.

#### Acceptance Criteria

1. WHEN a user creates a job, THE Dashboard SHALL allow the user to opt-in to email and/or SMS notifications
2. WHEN a job completes successfully, THE Dashboard SHALL send a notification to the user with job ID, completion time, and a link to results
3. WHEN a job fails, THE Dashboard SHALL send a notification with job ID, failure reason, and troubleshooting suggestions
4. WHEN a job detects a critical root cause (Ensemble_Score > 0.8), THE Dashboard SHALL send an alert notification highlighting the killer factor
5. THE Dashboard SHALL allow users to configure notification preferences in their profile settings
6. THE Dashboard SHALL rate-limit notifications to prevent spam (maximum 10 notifications per hour per user)

### Requirement 22: RESTful API for Programmatic Access

**User Story:** As a data scientist, I want to access the Dashboard functionality via a RESTful API, so that I can integrate analysis jobs into automated workflows and scripts.

#### Acceptance Criteria

1. THE Flask_API SHALL provide an endpoint POST /api/v1/jobs to create new analysis jobs
2. THE Flask_API SHALL provide an endpoint GET /api/v1/jobs/{job_id} to retrieve job status and results
3. THE Flask_API SHALL provide an endpoint GET /api/v1/jobs to list all jobs with filtering and pagination
4. THE Flask_API SHALL provide an endpoint POST /api/v1/jobs/{job_id}/files to upload data files
5. THE Flask_API SHALL provide an endpoint GET /api/v1/jobs/{job_id}/results to retrieve ranking results in JSON format
6. THE Flask_API SHALL require API key authentication for all endpoints
7. THE Flask_API SHALL return appropriate HTTP status codes (200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 500 Internal Server Error)
8. THE Flask_API SHALL document all endpoints using OpenAPI/Swagger specification

### Requirement 23: Responsive Design and Theme Support

**User Story:** As a manufacturing engineer, I want the Dashboard to work on desktop, tablet, and mobile devices with dark/light theme options, so that I can access analysis results from anywhere.

#### Acceptance Criteria

1. WHEN a user accesses the Dashboard on a desktop (screen width > 1024px), THE Dashboard SHALL display the full layout with sidebar navigation
2. WHEN a user accesses the Dashboard on a tablet (screen width 768-1024px), THE Dashboard SHALL display a responsive layout with collapsible sidebar
3. WHEN a user accesses the Dashboard on a mobile device (screen width < 768px), THE Dashboard SHALL display a mobile-optimized layout with hamburger menu navigation
4. WHEN a user selects dark theme, THE Dashboard SHALL apply a dark color scheme to all pages and components
5. WHEN a user selects light theme, THE Dashboard SHALL apply a light color scheme to all pages and components
6. THE Dashboard SHALL persist theme preference in the user's profile
7. THE Dashboard SHALL ensure all charts and visualizations are readable in both dark and light themes

### Requirement 24: Performance and Scalability

**User Story:** As a system administrator, I want the Dashboard to handle 100+ concurrent users and 1000+ jobs in the database with fast response times, so that the system remains performant under production load.

#### Acceptance Criteria

1. WHEN 100 concurrent users access the Dashboard, THE Dashboard SHALL maintain page load times under 2 seconds
2. WHEN the PostgreSQL_Database contains 1000+ jobs, THE Dashboard SHALL return job history queries within 1 second
3. WHEN a user loads a Data_Grid with 1000+ rows, THE Dashboard SHALL render the initial page within 1 second using pagination
4. THE Dashboard SHALL implement Redis caching for frequently accessed data (job status, user profiles)
5. THE Dashboard SHALL cache API responses for 30 seconds to reduce database load
6. WHEN the ML_Pipeline sends real-time updates, THE Dashboard SHALL handle 1000+ WebSocket connections simultaneously
7. THE Dashboard SHALL implement database connection pooling with a maximum of 50 connections

### Requirement 25: Security and Data Protection

**User Story:** As a system administrator, I want the Dashboard to implement security best practices including HTTPS, SQL injection prevention, XSS protection, and CSRF protection, so that manufacturing data remains secure.

#### Acceptance Criteria

1. THE Dashboard SHALL enforce HTTPS for all client-server communication
2. THE Flask_API SHALL use parameterized queries to prevent SQL injection attacks
3. THE Dashboard SHALL sanitize all user inputs to prevent cross-site scripting (XSS) attacks
4. THE Dashboard SHALL implement CSRF tokens for all state-changing operations
5. THE Dashboard SHALL hash user passwords using bcrypt with a cost factor of 12
6. THE Dashboard SHALL implement rate limiting on API endpoints (100 requests per minute per user)
7. THE Dashboard SHALL log all authentication attempts and security events to an audit log
8. THE Dashboard SHALL automatically log out users after 8 hours of inactivity

### Requirement 26: File Management and Storage

**User Story:** As a manufacturing engineer, I want the Dashboard to handle large CSV file uploads (up to 500MB) with progress indicators and secure storage, so that I can upload manufacturing data efficiently.

#### Acceptance Criteria

1. WHEN a user uploads a file, THE Dashboard SHALL display a progress bar showing upload percentage
2. WHEN a file upload is in progress, THE Dashboard SHALL allow the user to cancel the upload
3. THE Dashboard SHALL store uploaded files in a secure file system directory with restricted access permissions
4. THE Dashboard SHALL organize uploaded files by job ID in separate subdirectories
5. WHEN a job is deleted, THE Dashboard SHALL also delete all associated uploaded files
6. THE Dashboard SHALL implement virus scanning on all uploaded files before accepting them
7. WHEN a file upload fails due to network error, THE Dashboard SHALL allow the user to retry the upload

### Requirement 27: Background Job Processing

**User Story:** As a system administrator, I want the Dashboard to use Celery for asynchronous ML pipeline execution, so that long-running jobs don't block the web server.

#### Acceptance Criteria

1. WHEN a user submits a job for execution, THE Flask_API SHALL enqueue the job to Celery for background processing
2. THE Dashboard SHALL use Celery workers to execute ML_Pipeline jobs asynchronously
3. WHEN a Celery worker starts processing a job, THE Dashboard SHALL update job status to "Running"
4. WHEN a Celery worker completes a job, THE Dashboard SHALL update job status to "Complete" and store results in PostgreSQL_Database
5. IF a Celery worker fails to process a job, THEN THE Dashboard SHALL retry the job up to 3 times with exponential backoff
6. THE Dashboard SHALL monitor Celery worker health and display worker status in the admin panel
7. THE Dashboard SHALL configure Celery to use Redis as the message broker

### Requirement 28: Database Schema and Data Persistence

**User Story:** As a system administrator, I want the Dashboard to use a well-designed PostgreSQL schema with proper indexing and relationships, so that data is stored efficiently and queries are fast.

#### Acceptance Criteria

1. THE PostgreSQL_Database SHALL contain a "users" table with columns: id, username, email, password_hash, role, created_at, last_login
2. THE PostgreSQL_Database SHALL contain a "jobs" table with columns: id, user_id, job_name, analysis_type, status, created_at, started_at, completed_at, error_message
3. THE PostgreSQL_Database SHALL contain a "job_files" table with columns: id, job_id, file_name, file_path, file_size, uploaded_at
4. THE PostgreSQL_Database SHALL contain a "results" table with columns: id, job_id, factor_name, factor_type, ensemble_score, confidence, shap_score, xgboost_score, perm_score, mi_score, lasso_score, rank
5. THE PostgreSQL_Database SHALL contain a "job_status_history" table with columns: id, job_id, status, stage, sub_stage, progress_percent, timestamp
6. THE PostgreSQL_Database SHALL create indexes on frequently queried columns: jobs.user_id, jobs.status, results.job_id, results.ensemble_score
7. THE PostgreSQL_Database SHALL enforce foreign key constraints to maintain referential integrity

### Requirement 29: Real-Time Communication via WebSocket

**User Story:** As a manufacturing engineer, I want to see real-time updates of job progress without refreshing the page, so that I can monitor execution seamlessly.

#### Acceptance Criteria

1. WHEN a user views a running job, THE Dashboard SHALL establish a WebSocket connection to the Flask_API
2. WHEN the ML_Pipeline updates job status, THE Flask_API SHALL broadcast the update to all connected clients via WebSocket
3. WHEN a WebSocket message is received, THE Dashboard SHALL update the UI within 500 milliseconds
4. WHEN a WebSocket connection is lost, THE Dashboard SHALL attempt to reconnect automatically with exponential backoff
5. THE Dashboard SHALL display a connection status indicator showing "Connected" or "Reconnecting"
6. WHEN a user navigates away from a job page, THE Dashboard SHALL close the WebSocket connection to free resources
7. THE Flask_API SHALL support at least 1000 concurrent WebSocket connections

### Requirement 30: Contextual Help and User Guidance

**User Story:** As a manufacturing engineer, I want contextual help tooltips and expandable "What does this mean?" sections throughout the Dashboard, so that I can understand metrics and visualizations without external documentation.

#### Acceptance Criteria

1. WHEN a user hovers over any metric label, THE Dashboard SHALL display a tooltip with a brief explanation
2. WHEN a user clicks a "What does this mean?" link, THE Dashboard SHALL expand a section with detailed explanation and examples
3. THE Dashboard SHALL provide interpretation guidance for all ML model metrics (SHAP, XGBoost, Permutation, MI, LASSO)
4. THE Dashboard SHALL provide recommended actions based on analysis results (e.g., "Investigate this chamber", "Review this recipe")
5. THE Dashboard SHALL include links to detailed documentation for advanced topics
6. WHEN a user views the Root Cause Analysis page for the first time, THE Dashboard SHALL display an optional guided tour highlighting key sections
