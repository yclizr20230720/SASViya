# Implementation Plan: ML Analysis Web Dashboard

## Overview

This implementation plan breaks down the ML Analysis Web Dashboard into discrete, incremental coding tasks. The approach follows a bottom-up strategy: starting with database schema and core backend services, then building API endpoints, followed by frontend components, and finally integration and testing. Each task builds on previous work, ensuring no orphaned code.

The implementation uses React 18 + TypeScript + Vite for the frontend, Python Flask for the backend, PostgreSQL for the database, and Redis for caching and message brokering.

## Tasks

- [ ] 1. Set up project structure and development environment
  - Create backend directory structure: `backend/app/models/`, `backend/app/services/`, `backend/app/api/`, `backend/app/tasks/`
  - Create frontend directory structure: `frontend/src/components/`, `frontend/src/services/`, `frontend/src/stores/`, `frontend/src/types/`
  - Set up Python virtual environment and install dependencies: Flask, SQLAlchemy, Flask-SocketIO, Celery, Redis, Hypothesis, pytest
  - Set up Node.js project and install dependencies: React, TypeScript, Vite, TanStack Query, Zustand, Socket.IO Client, Recharts, AG-Grid, fast-check, vitest
  - Create Docker Compose file for PostgreSQL, Redis, and development services
  - Create `.env` files for configuration (database URL, Redis URL, JWT secret)
  - _Requirements: Infrastructure setup_


- [ ] 2. Implement database schema and SQLAlchemy models
  - [ ] 2.1 Create SQLAlchemy base configuration and database connection
    - Write `backend/app/database.py` with SQLAlchemy engine, session factory, and Base class
    - Configure connection pooling (max 50 connections)
    - _Requirements: 28.7_
  
  - [ ] 2.2 Implement User model with authentication fields
    - Write `backend/app/models/user.py` with User class
    - Include fields: id, username, email, password_hash, role, notification preferences, theme, timestamps
    - Add role constraint check (engineer, manager, admin)
    - Add indexes on username and email
    - _Requirements: 19.1, 19.2, 19.3, 28.1_
  
  - [ ] 2.3 Implement Job model with status tracking
    - Write `backend/app/models/job.py` with Job class
    - Include fields: id (UUID), user_id, job_name, description, analysis_type, status, stage info, progress, timestamps
    - Add status constraint check (pending, queued, running, complete, failed, deleted)
    - Add analysis_type constraint check (6 types)
    - Add unique constraint on (user_id, job_name) where deleted_at IS NULL
    - Add indexes on user_id, status, created_at
    - _Requirements: 1.4, 1.5, 3.2, 28.2_
  
  - [ ] 2.4 Implement JobFile, Result, JobStatusHistory, ModelMetric models
    - Write `backend/app/models/job_file.py` with JobFile class
    - Write `backend/app/models/result.py` with Result class (factor rankings)
    - Write `backend/app/models/job_status_history.py` with JobStatusHistory class
    - Write `backend/app/models/model_metric.py` with ModelMetric class
    - Add appropriate foreign keys, constraints, and indexes
    - _Requirements: 2.1, 5.3, 28.3, 28.4, 28.5, 28.6_
  
  - [ ] 2.5 Implement APIKey and AuditLog models
    - Write `backend/app/models/api_key.py` with APIKey class
    - Write `backend/app/models/audit_log.py` with AuditLog class
    - _Requirements: 22.6, 25.7, 28.7, 28.8_
  
  - [ ] 2.6 Create database migration scripts
    - Write Alembic migration to create all tables with constraints and indexes
    - Write seed script to create default admin user
    - _Requirements: 28.1-28.8_
  
  - [ ]* 2.7 Write property test for database foreign key integrity
    - **Property 37: Database Foreign Key Integrity**
    - **Validates: Requirements 28.7**

- [ ] 3. Implement core backend services
  - [ ] 3.1 Implement AuthService for authentication and session management
    - Write `backend/app/services/auth_service.py` with AuthService class
    - Implement `authenticate(username, password)` using bcrypt verification
    - Implement `create_session(user)` to generate JWT token with 8-hour expiration
    - Implement `validate_token(token)` to verify JWT and return User
    - Implement `check_permission(user, resource, action)` for RBAC
    - _Requirements: 19.1, 19.2, 19.3, 19.4_
  
  - [ ]* 3.2 Write property test for authentication token expiration
    - **Property 27: Authentication Token Expiration**
    - **Validates: Requirements 19.2**
  
  - [ ]* 3.3 Write property test for role-based access control
    - **Property 28: Role-Based Access Control**
    - **Validates: Requirements 19.3, 19.4, 19.5, 19.6, 19.7**
  
  - [ ] 3.4 Implement JobService for job lifecycle management
    - Write `backend/app/services/job_service.py` with JobService class
    - Implement `create_job(user_id, job_data)` with uniqueness validation
    - Implement `get_job(job_id, user)` with permission check
    - Implement `list_jobs(user, filters, page, page_size)` with filtering and pagination
    - Implement `update_job_status(job_id, status, stage_info)` with WebSocket emission
    - Implement `delete_job(job_id, user)` with file cleanup
    - _Requirements: 1.3, 1.4, 1.5, 3.2, 3.5, 6.1, 6.2, 6.3, 6.4, 6.6_
  
  - [ ]* 3.5 Write property test for job creation uniqueness
    - **Property 1: Job Creation Uniqueness**
    - **Validates: Requirements 1.3, 1.4, 1.5**
  
  - [ ]* 3.6 Write property test for job state machine transitions
    - **Property 4: Job State Machine Transitions**
    - **Validates: Requirements 1.4, 2.4, 3.1, 3.2, 3.4**
  
  - [ ]* 3.7 Write property test for job status persistence round-trip
    - **Property 6: Job Status Persistence Round-Trip**
    - **Validates: Requirements 3.5**

- [ ] 4. Implement file upload and validation services
  - [ ] 4.1 Define analysis type file requirements configuration
    - Write `backend/app/config/file_requirements.py` with FILE_REQUIREMENTS dict
    - Map each analysis type to its required files with patterns
    - Example: 'CP_VS_FDC_UCHART' → ['dtgb__detail.csv', 'dtgb__row.csv', 'dtrw1_0_all.csv', ...]
    - _Requirements: 1.2, 2.1_
  
  - [ ] 4.2 Implement FileUploadService for file handling
    - Write `backend/app/services/file_upload_service.py` with FileUploadService class
    - Implement `validate_file(file, job)` to check name pattern, size, format, headers
    - Implement `save_file(file, job_id)` to save to `/data/jobs/{job_id}/{filename}`
    - Implement `check_all_files_uploaded(job_id)` to verify completeness
    - Implement `trigger_job_execution(job_id)` to enqueue Celery task
    - Implement `scan_file_for_viruses(file_path)` using ClamAV (or mock for MVP)
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 26.1, 26.3, 26.4, 26.6_
  
  - [ ]* 4.3 Write property test for file upload validation
    - **Property 3: File Upload Validation**
    - **Validates: Requirements 2.2, 2.3, 2.5, 2.6**
  
  - [ ]* 4.4 Write property test for automatic job execution trigger
    - **Property 5: Automatic Job Execution Trigger**
    - **Validates: Requirements 2.4, 3.1**
  
  - [ ]* 4.5 Write property test for file cleanup on job deletion
    - **Property 35: File Cleanup on Job Deletion**
    - **Validates: Requirements 26.5**

- [ ] 5. Implement results and analysis services
  - [ ] 5.1 Implement ResultsService for retrieving and caching results
    - Write `backend/app/services/results_service.py` with ResultsService class
    - Implement `get_results(job_id)` with Redis caching (5-minute TTL)
    - Implement `get_factor_detail(job_id, factor_name)` for detailed analysis
    - Implement `compare_jobs(job_ids)` for multi-job comparison
    - Implement `export_results(job_id, format)` to generate CSV/XLSX
    - _Requirements: 7.1, 7.7, 7.8, 18.2, 18.3, 20.2, 20.3_
  
  - [ ]* 5.2 Write property test for ensemble score composition
    - **Property 23: Ensemble Score Composition**
    - **Validates: Requirements 8.2, 16.1**
  
  - [ ]* 5.3 Write property test for top N factor selection
    - **Property 24: Top N Factor Selection**
    - **Validates: Requirements 8.1, 17.2**
  
  - [ ]* 5.4 Write property test for export data consistency
    - **Property 21: Export Data Consistency**
    - **Validates: Requirements 7.7, 7.8**

- [ ] 6. Checkpoint - Ensure backend services tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 7. Implement Flask API endpoints - Authentication
  - [ ] 7.1 Create Flask app configuration and blueprints
    - Write `backend/app/app.py` with Flask app factory
    - Configure CORS, JSON serialization, error handlers
    - Register blueprints for auth, jobs, files, results, admin
    - _Requirements: 22.1-22.8_
  
  - [ ] 7.2 Implement authentication endpoints
    - Write `backend/app/api/auth.py` with auth blueprint
    - POST /api/v1/auth/login - authenticate and return JWT token
    - POST /api/v1/auth/logout - invalidate session
    - GET /api/v1/auth/me - get current user info
    - PUT /api/v1/auth/profile - update user preferences
    - Add JWT token validation middleware
    - _Requirements: 19.1, 19.2, 23.6_
  
  - [ ]* 7.3 Write unit tests for authentication endpoints
    - Test successful login, invalid credentials, token expiration
    - Test profile update with valid/invalid data
    - _Requirements: 19.1, 19.2_

- [ ] 8. Implement Flask API endpoints - Job Management
  - [ ] 8.1 Implement job management endpoints
    - Write `backend/app/api/jobs.py` with jobs blueprint
    - POST /api/v1/jobs - create new job with validation
    - GET /api/v1/jobs - list jobs with filtering, search, pagination
    - GET /api/v1/jobs/{job_id} - get job details with file status
    - DELETE /api/v1/jobs/{job_id} - soft delete job
    - Add permission checks for all endpoints
    - _Requirements: 1.3, 1.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [ ]* 8.2 Write property test for search filter correctness
    - **Property 13: Search Filter Correctness**
    - **Validates: Requirements 6.2**
  
  - [ ]* 8.3 Write property test for date range filter correctness
    - **Property 14: Date Range Filter Correctness**
    - **Validates: Requirements 6.3**
  
  - [ ]* 8.4 Write property test for pagination correctness
    - **Property 16: Pagination Correctness**
    - **Validates: Requirements 6.6, 7.10**

- [ ] 9. Implement Flask API endpoints - File Upload
  - [ ] 9.1 Implement file upload endpoints
    - Write `backend/app/api/files.py` with files blueprint
    - POST /api/v1/jobs/{job_id}/files - upload file with validation and progress
    - GET /api/v1/jobs/{job_id}/files - list uploaded files and requirements
    - Add multipart/form-data handling
    - Add file size limit check (500MB)
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 26.1_
  
  - [ ]* 9.2 Write property test for file upload progress tracking
    - **Property 34: File Upload Progress Tracking**
    - **Validates: Requirements 26.1**

- [ ] 10. Implement Flask API endpoints - Results and Analysis
  - [ ] 10.1 Implement results endpoints
    - Write `backend/app/api/results.py` with results blueprint
    - GET /api/v1/jobs/{job_id}/results - get ranking results with filtering, sorting, pagination
    - GET /api/v1/jobs/{job_id}/results/{factor_name} - get factor detail
    - PUT /api/v1/jobs/{job_id}/results/{factor_name}/mark - mark/unmark factor
    - GET /api/v1/jobs/{job_id}/results/export - export to CSV/XLSX
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 18.1, 18.2_
  
  - [ ]* 10.2 Write property test for data grid sorting
    - **Property 17: Data Grid Sorting**
    - **Validates: Requirements 7.2, 7.3**
  
  - [ ]* 10.3 Write property test for factor type filter
    - **Property 19: Factor Type Filter**
    - **Validates: Requirements 7.5**
  
  - [ ]* 10.4 Write property test for score range filter
    - **Property 20: Score Range Filter**
    - **Validates: Requirements 7.6**

- [ ] 11. Implement Flask API endpoints - Dashboard and Analysis
  - [ ] 11.1 Implement dashboard endpoints
    - Write `backend/app/api/dashboard.py` with dashboard blueprint
    - GET /api/v1/jobs/{job_id}/dashboard - get executive dashboard data
    - GET /api/v1/jobs/{job_id}/root-cause-analysis - get ML model analysis data
    - POST /api/v1/jobs/compare - compare multiple jobs
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 11.1-16.7, 17.1-17.6, 20.1-20.7_
  
  - [ ]* 11.2 Write property test for severity indicator thresholds
    - **Property 25: Severity Indicator Thresholds**
    - **Validates: Requirements 17.3**
  
  - [ ]* 11.3 Write property test for factor grouping by type
    - **Property 26: Factor Grouping by Type**
    - **Validates: Requirements 17.1**
  
  - [ ]* 11.4 Write property test for job comparison common factors
    - **Property 29: Job Comparison Common Factors**
    - **Validates: Requirements 20.2, 20.3, 20.4**

- [ ] 12. Implement Flask API endpoints - Admin
  - [ ] 12.1 Implement admin endpoints
    - Write `backend/app/api/admin.py` with admin blueprint
    - GET /api/v1/admin/users - list users with search and pagination
    - POST /api/v1/admin/users - create new user
    - PUT /api/v1/admin/users/{user_id} - update user
    - DELETE /api/v1/admin/users/{user_id} - delete user
    - GET /api/v1/admin/system-health - get system health metrics
    - Add admin-only permission checks
    - _Requirements: 19.6, 19.7_
  
  - [ ]* 12.2 Write unit tests for admin endpoints
    - Test user CRUD operations
    - Test permission enforcement (non-admin should be denied)
    - _Requirements: 19.6, 19.7_

- [ ] 13. Implement WebSocket server with Flask-SocketIO
  - [ ] 13.1 Set up Flask-SocketIO server
    - Write `backend/app/websocket.py` with SocketIO configuration
    - Configure Redis as message broker for multi-worker support
    - Implement connection handler with JWT authentication
    - _Requirements: 29.1, 29.7_
  
  - [ ] 13.2 Implement WebSocket event handlers
    - Implement `on_connect` with token validation
    - Implement `on_subscribe_job` to join job-specific room
    - Implement `on_unsubscribe_job` to leave room
    - Implement `on_disconnect` cleanup
    - Implement `broadcast_job_update` helper for Celery tasks
    - _Requirements: 29.2, 29.3, 29.4, 29.5, 29.6_
  
  - [ ]* 13.3 Write property test for WebSocket room subscription
    - **Property 38: WebSocket Room Subscription**
    - **Validates: Requirements 29.2, 29.3**

- [ ] 14. Checkpoint - Ensure backend API tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 15. Implement Celery background tasks
  - [ ] 15.1 Set up Celery configuration
    - Write `backend/app/celery_app.py` with Celery app configuration
    - Configure Redis as broker and result backend
    - Configure task routing and retry policies
    - _Requirements: 27.1, 27.7_
  
  - [ ] 15.2 Implement ML pipeline execution task
    - Write `backend/app/tasks/ml_pipeline_task.py` with execute_ml_pipeline task
    - Integrate with ML-Stats-Migration pipeline
    - Update job status at each stage (input, pipeline, output)
    - Emit WebSocket updates for real-time monitoring
    - Parse pipeline output and store results in database
    - Handle errors with retry logic (3 retries, exponential backoff)
    - Send completion/failure notifications
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 27.1, 27.2, 27.3, 27.4, 27.5_
  
  - [ ]* 15.3 Write property test for Celery task retry with exponential backoff
    - **Property 36: Celery Task Retry with Exponential Backoff**
    - **Validates: Requirements 27.5**
  
  - [ ] 15.4 Implement periodic cleanup task
    - Write `backend/app/tasks/cleanup_task.py` with cleanup_old_files task
    - Schedule to run daily at 2 AM
    - Delete files and records for jobs deleted > 30 days ago
    - _Requirements: 26.5_
  
  - [ ] 15.5 Implement cache warming task
    - Write `backend/app/tasks/cache_task.py` with warm_results_cache task
    - Schedule to run every 10 minutes
    - Pre-cache results for recently completed jobs
    - _Requirements: 24.5_

- [ ] 16. Implement notification service
  - [ ] 16.1 Implement NotificationService for email and SMS
    - Write `backend/app/services/notification_service.py` with NotificationService class
    - Implement `send_job_complete_notification(job, user)`
    - Implement `send_job_failed_notification(job, user, error)`
    - Implement `send_critical_root_cause_alert(job, user, killer_factors)`
    - Implement `check_rate_limit(user_id)` using Redis counter
    - Integrate with email service (SMTP) and SMS service (Twilio or mock)
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6_
  
  - [ ]* 16.2 Write property test for notification rate limiting
    - **Property 30: Notification Rate Limiting**
    - **Validates: Requirements 21.6**

- [ ] 17. Implement security middleware and input sanitization
  - [ ] 17.1 Implement security middleware
    - Write `backend/app/middleware/security.py` with security middleware
    - Implement CSRF token validation for state-changing operations
    - Implement rate limiting (100 requests/minute per user)
    - Implement audit logging for all requests
    - _Requirements: 25.4, 25.6, 25.7_
  
  - [ ] 17.2 Implement input sanitization utilities
    - Write `backend/app/utils/sanitization.py` with sanitization functions
    - Implement `sanitize_html(text)` to escape/remove HTML tags
    - Implement `sanitize_sql(text)` (note: SQLAlchemy handles this, but add validation)
    - _Requirements: 25.2, 25.3_
  
  - [ ]* 17.3 Write property test for input sanitization (XSS prevention)
    - **Property 32: Input Sanitization for XSS Prevention**
    - **Validates: Requirements 25.3**
  
  - [ ]* 17.4 Write property test for SQL injection prevention
    - **Property 33: SQL Injection Prevention**
    - **Validates: Requirements 25.2**

- [ ] 18. Set up frontend project structure and routing
  - [ ] 18.1 Create React app with Vite and TypeScript
    - Initialize Vite project with React and TypeScript template
    - Configure TypeScript with strict mode
    - Set up ESLint and Prettier
    - _Requirements: Frontend infrastructure_
  
  - [ ] 18.2 Set up routing with React Router
    - Write `frontend/src/App.tsx` with router configuration
    - Define routes: /login, /dashboard, /jobs, /jobs/:id, /jobs/:id/results, /jobs/:id/analysis, /admin
    - Implement protected route wrapper for authentication
    - _Requirements: Navigation_
  
  - [ ] 18.3 Set up global state management with Zustand
    - Write `frontend/src/stores/authStore.ts` for authentication state
    - Write `frontend/src/stores/uiStore.ts` for UI preferences (theme, sidebar)
    - _Requirements: State management_
  
  - [ ] 18.4 Set up API client with Axios and TanStack Query
    - Write `frontend/src/services/apiClient.ts` with Axios instance
    - Configure base URL, interceptors for JWT token, error handling
    - Write `frontend/src/services/queryClient.ts` with TanStack Query configuration
    - _Requirements: API integration_

- [ ] 19. Implement frontend authentication components
  - [ ] 19.1 Implement LoginPage component
    - Write `frontend/src/pages/LoginPage.tsx`
    - Create login form with username and password fields
    - Handle form submission and JWT token storage
    - Display error messages for invalid credentials
    - Redirect to dashboard on successful login
    - _Requirements: 19.1, 19.2_
  
  - [ ] 19.2 Implement authentication hooks and utilities
    - Write `frontend/src/hooks/useAuth.ts` with login, logout, and token validation
    - Write `frontend/src/utils/auth.ts` with token storage and retrieval
    - _Requirements: 19.1, 19.2_
  
  - [ ]* 19.3 Write unit tests for authentication components
    - Test login form validation
    - Test successful login flow
    - Test error handling
    - _Requirements: 19.1, 19.2_

- [ ] 20. Implement frontend job creation and management components
  - [ ] 20.1 Implement JobCreationForm component
    - Write `frontend/src/components/JobCreationForm.tsx`
    - Create form with job name, description, analysis type dropdown, notification checkboxes
    - Implement form validation (required fields, max lengths)
    - Handle form submission and navigate to job detail page
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [ ]* 20.2 Write property test for analysis type file requirements mapping
    - **Property 2: Analysis Type File Requirements Mapping**
    - **Validates: Requirements 1.2, 2.1**
  
  - [ ] 20.3 Implement JobListPage component
    - Write `frontend/src/pages/JobListPage.tsx`
    - Display job history table with columns: name, type, status, created date
    - Implement search input, status filter dropdown, date range picker
    - Implement pagination controls
    - Handle row click to navigate to job detail
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [ ] 20.4 Implement JobDetailPage component
    - Write `frontend/src/pages/JobDetailPage.tsx`
    - Display job summary (name, type, status, timestamps)
    - Display file upload section (if pending)
    - Display pipeline monitor (if running)
    - Display results link (if complete)
    - _Requirements: 1.4, 2.1, 4.1_

- [ ] 21. Implement frontend file upload components
  - [ ] 21.1 Implement FileUploadManager component
    - Write `frontend/src/components/FileUploadManager.tsx`
    - Display checklist of required files with upload status icons
    - Implement drag-and-drop upload zone
    - Display progress bars for active uploads
    - Display validation error messages
    - Show auto-start indicator when all files ready
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 26.1, 26.2_
  
  - [ ]* 21.2 Write unit tests for file upload components
    - Test file validation (size, format, name pattern)
    - Test upload progress tracking
    - Test error handling
    - _Requirements: 2.2, 2.3, 2.5, 2.6, 26.1_

- [ ] 22. Checkpoint - Ensure frontend basic components tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 23. Implement frontend real-time monitoring components
  - [ ] 23.1 Set up Socket.IO client
    - Write `frontend/src/services/socketClient.ts` with Socket.IO client configuration
    - Implement connection with JWT token authentication
    - Implement auto-reconnection with exponential backoff
    - Implement event handlers for job_update, model_update, job_complete, job_failed
    - _Requirements: 29.1, 29.4, 29.5_
  
  - [ ] 23.2 Implement PipelineMonitor component
    - Write `frontend/src/components/PipelineMonitor.tsx`
    - Display pipeline stage diagram with current stage highlighted
    - Display progress bar with percentage
    - Display elapsed time and estimated remaining time
    - Display model status cards (5 cards for SHAP, XGBoost, Permutation, MI, LASSO)
    - Subscribe to WebSocket updates for real-time status
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ]* 23.3 Write property test for pipeline stage display correctness
    - **Property 7: Pipeline Stage Display Correctness**
    - **Validates: Requirements 4.1, 4.3, 4.4, 4.5**
  
  - [ ]* 23.4 Write property test for progress percentage bounds
    - **Property 8: Progress Percentage Bounds**
    - **Validates: Requirements 4.6**
  
  - [ ]* 23.5 Write property test for model status completeness
    - **Property 10: Model Status Completeness**
    - **Validates: Requirements 5.1**

- [ ] 24. Implement frontend data grid component
  - [ ] 24.1 Implement ResultsDataGrid component with AG-Grid
    - Write `frontend/src/components/ResultsDataGrid.tsx`
    - Configure AG-Grid with columns: rank, factor name, factor type, ensemble score, confidence, model scores, sample size, effect size
    - Implement column sorting (multi-column)
    - Implement full-text search filter
    - Implement factor type filter dropdown
    - Implement score range slider filter
    - Implement mark/unmark toggle with persistence
    - Implement export to CSV/Excel buttons
    - Implement pagination (100 rows per page)
    - Implement column show/hide menu
    - Implement column reordering via drag-drop with persistence
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12_
  
  - [ ]* 24.2 Write property test for data grid text search
    - **Property 18: Data Grid Text Search**
    - **Validates: Requirements 7.4**
  
  - [ ]* 24.3 Write property test for user preference persistence round-trip
    - **Property 22: User Preference Persistence Round-Trip**
    - **Validates: Requirements 7.9, 7.11, 7.12**

- [ ] 25. Implement frontend visualization components
  - [ ] 25.1 Implement EnsembleBarChart component with Recharts
    - Write `frontend/src/components/EnsembleBarChart.tsx`
    - Render stacked bar chart with 5 segments per bar (SHAP, XGBoost, Perm, MI, LASSO)
    - Use consistent color coding: SHAP=blue, XGBoost=green, Perm=orange, MI=purple, LASSO=red
    - Implement hover tooltips showing model name and score
    - Implement click handler for drill-down to factor detail
    - Implement export to PNG/SVG button
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  
  - [ ] 25.2 Implement advanced chart components
    - Write `frontend/src/components/TrendChart.tsx` for factor importance over time
    - Write `frontend/src/components/BoxPlotChart.tsx` for score distribution
    - Write `frontend/src/components/ScatterPlotChart.tsx` for model correlation
    - Write `frontend/src/components/HeatmapChart.tsx` for factor importance matrix
    - Write `frontend/src/components/WaterfallChart.tsx` for contribution breakdown
    - Write `frontend/src/components/RadarChart.tsx` for multi-dimensional comparison
    - All charts should support zoom, pan, hover tooltips, and export
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_
  
  - [ ]* 25.3 Write unit tests for chart components
    - Test chart rendering with sample data
    - Test interactive features (hover, click, zoom)
    - Test export functionality
    - _Requirements: 8.1-9.9_

- [ ] 26. Implement frontend dashboard components
  - [ ] 26.1 Implement ExecutiveDashboard component
    - Write `frontend/src/components/ExecutiveDashboard.tsx`
    - Render JobSummaryCard with job details
    - Render TopKillersCard with top 5 factors and severity indicators
    - Render ModelPerformanceCard with agreement score and consensus pie chart
    - Render DataQualityCard with completeness, missing %, outliers
    - Render HistoricalComparisonCard with trend indicators (if historical data available)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
  
  - [ ]* 26.2 Write unit tests for dashboard cards
    - Test each card renders with correct data
    - Test "View Details" and "Generate Report" buttons
    - _Requirements: 10.1-10.7_

- [ ] 27. Implement frontend root cause analysis page
  - [ ] 27.1 Implement RootCauseAnalysis component
    - Write `frontend/src/pages/RootCauseAnalysisPage.tsx`
    - Render explanation sections for each ML model (SHAP, XGBoost, Permutation, MI, LASSO, Ensemble)
    - Each section includes: metric definition, interpretation guidance, visualizations, key insight
    - Render SHAP section with waterfall, force plot, dependence plot
    - Render XGBoost section with Gain/Weight/Coverage charts, tree visualization
    - Render Permutation section with bar chart + error bars
    - Render MI section with sorted bars, interaction heatmap
    - Render LASSO section with regularization path, coefficient bars
    - Render Ensemble section with composite chart, confidence gauge, agreement score
    - Render Factor Type Breakdown with grouping and severity indicators
    - Render Killer Factor Summary with top 10 factors
    - Render Actionable Recommendations list
    - _Requirements: 11.1-16.7, 17.1-17.6_
  
  - [ ]* 27.2 Write unit tests for root cause analysis sections
    - Test each ML model section renders with correct explanations
    - Test visualizations display correct data
    - _Requirements: 11.1-16.7_

- [ ] 28. Implement frontend factor detail view
  - [ ] 28.1 Implement FactorDetailView component
    - Write `frontend/src/pages/FactorDetailPage.tsx`
    - Display model score comparison table (all 5 models side-by-side)
    - Display historical trend line chart
    - Display wafer/lot breakdown table
    - Display related factors list with correlation
    - Implement "Export Raw Data" button (CSV)
    - Implement "Export Report" button (PDF/PPTX)
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7_
  
  - [ ]* 28.2 Write unit tests for factor detail view
    - Test data display correctness
    - Test export functionality
    - _Requirements: 18.1-18.7_

- [ ] 29. Implement frontend job comparison feature
  - [ ] 29.1 Implement JobComparisonPage component
    - Write `frontend/src/pages/JobComparisonPage.tsx`
    - Allow user to select multiple jobs from job history
    - Display jobs in columns for side-by-side comparison
    - Display top 10 factors from each job aligned by factor name
    - Highlight common factors across jobs
    - Display trend chart showing score changes
    - Display difference summary (factors that increased/decreased)
    - Implement export comparison button (CSV/PDF)
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7_
  
  - [ ]* 29.2 Write unit tests for job comparison
    - Test job selection and comparison display
    - Test common factor identification
    - _Requirements: 20.1-20.7_

- [ ] 30. Implement frontend admin panel
  - [ ] 30.1 Implement AdminPanel component
    - Write `frontend/src/pages/AdminPage.tsx`
    - Display user management table with search and pagination
    - Implement "Create User" button and modal form
    - Implement "Edit User" button and modal form
    - Implement "Delete User" button with confirmation
    - Display system health metrics (Celery workers, database, Redis, disk)
    - Add admin-only route protection
    - _Requirements: 19.6, 19.7_
  
  - [ ]* 30.2 Write unit tests for admin panel
    - Test user CRUD operations
    - Test permission enforcement
    - _Requirements: 19.6, 19.7_

- [ ] 31. Checkpoint - Ensure frontend components tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 32. Implement responsive design and theming
  - [ ] 32.1 Implement theme system with CSS variables
    - Write `frontend/src/styles/themes.css` with light and dark theme variables
    - Define color palette, typography, spacing for both themes
    - _Requirements: 23.4, 23.5, 23.6, 23.7_
  
  - [ ] 32.2 Implement responsive layout components
    - Write `frontend/src/components/Layout.tsx` with responsive sidebar and header
    - Implement breakpoints for desktop (>1024px), tablet (768-1024px), mobile (<768px)
    - Implement hamburger menu for mobile navigation
    - _Requirements: 23.1, 23.2, 23.3_
  
  - [ ] 32.3 Implement theme toggle component
    - Write `frontend/src/components/ThemeToggle.tsx`
    - Allow user to switch between light and dark themes
    - Persist theme preference to backend
    - _Requirements: 23.4, 23.5, 23.6_

- [ ] 33. Implement contextual help and user guidance
  - [ ] 33.1 Implement tooltip and help components
    - Write `frontend/src/components/Tooltip.tsx` for hover tooltips
    - Write `frontend/src/components/HelpSection.tsx` for expandable "What does this mean?" sections
    - _Requirements: 30.1, 30.2_
  
  - [ ] 33.2 Add contextual help to all metrics and visualizations
    - Add tooltips to all metric labels (ensemble score, confidence, SHAP, etc.)
    - Add help sections to all ML model explanations
    - Add interpretation guidance for all charts
    - Add recommended actions based on results
    - _Requirements: 30.1, 30.2, 30.3, 30.4, 30.5_
  
  - [ ] 33.3 Implement guided tour for first-time users
    - Write `frontend/src/components/GuidedTour.tsx` using a tour library (e.g., react-joyride)
    - Create tour steps for Root Cause Analysis page highlighting key sections
    - Show tour on first visit, allow user to skip or replay
    - _Requirements: 30.6_

- [ ] 34. Implement API rate limiting and error handling
  - [ ] 34.1 Implement rate limiting middleware in backend
    - Write `backend/app/middleware/rate_limiter.py` using Flask-Limiter
    - Configure 100 requests per minute per user
    - Return 429 Too Many Requests with retry_after header
    - _Requirements: 25.6_
  
  - [ ] 34.2 Implement global error handler in frontend
    - Write `frontend/src/utils/errorHandler.ts` with error handling utilities
    - Handle different error types (400, 401, 403, 404, 409, 429, 500)
    - Display user-friendly error messages
    - Implement retry logic for transient errors
    - _Requirements: Error handling for all requirements_
  
  - [ ]* 34.3 Write property test for API endpoint response codes
    - **Property 31: API Endpoint Response Codes**
    - **Validates: Requirements 22.7**

- [ ] 35. Implement integration with ML-Stats-Migration pipeline
  - [ ] 35.1 Create ML pipeline adapter
    - Write `backend/app/adapters/ml_pipeline_adapter.py` with MLPipelineAdapter class
    - Implement `execute(job_id, analysis_type, input_files, callback)` method
    - Parse pipeline output (CSV and JSON files) into Result and ModelMetric objects
    - Handle pipeline errors and map to job failure reasons
    - _Requirements: 3.3, 4.1-5.5_
  
  - [ ] 35.2 Implement pipeline progress callback
    - Implement callback function to receive pipeline progress updates
    - Update job status and emit WebSocket events
    - Track stage transitions (input → pipeline → output)
    - Track model execution status (queued → training → complete)
    - _Requirements: 4.1-5.5_
  
  - [ ]* 35.3 Write integration tests for ML pipeline adapter
    - Test successful pipeline execution
    - Test pipeline failure handling
    - Test progress callback updates
    - _Requirements: 3.3, 4.1-5.5_

- [ ] 36. Implement caching strategy with Redis
  - [ ] 36.1 Implement Redis cache service
    - Write `backend/app/services/cache_service.py` with CacheService class
    - Implement `get(key)`, `set(key, value, ttl)`, `delete(key)`, `clear_pattern(pattern)` methods
    - Use Redis for caching results (5-minute TTL)
    - Use Redis for rate limiting counters (1-hour TTL)
    - Use Redis for session storage
    - _Requirements: 24.4, 24.5, 24.6_
  
  - [ ]* 36.2 Write unit tests for cache service
    - Test cache hit/miss scenarios
    - Test TTL expiration
    - Test cache invalidation
    - _Requirements: 24.4, 24.5_

- [ ] 37. Implement logging and monitoring
  - [ ] 37.1 Set up structured logging
    - Write `backend/app/utils/logger.py` with logging configuration
    - Configure log levels (DEBUG, INFO, WARNING, ERROR)
    - Configure log format (JSON for production, human-readable for development)
    - Log all API requests, errors, and security events
    - _Requirements: 25.7, 25.8_
  
  - [ ] 37.2 Implement audit logging
    - Write `backend/app/services/audit_service.py` with AuditService class
    - Log all authentication attempts
    - Log all authorization failures
    - Log all data modifications (create, update, delete)
    - Log all file uploads/downloads
    - _Requirements: 25.7_
  
  - [ ] 37.3 Set up monitoring and alerting (optional for MVP)
    - Configure Prometheus metrics export
    - Set up Grafana dashboards for system health
    - Configure alerts for error rate, response time, resource usage
    - _Requirements: 24.1-24.7_

- [ ] 38. Implement database connection pooling and optimization
  - [ ] 38.1 Configure SQLAlchemy connection pooling
    - Update `backend/app/database.py` with connection pool configuration
    - Set pool size to 50 connections
    - Set pool timeout and recycle settings
    - _Requirements: 24.7_
  
  - [ ] 38.2 Add database query optimization
    - Add eager loading for relationships to avoid N+1 queries
    - Add database query logging in development mode
    - Review and optimize slow queries
    - _Requirements: 24.2_

- [ ] 39. Implement security hardening
  - [ ] 39.1 Configure HTTPS and security headers
    - Update Flask app to enforce HTTPS in production
    - Add security headers: Content-Security-Policy, X-Frame-Options, X-Content-Type-Options
    - _Requirements: 25.1_
  
  - [ ] 39.2 Implement password hashing with bcrypt
    - Update AuthService to use bcrypt with cost factor 12
    - Implement password strength validation
    - _Requirements: 25.5_
  
  - [ ] 39.3 Implement session timeout
    - Configure JWT token expiration (8 hours)
    - Implement automatic logout on inactivity
    - _Requirements: 25.8_

- [ ] 40. Write end-to-end integration tests
  - [ ]* 40.1 Write integration test for complete job creation and execution flow
    - Test: Create job → Upload files → Automatic execution → Monitor progress → View results
    - Verify all state transitions and data persistence
    - _Requirements: 1.1-5.5_
  
  - [ ]* 40.2 Write integration test for authentication and authorization flow
    - Test: Login → Access protected resources → Permission checks → Logout
    - Test different user roles (engineer, manager, admin)
    - _Requirements: 19.1-19.7_
  
  - [ ]* 40.3 Write integration test for real-time WebSocket communication
    - Test: Connect → Subscribe to job → Receive updates → Disconnect
    - Verify updates are received in real-time
    - _Requirements: 29.1-29.7_
  
  - [ ]* 40.4 Write integration test for file upload and validation
    - Test: Upload valid files → Upload invalid files → Verify validation errors
    - Test file size limits and format validation
    - _Requirements: 2.1-2.6, 26.1-26.7_

- [ ] 41. Create deployment configuration
  - [ ] 41.1 Create Docker images for backend and frontend
    - Write `backend/Dockerfile` for Flask app with Gunicorn
    - Write `frontend/Dockerfile` for React app with Nginx
    - _Requirements: Deployment_
  
  - [ ] 41.2 Create Docker Compose for full stack
    - Write `docker-compose.yml` with services: frontend, backend, PostgreSQL, Redis, Celery worker
    - Configure networking and volumes
    - _Requirements: Deployment_
  
  - [ ] 41.3 Create environment configuration templates
    - Write `.env.example` with all required environment variables
    - Document configuration options
    - _Requirements: Deployment_

- [ ] 42. Write API documentation
  - [ ] 42.1 Generate OpenAPI/Swagger documentation
    - Use Flask-RESTX or similar to generate API docs
    - Document all endpoints with request/response schemas
    - Add authentication requirements
    - _Requirements: 22.8_
  
  - [ ] 42.2 Create API usage examples
    - Write example API calls for common workflows
    - Document error responses and handling
    - _Requirements: 22.1-22.8_

- [ ] 43. Final checkpoint - Run all tests and verify system
  - Run all unit tests (backend and frontend)
  - Run all property tests (100+ iterations each)
  - Run all integration tests
  - Verify all 38 correctness properties are tested
  - Verify test coverage meets goals (80%+ overall)
  - Start full stack with Docker Compose
  - Perform manual smoke testing of key workflows
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test-related sub-tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples, edge cases, and integration points
- The implementation follows a bottom-up approach: database → services → API → frontend → integration
- All 38 correctness properties from the design document have corresponding property test tasks
