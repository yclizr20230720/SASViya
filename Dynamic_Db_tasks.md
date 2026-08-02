# Implementation Plan

- [x] 1. Scaffold project structure
  - Create `backend/` Flask app (app factory, config module for env-based settings, blueprint registration stub) and `frontend/` Vite+React+TypeScript app (routing shell, base layout, API client wrapper)
  - Set up `backend/requirements.txt` (Flask, SQLAlchemy, psycopg2/pyodbc/oracledb as needed, pandas, rapidfuzz, python-dotenv, pytest) and `frontend/package.json` deps (react-router-dom, @tanstack/react-query, vitest, @testing-library/react)
  - Add `.env.example` for both apps documenting required config (metadata DB URL, JWT secret, default timeout/row-limit settings)
  - _Requirements: 9.1_

- [x] 2. Metadata database models and migrations
  - [x] 2.1 Define SQLAlchemy models for `users`, `source_permissions`, `data_sources`, `schema_cache_tables`, `schema_cache_columns`, `dictionary_entries`, `dataset_drafts`, `query_audit_log`
    - _Requirements: 1.1, 2.1, 2.2, 6.5, 9.2, 9.4_
  - [x] 2.2 Set up Alembic migrations and initial migration for all tables
    - _Requirements: 1.1, 2.1_
  - [x] 2.3 Write model-level unit tests (constraints, relationships, JSON draft field round-trip)
    - _Requirements: 3.5, 5.5_

- [x] 3. Authentication and authorization
  - [x] 3.1 Implement password hashing, `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` with JWT/session issuance
    - _Requirements: 9.1_
  - [x] 3.2 Implement `before_request` auth guard and a `require_source_permission(can_browse|can_execute)` decorator
    - _Requirements: 9.1, 9.2, 9.3_
  - [x] 3.3 Write tests: login success/failure, unauthenticated access rejected, permission decorator blocks unauthorized source access
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 4. Data source registration (admin)
  - [x] 4.1 Implement `data_sources` CRUD endpoints (`/api/admin/sources`) storing dialect/host/port/db/credential-ref/timeout/max_rows/max_concurrent/full_scan_threshold
    - _Requirements: 1.1, 8.1, 8.2, 8.3_
  - [x] 4.2 Implement `data_source_schemas` CRUD (`/api/admin/sources/:id/schemas`) so one source/connection can register multiple browsable schemas (e.g. F12PEDA → VEDA, SIMM, MESMGR, SISPC, WATCH, FDC, PTS, OUTGO)
    - _Requirements: 1.1_
  - [x] 4.3 Implement credential resolution helper that reads the actual DB credential from env/secrets manager by ref at connection time only (never persisted/logged)
    - _Requirements: 8.4, 8.5_
  - [x] 4.4 Seed initial sources: F12PEDA (Oracle, PEDA-scan.vsmc.com:1587, service F12PEDA, 8 schemas above), F12PMES (Oracle, PMES-scan.vsmc.com:1581, service F12PMES, schemas SIMM/SISPC/WATCH), f12edagp (GreenPlum, 10.92.67.25:5432, schema veda) — credentials via env var refs, not the plaintext values found in legacy scripts
    - _Requirements: 1.1, 8.4, 8.5_
  - [x] 4.5 Build minimal `/admin/sources` frontend page to list/create/edit sources, their registered schemas, and their limits
    - _Requirements: 1.1, 8.1, 8.2, 8.3_

- [x] 5. Schema introspection and caching
  - [x] 5.1 Implement introspection service using SQLAlchemy `Inspector` to read tables, columns, data types, indexes/keys, and approximate row counts per dialect, iterating over each registered schema for a source (not just one default schema)
    - _Requirements: 1.2, 1.3, 6.2_
  - [x] 5.2 Implement `POST /api/sources/:id/refresh-metadata` to run introspection per registered schema and upsert into `schema_cache_tables`/`schema_cache_columns` keyed by `(source_id, schema_name, table_name)`
    - _Requirements: 1.5_
  - [x] 5.3 Implement `GET /api/sources`, `GET /api/sources/:id/schemas`, `GET /api/sources/:id/schemas/:schema/tables`, `GET /api/sources/:id/schemas/:schema/tables/:table/columns` reading only from cache, scoped by user's schema-level `can_browse` permission
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 9.2_
  - [x] 5.4 Write tests: introspection parses a sample multi-schema source (e.g. F12PEDA with 8 schemas) correctly into separate cache rows per schema; browse endpoints return per-source error without failing whole response when one source is unreachable; a user permitted on one schema but not another under the same source sees only the permitted schema
    - _Requirements: 1.4, 9.2_

- [x] 6. Data dictionary
  - [x] 6.1 Implement `dictionary_entries` upsert logic and `POST /api/dictionary/import` accepting Excel/CSV with validation (required columns: source, table, column?, description)
    - _Requirements: 2.4_
  - [x] 6.2 Merge dictionary descriptions into the schema browse responses from task 5.3
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 6.3 Build `/admin/dictionary` frontend page for upload + preview of parsed entries before commit
    - _Requirements: 2.4_
  - [x] 6.4 Write tests: import with missing/invalid columns rejected without partial writes; merged descriptions appear correctly, absence doesn't error
    - _Requirements: 2.3, 2.4_

- [x] 7. Frontend: Database & Schema Browser page
  - Build `/sources` page: source list → table list → column list with descriptions, per-source error banners, "refresh metadata" action for admins
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

- [x] 8. Dataset draft service
  - [x] 8.1 Implement `POST /api/datasets`, `GET /api/datasets/:id`, `PATCH /api/datasets/:id` operating on `definition_json`
    - _Requirements: 3.5, 5.5_
  - [x] 8.2 Write tests: draft persists partial updates across multiple PATCH calls, ownership enforced (user A cannot read/edit user B's draft)
    - _Requirements: 3.5, 9.2_

- [x] 9. Frontend: Column Selection page
  - Build `/datasets/:draftId/columns`: multi-table, multi-database column checkboxes, persists selections to draft via PATCH, retains state across tables, "specify join" UI when 2+ tables selected
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 10. Requirement description & file import
  - [x] 10.1 Implement manual requirement item CRUD within a draft (`fieldName`, `description`, `sourceHint`)
    - _Requirements: 4.1_
  - [x] 10.2 Implement file parsers for `.xlsx`/`.csv`/`.json` into normalized requirement items, with strict validation and a `/requirement/import` preview endpoint plus a separate `/commit` endpoint
    - _Requirements: 4.2, 4.3, 4.4_
  - [x] 10.3 Implement Source Matcher: rank candidate table/column matches per requirement item using name + dictionary description similarity (rapidfuzz)
    - _Requirements: 4.5, 6.1_
  - [x] 10.4 Write tests: malformed file rejected with no partial commit; matcher returns expected top candidate on sample fixtures
    - _Requirements: 4.4, 4.5_
  - [x] 10.5 Build `/datasets/:draftId/requirement` frontend page: manual entry table + file upload with parsed preview + matched-candidate review/confirm UI
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

- [x] 11. Filter criteria
  - [x] 11.1 Implement criteria CRUD within a draft: operator set per data type, scope grouping, type-compatibility validation
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [x] 11.2 Write tests: type-incompatible value rejected; scope correctly applied to all criteria on the same table
    - _Requirements: 5.3, 5.4_
  - [x] 11.3 Build `/datasets/:draftId/criteria` frontend page: per-table condition builder with type-aware operator/value inputs and scope grouping
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 12. Query Builder
  - [x] 12.1 Implement SQLAlchemy Core-based query construction from a draft's selections/joins/criteria, with foreign-key-based join inference and a "join clarification needed" response when ambiguous
    - _Requirements: 3.4, 6.1_
  - [x] 12.2 Implement resolution path for requirement-item-only drafts: use matcher-confirmed columns as the selection input to the same builder
    - _Requirements: 4.5, 6.1_
  - [x] 12.3 Write tests: generated SQL/params correct for single-table, multi-table joined, and matcher-resolved drafts; injection attempt via criteria value stays parameterized
    - _Requirements: 6.1, 8.4_

- [x] 13. Performance Evaluator
  - [x] 13.1 Implement static check: filter/join columns vs. cached index/key metadata
    - _Requirements: 6.2_
  - [x] 13.2 Implement row-count threshold check per source's `full_scan_threshold`
    - _Requirements: 6.2, 6.3_
  - [x] 13.3 Implement best-effort `EXPLAIN`-equivalent plan check per dialect with its own short timeout, parsing estimated cost/scan type
    - _Requirements: 6.2_
  - [x] 13.4 Combine checks into a single verdict (`pass`/`warn`/`block`) with findings list; wire into `POST /api/datasets/:id/generate-sql`; persist every evaluation to `query_audit_log`
    - _Requirements: 6.2, 6.3, 6.4, 6.5_
  - [x] 13.5 Write tests: missing-key filter triggers block on large table, indexed filter passes, warn requires acknowledgment flag before proceeding
    - _Requirements: 6.2, 6.3, 6.4_

- [x] 14. Frontend: SQL Review page
  - Build `/datasets/:draftId/review`: show generated SQL, verdict banner (pass/warn/block) with findings, block Preview/Execute buttons on `block`, require explicit acknowledgment checkbox on `warn`
  - _Requirements: 6.3, 6.4_

- [x] 15. Execution Engine — preview
  - [x] 15.1 Implement dialect-aware "top 100 rows" wrapping and `POST /api/datasets/:id/preview` with bound timeout, returning columns+rows or a database error message
    - _Requirements: 7.1, 7.2, 8.1, 8.4, 8.5_
  - [x] 15.2 Write tests: preview respects row cap of 100, timeout triggers cancellation and clear error, read-only connection enforced
    - _Requirements: 7.1, 7.2, 8.1, 8.5_
  - [x] 15.3 Build `/datasets/:draftId/preview` frontend page: results table, error display, "Approve & Run Full Query" action
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 16. Execution Engine — full run with safeguards
  - [x] 16.1 Implement async execution job model (status: queued/running/succeeded/failed/cancelled) and `POST /api/datasets/:id/execute` / `GET /api/datasets/:id/execute/:jobId`
    - _Requirements: 7.3, 8.1_
  - [x] 16.2 Implement chunked/server-side-cursor fetch with row-cap enforcement (abort + `ROW_LIMIT_EXCEEDED` reason on exceed) and timeout enforcement (`TIMEOUT` reason)
    - _Requirements: 8.1, 8.2_
  - [x] 16.3 Implement per-source and per-user concurrency limiter (`CONCURRENCY_LIMIT` reason on rejection)
    - _Requirements: 8.3_
  - [x] 16.4 Implement export-to-file (CSV/Excel) on success and signed/expiring download URL endpoint
    - _Requirements: 7.4_
  - [x] 16.5 Enforce execute-permission check and require prior successful preview approval before allowing execute
    - _Requirements: 7.5, 9.3_
  - [x] 16.6 Write tests: row cap aborts cleanly, timeout aborts cleanly, concurrency limit rejects extra requests, execute blocked without preview approval, execute blocked without `can_execute` permission
    - _Requirements: 7.5, 8.1, 8.2, 8.3, 9.3_
  - [x] 16.7 Build `/datasets/:draftId/execute` frontend page: job status polling, progress/row count, download link on success, error reason display on failure
    - _Requirements: 7.4, 8.1, 8.2, 8.3_

- [x] 17. Audit logging surfacing
  - Implement admin-facing `GET /api/admin/audit-log` (filter by user/source/date) surfacing `query_audit_log` entries for compliance review
  - _Requirements: 6.5, 9.4_

- [x] 18. End-to-end wiring and happy-path verification
  - [x] 18.1 Wire full frontend wizard navigation draft-to-draft (sources → columns/requirement → criteria → review → preview → execute) with draft status guarding step access
  - [x] 18.2 Write Playwright e2e test: full happy path against a seeded test database produces a downloadable result
  - [x] 18.3 Write Playwright e2e test: a deliberately unindexed large-table query is blocked at the review step
  - _Requirements: 6.3, 7.1, 7.3, 7.4_
