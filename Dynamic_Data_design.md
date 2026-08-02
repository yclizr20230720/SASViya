# Design Document

## Overview

The platform is a wizard-driven dataset builder. Users move through a linear flow (Browse → Select Columns → Describe Requirement → Set Criteria → Generate SQL → Preview → Execute) backed by a single "dataset definition" draft object that accumulates state at each step. The backend never lets user input become raw SQL directly: it builds queries through a query-object model (tables, joins, columns, filters), evaluates them against each target database's execution plan before allowing preview/execution, and always executes through parameterized, read-only, timeout- and row-limited connections.

Stack:
- Frontend: React + Vite, TypeScript, React Router, TanStack Query for server state.
- Backend: Python Flask, SQLAlchemy Core (not ORM, since target schemas are external/pre-existing) for connection + query building, Flask-Login/JWT for auth.
- Metadata store: a small application database (Postgres or SQLite for dev) holding registered data sources, cached schema metadata, data dictionary entries, dataset drafts, and audit logs. This is separate from the target/source databases users query.

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────────────────────┐
│   React + Vite SPA  │  REST  │              Flask API                    │
│                      │ <────> │  ┌────────────┐  ┌──────────────────┐    │
│  - Browser page      │  JSON  │  │ Auth/RBAC  │  │ Schema Service    │    │
│  - Column Picker     │        │  └────────────┘  └──────────────────┘    │
│  - Requirement page  │        │  ┌────────────┐  ┌──────────────────┐    │
│  - Criteria page     │        │  │ Dictionary │  │ Draft Service     │    │
│  - SQL Review/Preview│        │  │ Service    │  │ (dataset defn)    │    │
│  - Execution page    │        │  └────────────┘  └──────────────────┘    │
└─────────────────────┘        │  ┌────────────────────────────────────┐   │
                                │  │ Query Builder + Source Matcher     │   │
                                │  └────────────────────────────────────┘   │
                                │  ┌────────────────────────────────────┐   │
                                │  │ Performance Evaluator               │   │
                                │  └────────────────────────────────────┘   │
                                │  ┌────────────────────────────────────┐   │
                                │  │ Execution Engine (preview/full)     │   │
                                │  └────────────────────────────────────┘   │
                                └───────────────┬────────────────────────────┘
                                                │  read-only, pooled, per-source
                                                ▼
                          ┌───────────────────────────────────────────┐
                          │  Registered Source Databases (N)           │
                          │  (Oracle / MSSQL / MySQL / Postgres ...)   │
                          └───────────────────────────────────────────┘
                                                │
                                                ▼
                          ┌───────────────────────────────────────────┐
                          │  App Metadata DB (sources, schema cache,   │
                          │  dictionary, drafts, audit log, users)     │
                          └───────────────────────────────────────────┘
```

Each registered source database has its own connection profile (dialect, host, credentials via secrets manager/env, read-only role) and its own configured limits (timeout, max rows, max concurrent queries). The Query Builder never talks to a source with the app's own credentials for anything but SELECT.

## Components and Interfaces

### Frontend pages (routes)

- `/sources` — Database & Schema Browser (Req 1, 2)
- `/datasets/:draftId/columns` — Column Selection (Req 3)
- `/datasets/:draftId/requirement` — Requirement description, manual + import (Req 4)
- `/datasets/:draftId/criteria` — Filter criteria (Req 5)
- `/datasets/:draftId/review` — Generated SQL + performance evaluation result (Req 6)
- `/datasets/:draftId/preview` — 100-row preview (Req 7)
- `/datasets/:draftId/execute` — Full execution status + download (Req 7, 8)
- `/admin/dictionary` — Data dictionary import/management (Req 2.4)
- `/admin/sources` — Register/manage data sources and their limits (Req 1, 8, 9)
- `/login`

All dataset-builder pages read/write a single draft via the Draft Service so users can leave and resume.

### Backend services (Flask blueprints)

**Auth service** (`/api/auth/*`)
- `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`
- Issues a signed session/JWT; every other endpoint requires it.
- Permission model: `role` per user, `source_permission(user_id, source_id, can_browse, can_execute)`. Enforced as a decorator on every route touching a specific source.

**Schema service** (`/api/sources/*`)
- `GET /api/sources` — list sources visible to current user.
- `GET /api/sources/:id/schemas` — list schemas registered for that source (a single source/connection can expose multiple schemas — see multi-schema note below).
- `GET /api/sources/:id/schemas/:schema/tables` — list tables in that schema (from cache).
- `GET /api/sources/:id/schemas/:schema/tables/:table/columns` — list columns + dictionary description (from cache).
- `POST /api/sources/:id/refresh-metadata` — admin-triggered re-introspection via SQLAlchemy `Inspector`, run once per registered schema, writes into schema cache tables.
- Schema cache refresh is a bounded background job (Flask + a simple job queue, e.g. RQ, or a synchronous call for small dev setups) so page loads never hit source DBs directly.

**Multi-schema sources.** A single physical connection (e.g. one Oracle service/DSN) can expose several schemas that a user is allowed to browse independently. Rather than modeling "one schema per source," a `data_source` owns a connection profile plus a list of registered schemas (`data_source_schemas`), and browse/permission/audit granularity is at the schema level, not just the source level. This matches Oracle services like F12PEDA, which expose multiple owners/schemas (e.g. `VEDA`, `SIMM`, `SISPC`, `WATCH`, `FDC`, `PTS`, `MESMGR`, `OUTGO`) under one DSN — each is registered as its own browsable/permission-scoped schema entry under the same `data_source`, instead of forcing a separate connection per schema.

**Dictionary service** (`/api/dictionary/*`)
- `POST /api/dictionary/import` — upload Excel/CSV mapping (source, table, column, description); validated then upserted.
- `GET /api/dictionary/:sourceId` — descriptions merged into schema responses by the Schema service.

**Draft service** (`/api/datasets/*`)
- `POST /api/datasets` — create new draft.
- `GET /api/datasets/:id`, `PATCH /api/datasets/:id` — read/update draft state (selected columns, joins, requirement items, criteria).
- Draft stored as JSON document in the metadata DB (see Data Models) so partial progress survives navigation/reload.

**Requirement import service** (`/api/datasets/:id/requirement/import`)
- Accepts `.xlsx/.csv/.json`, parsed with `pandas` (or `openpyxl`/`csv`/`json` directly to avoid heavy deps) into a normalized list of `{field_name, description, source_hint}`.
- Returns parsed preview; a separate `commit` call writes it into the draft only after user confirms.
- Runs each item through the **Source Matcher** (name similarity + dictionary description similarity, e.g. simple token overlap/fuzzy match via `rapidfuzz`) to produce ranked column candidates.

**Query builder + Source Matcher** (`/api/datasets/:id/generate-sql`)
- Input: draft's selected tables/columns/joins/criteria (or matcher-resolved columns from requirement items).
- Builds a dialect-aware SQL string using SQLAlchemy Core `select()`/`Table` reflection objects, not string interpolation. Filter values are always bound parameters.
- Infers joins from foreign keys in cached metadata when the user didn't explicitly specify one; if ambiguous, returns a 409-style "needs join clarification" response instead of guessing silently (supports Req 3.4).

**Performance Evaluator** (invoked as part of `generate-sql`, before returning "runnable" status)
- For each candidate query:
  1. Static check: every column used in a `WHERE`/`JOIN ON` clause is checked against cached index/key metadata for that table. Missing coverage → `warn` or `block` per configured severity.
  2. Row-count check: cached approximate row count for each table in the `FROM`/`JOIN` list compared against a configurable `full_scan_threshold` (per source). If a table exceeds it and has no key-covered filter, that's a `block`.
  3. Plan check (best-effort, dialect-dependent): runs `EXPLAIN`/`EXPLAIN PLAN`/`SET SHOWPLAN` against the source using a lightweight, timeout-bound connection, parses estimated cost/rows, escalates severity if the plan shows a full/table scan operator on a large table.
  4. Result is one of `pass`, `warn` (user must acknowledge), `block` (cannot proceed to preview/execute until the query is changed).
- Every evaluation (SQL text, verdict, reasons, user, timestamp) is written to the audit log table (Req 6.5).

**Execution Engine** (`/api/datasets/:id/preview`, `/api/datasets/:id/execute`)
- Preview: wraps the generated query as a bounded subquery/TOP/LIMIT/FETCH FIRST equivalent per dialect, executes with a short timeout (e.g. 15s configurable), returns up to 100 rows + column metadata.
- Full execute: runs asynchronously (job + status polling, since large pulls can be slow), enforces:
  - Per-source query timeout (cancel via driver-level statement timeout where supported, else a watchdog thread that closes the connection).
  - Max row cap: query executes with a server-side cursor / chunked fetch; if rows fetched exceeds the cap before completion, execution is aborted and the user is told to narrow criteria (Req 8.2) rather than silently truncating.
  - Max concurrent executions: a simple semaphore per source (and per user) held in the metadata DB or an in-process limiter (Redis-backed if multi-worker).
  - Connection uses a read-only DB role/credential configured per source.
- On completion, result is written to a temp export file (CSV/Excel) and a signed download link is returned (Req 7.4).

### Frontend/backend contract shape (illustrative, not exhaustive)

```
DatasetDraft {
  id, ownerId, status,
  selections: [{ sourceId, table, columns: [colName], alias }],
  joins: [{ leftTable, leftColumn, rightTable, rightColumn, type }],
  requirementItems: [{ fieldName, description, sourceHint, matchedColumn? }],
  criteria: [{ table, column, operator, value, scope? }],
  generatedSql?: string,
  evaluation?: { verdict: 'pass'|'warn'|'block', findings: [...] },
  previewResult?: { columns, rows, rowCountShown },
  executionJob?: { id, status, rowCount, downloadUrl }
}
```

## Data Models (metadata DB)

- `users(id, email, password_hash, role)`
- `source_permissions(user_id, source_id, schema_name null-able, can_browse, can_execute)` — `schema_name` null means "applies to all schemas under this source"; a row scoped to a specific schema overrides that for finer-grained access (e.g. a user allowed to browse `VEDA` but not `MESMGR` on the same F12PEDA connection).
- `data_sources(id, name, dialect, host, port, database, readonly_credential_ref, query_timeout_sec, max_rows, max_concurrent_queries, full_scan_threshold)` — one row per physical connection/DSN (e.g. one row for F12PEDA, one for F12PMES, one for the GreenPlum EDA warehouse).
- `data_source_schemas(id, source_id, schema_name)` — the set of browsable schemas exposed by a given source's connection (e.g. F12PEDA → `VEDA`, `SIMM`, `MESMGR`, `SISPC`, `WATCH`, `FDC`, `PTS`, `OUTGO`, each its own row).
- `schema_cache_tables(source_id, schema_name, table_name, approx_row_count, last_refreshed_at)`
- `schema_cache_columns(source_id, schema_name, table_name, column_name, data_type, is_indexed, is_key)`
- `dictionary_entries(source_id, schema_name, table_name, column_name null-able, description)`
- `dataset_drafts(id, owner_id, status, definition_json, created_at, updated_at)`
- `query_audit_log(id, user_id, source_id, sql_text, evaluation_verdict, evaluation_findings_json, executed_at, row_count, duration_ms)`

Credentials for source databases are never stored in `definition_json` or logs; `readonly_credential_ref` points to an env var / secrets manager key, resolved only at connection time.

**Known sources to seed for this project** (dialect/host/port confirmed from existing extraction scripts in the workspace; schema list for F12PEDA as provided by the user):
- `F12PEDA` (Oracle, `PEDA-scan.vsmc.com:1587`, service `F12PEDA`) — schemas: `VEDA`, `SIMM`, `MESMGR`, `SISPC`, `WATCH`, `FDC`, `PTS`, `OUTGO`.
- `F12PMES` (Oracle, `PMES-scan.vsmc.com:1581`, service `F12PMES`) — schemas seen in use: `SIMM` (MES), `SISPC` (Inline SPC), `WATCH` (WAT).
- `f12edagp` (GreenPlum/PostgreSQL, `10.92.67.25:5432`) — schema: `veda`.

Credentials observed hardcoded in the existing extraction scripts must NOT be reused as-is; they should be rotated and stored via `readonly_credential_ref` (env var/secrets manager) before this platform connects to these sources.

## Error Handling

- Validation errors (bad file format, incompatible criteria value/type, ambiguous join) → `400` with a structured `{ field, message }` list; frontend surfaces inline, no partial commits (Req 4.4, 5.4).
- Source connection failures → schema/browse endpoints return per-source error objects rather than failing the whole page (Req 1.4).
- Performance `block` verdicts → `422` with findings array; frontend renders a blocking banner and disables the "Preview"/"Execute" actions until the query is revised, or forces explicit acknowledgment for `warn`.
- Execution timeout/row-cap exceeded → job status `failed` with a specific reason code (`TIMEOUT`, `ROW_LIMIT_EXCEEDED`, `CONCURRENCY_LIMIT`); frontend shows the reason and suggests narrowing criteria.
- All unhandled exceptions return a generic `500` to the client but log full stack traces server-side; source connection strings/credentials are scrubbed from any logged error text.

## Security

- All endpoints require authentication except `/api/auth/login`; enforced via a Flask `before_request` hook checking session/JWT.
- Authorization checked per source (`can_browse`, `can_execute`) on every schema, draft, generate-sql, preview, and execute call — not just at the UI layer.
- All SQL execution uses parameter binding (SQLAlchemy `bindparam`/driver params), never string-formatted user values, closing the injection vector required by Req 8.4.
- Source DB connections use a dedicated read-only role; the app itself never has write credentials to source databases.
- File imports (dictionary, requirement) are size-limited and parsed with strict schema validation before any data is persisted.
- Audit log records who ran what, when, against which source, for compliance review (Req 6.5, 9.4).

## Testing Strategy

- Backend: pytest unit tests for Query Builder (given a draft, correct SQL/params produced), Performance Evaluator (given mock schema stats, correct verdicts), Source Matcher (given sample requirement items and dictionary entries, expected candidate ranking). Integration tests against a disposable SQLite/Postgres test database for end-to-end generate → preview → execute flow, including timeout and row-cap enforcement using intentionally slow/large test tables.
- Frontend: component tests (Vitest + React Testing Library) for the wizard step components and the SQL review/warning banner logic; a couple of end-to-end tests (Playwright) covering the full happy path and the "blocked by performance evaluator" path.
- Security-focused tests: attempt SQL injection through criteria values and requirement free text, confirm parameterization holds; confirm a user without `can_execute` cannot hit the execute endpoint even with a valid draft.
