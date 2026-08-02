# Requirements Document

## Introduction

This project is a web application that lets users build custom datasets from one or more system-registered databases without writing SQL by hand. Users browse available databases, tables, and columns (enriched with data dictionary descriptions when available), pick the columns they care about, describe what data they want (manually or via file import), define filter criteria, and let the system generate a SQL script. Every generated query is checked for performance risk (missing key usage, full table scans, excessive size) before it's allowed to run. Users preview 100 rows, then optionally execute the full query, all within enforced timeout and result-size limits to protect source databases.

Stack: React + Vite (frontend), Python Flask (backend).

## Requirements

### Requirement 1: Database & Schema Browser

**User Story:** As a data user, I want to see all available databases and their tables/columns, so that I know what data exists before building a dataset.

#### Acceptance Criteria

1. WHEN a user opens the database browser page THEN the system SHALL display all databases registered in the system that the user is authorized to access.
2. WHEN a user selects a database THEN the system SHALL display the list of tables in that database.
3. WHEN a user selects a table THEN the system SHALL display all columns of that table, including name and data type.
4. IF the metadata cannot be retrieved from a database (e.g. connection failure) THEN the system SHALL show an error state for that database without blocking the rest of the page.
5. WHEN schema metadata is requested THEN the system SHALL retrieve it from a cached metadata store rather than querying the live database on every page load, with a manual "refresh metadata" action available.

### Requirement 2: Data Dictionary Integration

**User Story:** As a data user, I want to see business descriptions of tables and columns, so that I understand what each column actually means.

#### Acceptance Criteria

1. IF a data dictionary entry exists for a table THEN the system SHALL display the table-level description alongside the table name.
2. IF a data dictionary entry exists for a column THEN the system SHALL display the column-level description alongside the column name and type.
3. IF no data dictionary entry exists for a table or column THEN the system SHALL display it without a description rather than an error.
4. WHEN an administrator imports or updates a data dictionary (e.g. via file upload) THEN the system SHALL persist the descriptions and associate them with the correct database/table/column.

### Requirement 3: Column Selection

**User Story:** As a data user, I want to go table by table and pick exactly the columns I need, so that my dataset only contains relevant fields.

#### Acceptance Criteria

1. WHEN a user is on the column selection page THEN the system SHALL let them select one or more tables from one or more databases.
2. WHEN a user selects a table THEN the system SHALL let them check/uncheck individual columns to include in the dataset definition.
3. WHEN a user selects columns from multiple tables THEN the system SHALL retain all selections across tables until the user explicitly clears them.
4. WHEN a user selects columns spanning multiple tables THEN the system SHALL allow the user to specify or SHALL infer the join relationship between those tables before generating SQL.
5. THE system SHALL persist the current column selection as part of the in-progress dataset definition (draft) so users can navigate away and return.

### Requirement 4: Data Requirement Description (Manual & File Import)

**User Story:** As a data user, I want to describe the data I need either by typing it in or importing a file, so that I don't have to manually configure everything by hand every time.

#### Acceptance Criteria

1. WHEN a user is on the data requirement page THEN the system SHALL allow manual entry of requirement items (field name, description, expected source hint) one by one.
2. WHEN a user imports a file THEN the system SHALL accept Excel (.xlsx), CSV, and JSON formats.
3. WHEN a file is imported THEN the system SHALL parse it into requirement items and show a preview before committing them to the dataset definition.
4. IF an imported file has invalid or unrecognized structure THEN the system SHALL reject it with a clear error message and SHALL NOT partially commit malformed rows.
5. WHEN requirement items exist (manual or imported) THEN the system SHALL attempt to match each item to candidate database/table/column combinations using name and data-dictionary description matching, and present the matches to the user for confirmation.

### Requirement 5: Filter Criteria Setup

**User Story:** As a data user, I want to define filter criteria per data object and scope, so that I only get the rows relevant to my use case.

#### Acceptance Criteria

1. WHEN a user is on the criteria page THEN the system SHALL let them add one or more filter conditions per selected table (data object).
2. THE system SHALL support common operators (equals, not equals, in, between, greater/less than, date range, like) appropriate to each column's data type.
3. WHEN a user defines a scope (e.g. date range, lot/wafer/site scope relevant to the source data) THEN the system SHALL apply that scope to all criteria referencing that table.
4. THE system SHALL validate that criteria values are compatible with the target column's data type before allowing the user to proceed.
5. THE system SHALL persist criteria as part of the in-progress dataset definition.

### Requirement 6: SQL Generation with Performance Evaluation

**User Story:** As a data user, I want the system to generate the SQL for me and warn me if it will hurt database performance, so that I don't accidentally run something dangerous.

#### Acceptance Criteria

1. WHEN a user finishes defining columns and criteria THEN the system SHALL generate a SQL script using the user-identified tables/columns, or SHALL suggest the best-matching data source when the user provided a free-text/imported description instead of explicit table/column selections.
2. BEFORE presenting a generated SQL script as runnable, the system SHALL evaluate it for performance risk, including at minimum: (a) whether filter/join columns are covered by an index or primary/unique key, (b) whether the query would require a full table scan on a table above a configurable row-count threshold, (c) estimated/explained cost from the database's query planner where available.
3. IF the performance evaluation detects a full table scan on a large table or missing key usage on a join/filter column THEN the system SHALL block automatic execution and SHALL notify the user with the specific reason and the offending table/column.
4. WHEN a performance warning is raised THEN the system SHALL still show the generated SQL to the user and SHALL require explicit user acknowledgment before allowing a preview or execution attempt.
5. THE system SHALL log every performance evaluation result (pass/warn/block) with the associated SQL and user for audit purposes.

### Requirement 7: Preview and Full Execution

**User Story:** As a data user, I want to preview 100 rows before running the full query, so that I can confirm the result is correct before committing to a full data pull.

#### Acceptance Criteria

1. WHEN a SQL script passes or is acknowledged past performance evaluation THEN the system SHALL execute a preview variant of the query limited to 100 rows and display the results in the GUI.
2. IF the preview execution fails THEN the system SHALL display the database error message to the user without executing the full query.
3. WHEN a user reviews the preview and approves it THEN the system SHALL execute the full query against the source database.
4. WHEN full execution completes THEN the system SHALL make the resulting dataset available to the user for download/export (e.g. CSV/Excel).
5. THE system SHALL NOT execute the full query without an explicit user approval action following a successful preview.

### Requirement 8: Execution Safeguards (Timeout & Size Limits)

**User Story:** As a system operator, I want enforced timeouts and result size caps, so that user-generated queries can't overload or take down a source database.

#### Acceptance Criteria

1. THE system SHALL enforce a configurable query timeout for both preview and full execution; if exceeded, the system SHALL cancel the query and notify the user.
2. THE system SHALL enforce a configurable maximum result row/size limit for full execution; if a query would exceed it, the system SHALL stop the query, notify the user, and suggest narrowing the criteria.
3. THE system SHALL enforce a configurable maximum number of concurrent query executions per user and system-wide, queuing or rejecting additional requests beyond the limit.
4. THE system SHALL run all user-submitted SQL through parameterized execution (no raw string concatenation of user input) to prevent SQL injection.
5. THE system SHALL run each dataset query using a database role/connection scoped to read-only access.

### Requirement 9: Authentication & Access Control

**User Story:** As a system operator, I want users to log in and only see databases/tables they're permitted to access, so that sensitive data isn't exposed to unauthorized users.

#### Acceptance Criteria

1. THE system SHALL require users to authenticate before accessing any database browsing, dataset building, or execution feature.
2. THE system SHALL restrict the databases/tables/columns visible to a user based on that user's assigned permissions.
3. THE system SHALL restrict execution of full queries to users with an explicit "execute" permission, separate from a "browse/design" permission.
4. THE system SHALL log all executed queries with user identity, timestamp, and target database for audit purposes.
