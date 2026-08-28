Yes—your framing is correct, and the right way to build this is as an **engineering-grade spatial-layout and analytics platform**, not merely a charting widget.

For a production 300 mm wafer-map system that also represents reticle/shot structure, use a layered model:

\[
\text{Wafer geometry} \rightarrow \text{Shot/field grid} \rightarrow \text{Reticle cell layout} \rightarrow \text{Die grid} \rightarrow \text{Die data overlays}
\]

This preserves the crucial separation between physical geometry, lithography exposure placement, product design hierarchy, and test/inspection results.

## Target system architecture

### 1. Separate geometry from data

Do not encode wafer geometry, shot placement, die coordinates, and bin/parametric data into a single flattened drawing table. Build four durable domain entities.

| Layer | Purpose | Key attributes |
|---|---|---|
| Wafer | Physical substrate boundary and orientation | Wafer diameter, notch/flat, center, orientation, edge exclusion, usable radius |
| Exposure field / shot | Each scanner exposure position | Shot ID, field origin, field pitch, row/column, field rotation, exposure status |
| Reticle | Reusable layout template within a field | Reticle ID/revision, magnification, field size, die-cell templates, scribe lanes, alignment marks |
| Die | Individually testable product unit | Die X/Y, shot ID, local die coordinate, physical centre, active/partial/edge status |
| Overlay | Data attached to a die, field, or wafer | Bin, yield, parametric value, defect count, inspection classification, lot/wafer/process metadata |

The output should be a **canonical die-location table**. Every visual representation is generated from it; none should be stored as the source of truth.

Example relational/logical structure:

```text
Wafer
  ├── wafer_id
  ├── diameter_mm
  ├── notch_angle_deg
  ├── edge_exclusion_mm
  ├── coordinate_convention_id
  └── reticle_plan_id

ReticlePlan
  ├── reticle_id
  ├── revision
  ├── field_size_x_mm
  ├── field_size_y_mm
  ├── shot_pitch_x_mm
  ├── shot_pitch_y_mm
  ├── field_origin_x_mm
  ├── field_origin_y_mm
  └── rotation_deg

ReticleCell
  ├── reticle_id
  ├── local_row
  ├── local_col
  ├── die_type
  ├── local_x_mm
  ├── local_y_mm
  ├── width_mm
  ├── height_mm
  ├── active_flag
  └── product_die_flag

DieInstance
  ├── wafer_id
  ├── die_uid
  ├── global_die_x
  ├── global_die_y
  ├── shot_row
  ├── shot_col
  ├── local_reticle_row
  ├── local_reticle_col
  ├── center_x_mm
  ├── center_y_mm
  ├── coverage_ratio
  ├── die_state
  └── edge_die_flag

DieMeasurement
  ├── wafer_id
  ├── die_uid
  ├── measurement_name
  ├── measurement_value
  ├── unit
  ├── test_program_revision
  ├── timestamp
  └── source_system
```

This design lets one die be traced in both directions:

```text
Global die coordinate
  → shot / field
  → reticle-local cell
  → reticle revision
  → product / design variant
  → probe bin + parametric results + inspection data
```

That traceability is important for identifying field-repeaters, reticle signatures, systematic scanner effects, edge effects, probe artifacts, and process-module excursions.

## Coordinate system design

Coordinate ambiguity is one of the largest causes of wafer-map defects. Treat coordinate conventions as versioned master data—not UI choices.

### Canonical physical coordinate system

Use a machine-independent Cartesian wafer coordinate system internally:

- Wafer centre: \( (0,0) \)
- Unit: millimetres
- Positive X: right
- Positive Y: up
- Angular convention: counter-clockwise from positive X
- Wafer notch: represented as a physical orientation angle
- Die and shot positions: stored as centres, not only row/column labels
- Display orientation: a transform applied at render time

For a die centre \( (x_d, y_d) \), calculate radial position:

\[
r_d = \sqrt{x_d^2+y_d^2}
\]

If the wafer radius is \(R\) and the edge exclusion is \(e\), a quick centre-point eligibility condition is:

\[
r_d \le R - e
\]

For accurate edge-die treatment, test the die polygon against the usable-wafer circle. Classify each die as:

- `FULL`: die polygon completely inside usable wafer area
- `PARTIAL`: die intersects wafer boundary or exclusion boundary
- `OUTSIDE`: no usable intersection
- `EXCLUDED`: intentionally skipped by product, test, or exposure plan
- `SCRIBE`: non-product cell such as alignment/monitor/kerf

Avoid using simple die-centre inclusion alone for production yield. It can misclassify partial edge dies and distort gross-die, net-die, and yield calculations.

### Affine transform model

Represent each layer with affine transforms rather than separate procedural logic.

\[
\mathbf{p}_{wafer}
=
\mathbf{T}_{wafer}
\cdot
\mathbf{T}_{shot}
\cdot
\mathbf{T}_{reticle}
\cdot
\mathbf{p}_{local}
\]

For a reticle-local die coordinate \( \mathbf{p}_{local}=(x_l,y_l,1)^T \):

\[
\mathbf{p}_{wafer}
=
\begin{bmatrix}
\cos \theta & -\sin \theta & x_s \\
\sin \theta & \cos \theta & y_s \\
0 & 0 & 1
\end{bmatrix}
\mathbf{p}_{local}
\]

where:

- \(x_s,y_s\): shot origin on the wafer
- \(\theta\): shot or wafer orientation
- \(\mathbf{p}_{local}\): die coordinate in reticle-field coordinates

This supports rotations, offsets, mirrored coordinate conventions, scanner/stepper field offsets, and multiple reticle arrangements without redesigning the geometry engine.

### Coordinate conventions to explicitly configure

Your import and export adapters should handle these independently:

| Convention element | Examples |
|---|---|
| Origin | Wafer centre, upper-left, upper-right, lower-left, lower-right |
| Row direction | Up or down |
| Column direction | Left or right |
| Index base | 0-based or 1-based |
| Physical unit | mm, µm, die pitch, grid index |
| Orientation | 0°, 90°, 180°, 270°, arbitrary angle where needed |
| Die reference point | Centre, lower-left corner, field origin |
| Notch reference | Top, bottom, left, right, angle-based |
| Reticle magnification | Design-scale coordinates versus wafer-scale coordinates |

SEMI E142 is the most relevant interoperability reference for substrate maps. It covers two-dimensional maps, layout dimensions, step size, orientation, origin and axis direction, bin definitions, and die-level overlays; it is intended for exchanging map data between host and equipment through GEM/SECS interfaces. [pdf](https://www.pdf.com/standards/semi-e142-specification-for-substrate-mapping/)

A practical implication: internally retain your own richer canonical model, but provide a controlled **SEMI E142 import/export adapter** for equipment and MES-facing integration.

## Reticle and shot-map engine

### Reticle template model

A reticle is not always a single die. Your editor must support:

- Single-die field
- Arrayed identical dies
- Multi-product reticles
- Different die sizes in one field
- Test-key / monitor structures
- Scribe lanes and saw streets
- Alignment marks and fiducial regions
- Dummy structures
- Partial product cells
- Field-level exclusion regions
- Reticle revision comparison
- Optional design-reference overlays from GDSII/OASIS-derived metadata

Do not begin by directly rendering raw GDS polygons in the primary wafer view. Full layout geometry is far too dense and visually inappropriate at wafer scale. Instead, generate simplified reticle-cell polygons or bounding boxes from design data, then expose detailed GDS/OASIS viewing as a separate drill-down workflow.

### Shot generation

Define a shot lattice using:

- Field size \(W_f \times H_f\)
- Field pitch \(P_x, P_y\)
- Central-shot offset \(O_x, O_y\)
- Wafer orientation
- Valid-shot rule
- Exposure exclusions
- Shot traversal order, if representing scanner recipe behavior

For shot row \(i\), column \(j\):

\[
x_{shot}=O_x+jP_x
\]

\[
y_{shot}=O_y+iP_y
\]

A candidate field rectangle can be tested against the usable wafer boundary. Classify fields similarly to dies:

- Fully exposed/contained
- Partial edge shot
- Excluded
- Outside wafer
- Non-exposed field
- Rework/re-exposure state, if process context requires it

Commercial reticle-map generators commonly model shot layout using parameters including field pitch and central-shot offset, and can export DXF/GDSII-style layout representations. That is a useful design reference for the separation between a parametric shot plan and its generated geometry. [artwork](https://www.artwork.com/package/wlcsp/reticle_map/index.htm)

### Hierarchical identifiers

Use a stable identifier hierarchy. Never rely on display row/column strings as your sole primary key.

```text
wafer_id
reticle_plan_revision
shot_id
reticle_cell_id
die_uid
measurement_record_id
```

Example:

```text
Wafer:        LOT7A25-W12
Shot:         S(+04,-03)
Reticle cell: R(02,05)
Global die:   D(+38,-29)
Die UID:      LOT7A25-W12-S+04-03-R02-05
```

This enables field-repeat analysis: group observations both by `global_die_x/global_die_y` and by `local_reticle_row/local_reticle_col`.

That distinction is very valuable:

- A defect that repeats at the same **reticle-local location** across many shots suggests reticle, pellicle, scanner field, or reticle-related behavior.
- A defect concentrated at the same **wafer-global position** can indicate wafer/process spatial variation.
- A signature repeating every shot pitch can indicate field-based exposure or stage behavior.
- Radial or edge-concentrated signatures commonly point to wafer/process nonuniformity, edge exclusion, CMP, deposition, etch, thermal, or handling effects.

## Rendering design

### Use a hybrid renderer

For production-scale 300 mm wafers, implement two rendering modes:

| View | Recommended technology | Why |
|---|---|---|
| Full-wafer overview | WebGL2/WebGPU | Fast rendering of hundreds of thousands to millions of die instances |
| Pan/zoom die map | WebGL with level-of-detail | Maintains interactive response while zooming |
| Reticle/shot inspection | SVG, Canvas, or WebGL | Moderate geometry count; labels and editing are easier |
| Detailed die/field editor | SVG or Canvas | Accurate selection handles, annotations, and editable polygons |
| Static/export rendering | Server-side SVG/PDF/PNG | Reproducible reports and print-quality output |

For a web-first implementation:

- **Frontend:** React + TypeScript
- **GPU layer:** deck.gl, custom WebGL2, or WebGPU
- **Geometry engine:** TypeScript/Rust/WASM depending on calculation volume
- **API:** Python FastAPI, .NET, Java/Spring Boot, or Node.js—choose based on your fab data ecosystem
- **Data store:** PostgreSQL plus PostGIS or a columnar analytical store
- **Analytical data:** Parquet + object storage for large die/measurement data
- **Event / job pipeline:** Kafka, RabbitMQ, or your existing data platform scheduler
- **Cache:** Redis for layout metadata and active-view aggregates
- **Export service:** headless SVG/PDF renderer with deterministic color palettes and audit stamping

deck.gl is an especially strong candidate for the wafer overview because it uses GPU rendering and supports high-volume data. Its own performance guidance reports smooth pan/zoom for basic layers around one million items on capable hardware, although actual limits depend on device, attributes, layers, and interactivity. [deck](https://deck.gl/)

### Do not render every die as SVG

SVG is suitable for smaller reticle views and editable overlays, but it will become slow and memory-heavy when a 300 mm wafer contains tens of thousands to hundreds of thousands of dies, especially with labels, hit testing, and multiple overlays.

Use GPU instancing:

- One rectangle primitive
- Per-instance position
- Per-instance size
- Per-instance fill colour
- Per-instance border/flag state
- Per-instance die identifier index
- Optional texture/palette lookup for bin class

For categorical bin maps, encode bin values as small integers and map them to colours in a shader. Do not send CSS-like per-die styling strings to the browser.

### Level-of-detail rules

At low zoom, show aggregate information—not all cell borders:

| Zoom level | Content |
|---|---|
| Wafer overview | Wafer boundary, notch, edge exclusion, field/shot boundaries, aggregated bin or parametric heatmap |
| Shot level | Shot IDs, reticle-cell pattern, field yield, exposure/skip status |
| Die level | Die cells, bins, pass/fail, local and global coordinates |
| Selected die level | Full die metadata, test values, defect images, process history, annotations |

Use screen-space thresholds. For example:

- If die width is less than 2 pixels: aggregate per tile/shot.
- If die width is 2–8 pixels: draw solid colour cells without borders.
- If die width exceeds 8 pixels: enable borders and hover picking.
- If die width exceeds 20–30 pixels: render labels, selected measurements, and local coordinate markers.

### Picking and interaction

Use GPU color picking or spatial indexing. Avoid scanning all dies on mouse move.

Recommended interaction behavior:

- Hover: die ID, bin, selected metric, X/Y, shot ID, local reticle position
- Click: pin a die and open a metadata panel
- Shift-click: multi-select region/dies
- Lasso/rectangle selection: query selected die set
- Brush filter: bin, parametric range, shot, product, edge status, lot, process tool
- Drill-down: wafer → shot → reticle cell → die → measurement history
- Linked views: wafer map, histogram, CDF, box plot, trend chart, defect image panel
- Comparison mode: wafer-to-wafer, lot-to-lot, reticle revision-to-revision, pre/post process step

For high-volume datasets, maintain a tile index such as:

```text
wafer_id / overlay_id / zoom_level / tile_x / tile_y
```

Precompute or dynamically cache aggregate statistics per tile:

```text
die_count
pass_count
fail_count
yield
mean
median
standard_deviation
min
max
selected_bin_counts
defect_density
```

This lets a full-wafer screen remain responsive even when raw die-level data are much larger than the immediate rendering requirement.

## Analytics and algorithms

A professional system should combine deterministic engineering features with ML-assisted classification. Do not begin with deep learning alone; production users need explainable features and auditable root-cause workflows.

### Baseline spatial analytics

Implement these before advanced ML:

| Analysis | Engineering question answered |
|---|---|
| Die/wafer yield | What is the pass percentage and gross/net die count? |
| Radial bins | Is there centre-to-edge degradation or edge-ring behavior? |
| Angular sectors | Is there directional asymmetry, notch-related behavior, or handling signature? |
| Shot-level yield | Which exposure fields underperform? |
| Reticle-local yield | Does the same reticle cell fail across fields? |
| Row/column profiles | Are there scanner/stage/grid-direction signatures? |
| Nearest-neighbour density | Are failures clustered rather than random? |
| Connected components | How many discrete fail clusters exist? |
| Moran’s I / spatial autocorrelation | Is die behavior spatially correlated? |
| Ripley’s K / L | Is clustering present across multiple distance scales? |
| DBSCAN/HDBSCAN | Where are dense non-random failure clusters? |
| PCA/SVD on wafer matrices | What dominant spatial modes describe a wafer population? |
| Robust z-score | Which die, field, or wafer is anomalous relative to peer data? |
| EWMA/CUSUM | Is a map-derived metric drifting across time or lots? |

### Signature-specific detectors

Add deterministic feature extractors for known wafer signatures:

- **Edge ring:** Compare outer radial-band defect rate with central baseline.
- **Centre cluster:** High defect density within a central radius.
- **Scratch:** Hough transform or RANSAC line fitting on fail-die coordinates; use curved-line fitting where appropriate.
- **Radial spoke:** Angular concentration plus elongated clusters from centre outward.
- **Donut/ring:** Radial histogram peak at one or more characteristic radii.
- **Field repeaters:** Correlate defect masks across repeated shot cells.
- **Checkerboard/repeater:** Fourier analysis or autocorrelation at field/die pitch frequencies.
- **Row/column banding:** Project failures onto X/Y axes; detect statistically significant peaks.
- **Local cluster:** DBSCAN/HDBSCAN plus cluster shape descriptors.
- **Random pattern:** Compare observed spatial statistics against a simulated complete-spatial-randomness baseline.

Spatial filtering, clustering, and pattern recognition are established approaches in wafer-defect work because simple global defect counts can miss clustering and shape information. Published work specifically highlights the use of spatial filters, entropy fuzzy C-means, spectral clustering, and related methods to separate meaningful defect clusters from random noise. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10272367/)

### Reticle-aware pattern correlation

This is one of the most valuable differentiators for your system.

Transform every die location into two coordinate frames:

1. **Wafer-global frame**
   \[
   (x_w,y_w)
   \]

2. **Reticle-local frame**
   \[
   (x_r,y_r)
   \]

Then calculate defect-rate maps in both frames across many wafers.

```text
Global spatial alignment:
Same physical wafer positions across wafer population.

Reticle-local alignment:
Same within-field/within-reticle positions across repeated shots and wafers.
```

A production analysis query might be:

```sql
SELECT
    reticle_revision,
    local_reticle_row,
    local_reticle_col,
    COUNT(*) AS tested_dies,
    SUM(CASE WHEN final_bin IN ('FAIL', 'BAD') THEN 1 ELSE 0 END) AS failed_dies,
    1.0 * SUM(CASE WHEN final_bin IN ('FAIL', 'BAD') THEN 1 ELSE 0 END)
        / COUNT(*) AS fail_rate
FROM die_measurements
WHERE process_layer = 'TARGET_LAYER'
  AND measurement_timestamp >= :start_time
  AND measurement_timestamp < :end_time
GROUP BY
    reticle_revision,
    local_reticle_row,
    local_reticle_col;
```

Display this as a “reticle-local aggregate map.” If one location repeatedly shows abnormal failure over many fields/wafers, it becomes a high-priority investigation target.

### Machine learning roadmap

Use ML as a second-stage function after clean data lineage and deterministic analytics exist.

1. **Feature engineering baseline**
   - Radial failure histogram
   - Angular-sector histogram
   - Cluster count/area/eccentricity
   - Largest-cluster proportion
   - Edge/centre defect ratio
   - X/Y projections
   - Reticle-local positional features
   - Per-shot yield distribution
   - Frequency-domain energy at known shot/die pitches

2. **Unsupervised detection**
   - Isolation Forest for wafer-level anomalies
   - Robust covariance / Mahalanobis distance on feature vectors
   - HDBSCAN for map-family discovery
   - UMAP for visual grouping of historical wafers
   - Autoencoder reconstruction error for novel map signatures

3. **Supervised classification**
   - CNN/ResNet/EfficientNet on standardized map images
   - Vision Transformer only when training data and label quality justify it
   - Hybrid image + tabular models combining wafer-map image embeddings with tool/chamber/recipe/time metadata
   - Multi-label classifier because real wafers can show mixed patterns, such as edge ring plus scratch

4. **Explainability**
   - Grad-CAM or attention overlays for image models
   - Feature attribution for tabular models
   - Matched historical cases
   - Confidence, data-quality score, and “unknown pattern” output

WM-811K is widely used as a public reference dataset for wafer-map defect classification; it contains 811,457 maps, of which 172,950 are labelled, with map values representing background, good dies, and defective dies. It is useful for prototyping classifiers but should not be treated as a substitute for your fab’s own coordinate conventions, bin taxonomy, mixed-pattern data, and process context. [mathworks](https://www.mathworks.com/help/vision/ug/classify-defects-on-wafer-maps-using-deep-learning.html)

## Data contracts and standards

### Input adapters

Design an adapter layer rather than hardwiring each vendor file format into the UI.

Support:

- CSV/TSV files from probe/CP, final test, metrology, defect inspection, and yield systems
- Parquet for high-throughput analytical ingestion
- SEMI E142 XML import/export
- Custom fab MES or historian interfaces
- KLA/inspection result adapters where applicable
- Scanner/stepper shot-plan input
- Reticle-layout metadata input
- GDSII/OASIS-derived simplified geometry
- SQL warehouse ingestion
- REST, message-bus, or file-drop ingestion methods

SEMI E142 provides a useful interchange target because it defines a substrate map as a two-dimensional representation with layout information and per-location result or bin data; it applies not only to wafers but also strips, trays, and frames. [3dincites](https://www.3dincites.com/2020/06/semiconductor-backend-processes-additional-semi-standards-related-to-gem/)

### Canonical input table

At minimum, every die-level import should normalize to:

```text
wafer_id
lot_id
product_id
operation_id
process_layer
timestamp
source_system

die_x_source
die_y_source
coordinate_convention_id

die_x_canonical
die_y_canonical
center_x_mm
center_y_mm

shot_id
shot_row
shot_col
reticle_id
reticle_revision
reticle_local_row
reticle_local_col

bin_code
bin_name
bin_quality
pass_fail
measurement_name
measurement_value
measurement_unit

data_quality_flag
raw_record_reference
ingestion_run_id
```

Keep `die_x_source` and `die_y_source` permanently. They are essential for auditability when vendors, testers, probe programs, and internal applications use different orientation conventions.

### Validation rules

Before a map is released to users, validate:

- Wafer diameter matches product/process configuration.
- Notch orientation is available or explicitly unknown.
- Coordinate convention is declared.
- Die pitch is physically plausible.
- Die count is within expected limits.
- No duplicate die IDs after coordinate normalization.
- No impossible field assignment.
- Bin definitions are valid and versioned.
- Parametric units are normalized or explicitly retained.
- Percentage of out-of-bound die coordinates is below a configured threshold.
- Reticle revision and shot-map revision are traceable.
- Partial/edge-die policy is explicit.

A map that “looks right” can still be rotated, mirrored, or shifted. Build automated coordinate-consistency tests, including synthetic patterns deliberately placed at known locations such as top edge, notch-side edge, centre, and selected reticle cells.

## Recommended implementation sequence

### Phase 1: Minimum viable engineering viewer

Build a robust static-layout and bin-map product first.

- 300 mm circular wafer with notch and edge exclusion
- Configurable die pitch and die dimension
- Canonical Cartesian coordinates
- Full/partial/outside die classification
- Basic bin-map import
- GPU-rendered die cells
- Pan, zoom, hover, click, selection
- Die metadata side panel
- Export to PNG/SVG/PDF
- Orientation and coordinate-convention controls
- Reproducible view state encoded in URL or saved analysis session

**Acceptance benchmark:** render a full wafer interactively, with accurate die placement and a deterministic screenshot/export.

### Phase 2: Reticle and shot hierarchy

Add the lithography-aware layout engine.

- Parametric reticle templates
- Multi-die reticle-cell patterns
- Shot lattice/field placement
- Central shot offset
- Field boundary overlays
- Shot-level metadata and yield
- Reticle-local coordinates
- Field-repeat and reticle-repeat comparison
- Reticle revision management
- Import/export of shot-plan configuration

**Acceptance benchmark:** select any die and navigate to its shot, reticle-local cell, and global wafer position without ambiguity.

### Phase 3: Parametric and comparison analytics

- Continuous colour scales with fixed/spec-relative/quantile options
- Histogram, CDF, box plot, and radial profile
- Wafer-to-wafer and lot-to-lot comparison
- Shot-level and reticle-local aggregation
- Drill-down from outlier statistics to die locations
- Region-of-interest selection
- Data-quality indicators
- Saved filters, bookmarks, and reports

**Acceptance benchmark:** an engineer can identify whether a yield issue is global-wafer, edge-related, shot-related, reticle-local, or randomly distributed.

### Phase 4: Automated signature detection

- Cluster detection
- Edge/ring/scratch/spoke/banding signatures
- Spatial autocorrelation
- Wafer similarity search
- Rule-based alarms
- Historical baselining
- ML-driven pattern classification and anomaly detection

**Acceptance benchmark:** the system produces an explainable ranked set of signatures, supporting evidence, and comparable historical wafers—not just a model label.

### Phase 5: Enterprise hardening

- Role-based access control
- Product/lot data entitlements
- Immutable data lineage
- Audit log
- Dataset/schema versioning
- Job orchestration
- Observability and performance telemetry
- Test fixtures with known coordinate transformations
- SSO and MES/warehouse integration
- API contracts and backward compatibility

## Practical technology recommendation

For your background in semiconductor analytics, SAS/data engineering, and production software, I would recommend this stack:

| Concern | Recommended choice |
|---|---|
| Core frontend | React + TypeScript |
| Wafer-scale drawing | deck.gl or custom WebGL2/WebGPU instanced renderer |
| Reticle editor | SVG or Canvas overlay on a shared transform system |
| Backend API | FastAPI/Python or .NET if aligned with internal enterprise standards |
| Geometry calculations | Python initially; migrate critical polygon operations to Rust/WASM or a compiled backend service if required |
| Transactional metadata | PostgreSQL |
| Spatial geometry | PostGIS, with geometry stored in wafer-centred mm coordinates |
| Analytical data | Parquet on object storage plus DuckDB/Spark/Databricks depending on volume |
| Data processing | Python, SQL, Spark, or Databricks jobs |
| Statistical/SPC interface | SAS integration for existing statistical workflows, with a stable canonical export/API |
| ML experiment tracking | MLflow |
| Authentication | Corporate SSO through OIDC/SAML |
| Operational integration | SEMI E142 adapter plus MES/test/inspection connectors |

This architecture also lets you preserve current SAS workflows: SAS can generate standardized die-level Parquet/CSV extracts or call a backend API, while the mapping system handles canonical geometry, data lineage, GPU rendering, and interactive exploration.

## Key design decisions

1. Make physical wafer-centred millimetre coordinates canonical; treat all vendor row/column systems as transforms.

2. Maintain a strict hierarchy: wafer → shot → reticle cell → die → measurement overlay.

3. Preserve both **global wafer coordinates** and **reticle-local coordinates** for every die.

4. Use WebGL/WebGPU for full-wafer rendering; reserve SVG for low-volume/editable geometry.

5. Build tile-based level-of-detail aggregation from the start.

6. Store geometry, measurements, classifications, source coordinates, and layout versions separately.

7. Treat SEMI E142 as an interchange contract, not as the complete internal domain model. [pdf](https://www.pdf.com/standards/semi-e142-specification-for-substrate-mapping/)

8. Implement transparent deterministic spatial analytics before black-box ML.

9. Make coordinate validation, orientation validation, and layout revisioning mandatory release gates.

10. Ensure every chart, image export, model output, and root-cause finding can be traced to wafer ID, source files, coordinate convention, reticle revision, shot-map revision, and analysis version.

The core competitive capability will not be the wafer-map drawing itself. It will be reliable **cross-domain spatial attribution**: showing whether a signal belongs to the wafer, the scanner shot, the reticle-local position, the test flow, a process tool/chamber, or a specific time/lot excursion.
