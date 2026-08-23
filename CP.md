A professional CP/EWS analytics system should combine **yield/bin analytics**, **die-level parametric analytics**, and **spatial wafer-map pattern detection**. The most valuable design is a progressive drill-down: **Product → Lot → Wafer → Bin/failure category → Test parameter → Die locations**.

CP bin maps are central because each die carries a pass/fail or bin classification, while spatial patterns can distinguish random loss from systematic excursions such as edge rings, center clusters, scratches, donuts, or localized regions. [pdf](https://www.pdf.com/understanding-semiconductor-data-visualization-with-wafer-maps-an-introduction/)

## Core CP analysis functions

| Function | Engineering question answered | Typical output |
|---|---|---|
| Yield dashboard | Which product, lot, wafer, tester, program, or probe card has yield loss? | Wafer yield trend, lot trend, yield distribution, excursion flags |
| Hard-bin / soft-bin Pareto | Which failure category causes the largest yield loss? | Pareto bars, cumulative yield-loss %, top bins |
| Bin contribution drill-down | Which wafers and dies contribute to a selected bad bin? | Wafer ranking, bin count, bin %, die list/map |
| Wafer-to-wafer comparison | How does a bad wafer differ from baseline good wafers? | Delta yield, delta bin mix, parameter-shift table |
| Parametric yield analysis | Which CP test parameter is failing or becoming marginal? | Pass/fail rate by test, LSL/USL violations, test Pareto |
| Distribution analysis | Has the electrical distribution shifted, widened, become bimodal, or developed tails? | Histogram, KDE/density, box plot, violin plot, percentile trend |
| Specification/capability analysis | Is the parameter centered and capable relative to test limits? | Mean, sigma, Cp/Cpk or Pp/Ppk, distance-to-limit |
| Good-versus-bad discrimination | Which parameters best separate good and bad die/wafer populations? | Effect size, KS statistic, AUC, logistic-regression importance |
| Parameter correlation | Which electrical parameters move together or explain a fail bin? | Correlation matrix, scatter matrix, regression / feature importance |
| Spatial wafer-map analysis | Is loss random or systematic by die position? | Bin map, parametric heat map, clustering score, spatial pattern label |
| Zone / radial analysis | Is yield loss concentrated at edge, center, quadrant, row, column, or custom region? | Yield by radial ring, quadrant, reticle field, custom zone |
| Site / touchdown analysis | Is the issue tied to a probe site, touchdown sequence, or test cell? | Site-to-site yield, site bin Pareto, first/last-touchdown trend |
| Retest / fail-flip analysis | Which dies change status across CP passes or re-probes? | First-pass vs final-pass yield, fail-flip map, recoverable-yield estimate |
| Cross-stage correlation | Does CP predict FT, reliability, inline, WAT, or inspection behavior? | CP-to-FT correlation, bin transition matrix, matched-die map |

Bin Pareto, pass/fail maps, fail-mode maps, soft/hard-bin views, fail-flip maps, defect overlays, trends, histograms, and correlation charts are established yield-analytics functions.  CP/EWS data preparation should also standardize test units, coordinates, test conditions, and metadata across sources before comparison. [dryield](https://dryield.com/semiconductor-test-and-yield-data-visualization/)

## Good-versus-bad wafer workflow

For your first requirement, avoid comparing only “good wafer” versus “bad wafer” overall yield. Use a controlled baseline and compare at several levels.

1. **Define population and baseline**
   - Good wafer: qualified wafers from the same product, revision, CP program, tester family, temperature, and preferably the same process window.
   - Bad wafer: excursion wafers based on yield threshold, excess bin loss, customer-return correlation, or engineering disposition.
   - Exclude incomplete wafers, engineering wafers, abnormal retest handling, and map/coordinate errors.

2. **Compare bin composition**
   - Calculate per wafer:
     \[
     \text{Bin Rate}_{b} = \frac{N_{\text{dies in bin }b}}{N_{\text{tested dies}}}
     \]
   - Rank bins by:
     - Bad-wafer bin rate
     - Delta versus baseline
     - Yield-loss contribution
     - Number of affected wafers
     - Spatial systematicity score

3. **Drill down from bin to fail tests**
   - For each dominant hard/soft bin, show:
     - Associated failing test item(s)
     - First failing test versus all failing tests
     - High-side / low-side fail direction
     - Test limit, guardband, and distance-to-limit
     - Fail count and unique-die count
   - Preserve both **hard bin** and **soft bin** logic. A hard bin is usually disposition-oriented; soft bins preserve more diagnostic detail.

4. **Compare parameters statistically**
   - For every one of the 100+ CP parameters, compute baseline-versus-bad metrics:
     - Mean and median delta
     - Standard deviation / variance ratio
     - P1, P5, P50, P95, P99 movement
     - Out-of-spec rate
     - Cpk/Ppk change where limits are meaningful
     - KS statistic for distribution shift
     - Effect size, such as standardized mean difference
     - Correlation with fail label or selected fail bin
   - Rank parameters using a composite score instead of p-value alone. With millions of dies, even tiny, irrelevant differences can become statistically significant.

5. **Check confounders before declaring root cause**
   - Tester, test head, probe card, site, program revision
   - Probe temperature, contact count, retest status
   - Lot, wafer sequence, process layer, fab route, recipe revision
   - Wafer orientation, coordinate transform, die/reticle position

A practical ranking score can be:

\[
\text{Priority} =
(\text{Yield Loss Contribution})
\times
(\text{Bad-vs-Good Effect Size})
\times
(\text{Spatial/Systematic Score})
\]

This prioritizes a parameter that both moves electrically and explains meaningful yield loss—not merely one with a small statistical shift.

## Wafer-map visualizations

Provide an interactive wafer map as the engineering “control center,” not merely an image. Bin maps are categorical maps of die performance, and zonal analyses commonly divide wafers by circle/ring, quadrant, column, or custom zones. [pdf](https://www.pdf.com/understanding-semiconductor-data-visualization-with-wafer-maps-an-introduction/)

### Essential map modes

| Map mode | Die color | Primary use |
|---|---|---|
| Overall pass/fail map | Good / bad / untested | Rapid yield-loss localization |
| Hard-bin map | One color per hard bin | Dominant failure-mode distribution |
| Soft-bin / fail-category map | One color per soft bin or category | Fine-grain diagnosis |
| Selected-test fail map | Pass / high fail / low fail / invalid | Test-specific failure localization |
| Parametric heat map | Continuous value scale | Gradients, center/edge shifts, local excursions |
| Delta map | Deviation from good-wafer baseline | Shows abnormal magnitude by die |
| Bin-overlay map | Selected bin over pass die | Quickly inspect one failure category |
| Fail-stack map | Pie/stack or priority color | Multi-fail die behavior |
| Retest/fail-flip map | Stable pass/fail, recovered, new fail | Probe/contact/retest diagnosis |
| Site / touchdown map | Site or sequence identifier | Tester/prober/probe-card signatures |
| Zone map | Radial ring, quadrant, reticle, custom polygon | Quantified spatial comparison |

### Key interaction behavior

- Click a die to expose: `LOT_ID, WAFER_ID, X, Y, HBIN, SBIN, all failing tests, selected parameter values, limits, site, touchdown, retest status`.
- Lasso or box-select dies, then automatically update:
  - Bin Pareto
  - Parameter distribution
  - Fail-test Pareto
  - Die table/export
- Select a bin or test item from the Pareto and immediately highlight only relevant dies.
- Synchronize maps: **bad wafer**, **good reference wafer**, and **delta map** side by side.
- Provide fixed, version-controlled bin definitions and color dictionaries. The same bin must never mean different failure modes across program revisions without an explicit mapping version.

### Spatial metrics to calculate

Do not rely only on visual judgement. Calculate quantitative features per wafer and per bin/test:

- Edge loss versus center loss
- Radial-ring yield profile
- Quadrant / half-wafer / row / column yield
- Spatial autocorrelation, such as Moran’s \(I\)
- Local clustering/hotspot score
- Largest connected fail cluster
- Cluster count, area, density, elongation, orientation
- Nearest-neighbor distance
- Reticle/shot periodicity score
- Site/touchdown periodicity score
- Random-versus-systematic classification confidence

Typical wafer-map categories include center, edge-ring, edge-local, local cluster, scratch, donut/ring, near-full, and random patterns.  Spatial analysis normally includes separating systematic patterns from random failures, then clustering the systematic regions. [arxiv](https://arxiv.org/html/2404.15436v1)

## Robust 680k-die architecture

At 680,000 dies × 100+ parameters, a raw wide table can exceed 68 million parameter values per wafer. The web application must not attempt to send all raw data to the browser.

### Recommended data model

Use a layered schema:

| Layer | Grain | Main contents |
|---|---|---|
| `wafer_summary` | One row per wafer | Yield, tested count, pass count, bin counts, spatial features, top excursion scores |
| `wafer_bin_summary` | Wafer × bin | Count, rate, contribution, zone/site statistics |
| `wafer_param_summary` | Wafer × parameter | Mean, sigma, quantiles, fail rate, Cpk/Ppk, distribution-shift metrics |
| `die_core` | One row per die | Wafer ID, X/Y, pass/fail, hard/soft bin, site, flags, compact categorical IDs |
| `die_measurement_long` | Die × parameter only as needed | Parameter ID, value, pass/fail direction, test-number metadata |
| `wafer_spatial_tiles` | Wafer × zoom tile × map mode | Pre-aggregated count/ratio/value tiles for rendering |
| `analysis_result` | Analysis run × entity | Pareto, ranked parameters, zones, clusters, anomaly score, model version |

### Performance design principles

- **Use columnar storage**: Parquet/Delta/Iceberg with compression and partitioning by product, test program, lot, wafer, and test date.
- **Keep raw measurements in long form** for scalable filtering and calculation; create a curated wide representation only for selected parameters or model training.
- **Precompute summaries** after CP ingestion: yield, bins, parameter quantiles, test fail counts, spatial features, and zone metrics.
- **Use a server-side query engine**: DuckDB for smaller departmental systems, or Spark/Databricks, ClickHouse, Trino, Snowflake, or SAS Viya/CAS for larger enterprise implementations.
- **Return only visible map data**: use level-of-detail rendering. At low zoom, return aggregate tiles; fetch individual dies only at high zoom or after a selection.
- **Avoid SVG for 680k dies**. Use WebGL/canvas point rendering or raster/vector tiles. SVG DOM objects at this scale will be slow and memory-heavy.
- **Represent bin and state as integers**, not repeated strings. Encode map colors client-side from a controlled lookup table.
- **Cache by wafer and analysis version**, especially static CP maps, spatial feature results, and parameter summaries.
- **Use asynchronous jobs** for expensive tasks such as all-parameter comparisons, clustering, reticle-periodicity analysis, or model inference.
- **Adopt progressive drill-down**: dashboard summaries first; then selected wafer/map; then selected bin/test; finally selected die measurements.

### Practical map rendering levels

| Zoom level | Returned content |
|---|---|
| Lot/product overview | Wafer-level yield and dominant-bin tiles |
| Wafer overview | Aggregated grid or radial/zone tiles |
| Mid zoom | One point per die or cluster representative points |
| High zoom / selected region | Individual dies plus selected test values |
| Die inspection | Full die record and failure sequence |

## Recommended engineering dashboard

Build the page around these coordinated panels:

1. **Filter and baseline panel**: product, device revision, lot, wafer, CP program, tester, prober, probe card, site, temperature, date, good/bad definition.

2. **Yield and bin panel**: wafer-yield trend, wafer ranking, bin Pareto, selected-bin contribution, bin transition/retest chart.

3. **Wafer-map panel**: synchronized bad/reference/delta maps, map-mode selector, die selection, zone overlays, coordinate and orientation controls.

4. **Parameter panel**: ranked bad-versus-good parameter table, histogram with LSL/USL, box/violin plot by wafer group, scatter plot with selected correlated parameter, parameter heat map.

5. **Root-cause evidence panel**: spatial-pattern score, site/touchdown association, process/tester stratification, lot chronology, matched CP-to-FT results if available.

## Governance and validation

For production engineering use, make the analytics results reproducible:

- Version CP test programs, limit tables, bin definitions, failure-category rules, baseline populations, and map coordinate transformations.
- Store the exact analysis query, data snapshot, parameter list, and code/model version with every engineering result.
- Include data-quality rules: duplicate die detection, illegal X/Y coordinates, invalid bin mapping, incomplete test records, missing/zero values, mixed test-program versions, and inconsistent wafer orientation.
- Validate each map against known-good raw STDF/ATDF or tester export records before operational deployment.
- Treat automated root-cause outputs as **prioritized hypotheses**, not proof. Corroborate dominant CP signatures with WAT, inline defect/inspection, process history, and final-test correlation where available. CP data is especially useful for cross-stage correlation when merged consistently with other manufacturing sources. [yieldwerx](https://yieldwerx.com/blog/complete-guide-to-test-data-manipulation-in-semiconductor-manufacturing/)

A robust minimum viable release should include: **yield trend, bin Pareto, interactive hard/soft-bin wafer map, selected-test fail map, parameter histogram with limits, good-vs-bad ranking, radial/quadrant analysis, site analysis, and exportable die lists**. After that, add automated pattern classification, CP-to-FT correlation, and anomaly detection.

A professional CP/EWS analytics system should combine **yield/bin analytics**, **die-level parametric analytics**, and **spatial wafer-map pattern detection**. The most valuable design is a progressive drill-down: **Product → Lot → Wafer → Bin/failure category → Test parameter → Die locations**.

CP bin maps are central because each die carries a pass/fail or bin classification, while spatial patterns can distinguish random loss from systematic excursions such as edge rings, center clusters, scratches, donuts, or localized regions. [pdf](https://www.pdf.com/understanding-semiconductor-data-visualization-with-wafer-maps-an-introduction/)

## Core CP analysis functions

| Function | Engineering question answered | Typical output |
|---|---|---|
| Yield dashboard | Which product, lot, wafer, tester, program, or probe card has yield loss? | Wafer yield trend, lot trend, yield distribution, excursion flags |
| Hard-bin / soft-bin Pareto | Which failure category causes the largest yield loss? | Pareto bars, cumulative yield-loss %, top bins |
| Bin contribution drill-down | Which wafers and dies contribute to a selected bad bin? | Wafer ranking, bin count, bin %, die list/map |
| Wafer-to-wafer comparison | How does a bad wafer differ from baseline good wafers? | Delta yield, delta bin mix, parameter-shift table |
| Parametric yield analysis | Which CP test parameter is failing or becoming marginal? | Pass/fail rate by test, LSL/USL violations, test Pareto |
| Distribution analysis | Has the electrical distribution shifted, widened, become bimodal, or developed tails? | Histogram, KDE/density, box plot, violin plot, percentile trend |
| Specification/capability analysis | Is the parameter centered and capable relative to test limits? | Mean, sigma, Cp/Cpk or Pp/Ppk, distance-to-limit |
| Good-versus-bad discrimination | Which parameters best separate good and bad die/wafer populations? | Effect size, KS statistic, AUC, logistic-regression importance |
| Parameter correlation | Which electrical parameters move together or explain a fail bin? | Correlation matrix, scatter matrix, regression / feature importance |
| Spatial wafer-map analysis | Is loss random or systematic by die position? | Bin map, parametric heat map, clustering score, spatial pattern label |
| Zone / radial analysis | Is yield loss concentrated at edge, center, quadrant, row, column, or custom region? | Yield by radial ring, quadrant, reticle field, custom zone |
| Site / touchdown analysis | Is the issue tied to a probe site, touchdown sequence, or test cell? | Site-to-site yield, site bin Pareto, first/last-touchdown trend |
| Retest / fail-flip analysis | Which dies change status across CP passes or re-probes? | First-pass vs final-pass yield, fail-flip map, recoverable-yield estimate |
| Cross-stage correlation | Does CP predict FT, reliability, inline, WAT, or inspection behavior? | CP-to-FT correlation, bin transition matrix, matched-die map |

Bin Pareto, pass/fail maps, fail-mode maps, soft/hard-bin views, fail-flip maps, defect overlays, trends, histograms, and correlation charts are established yield-analytics functions.  CP/EWS data preparation should also standardize test units, coordinates, test conditions, and metadata across sources before comparison. [dryield](https://dryield.com/semiconductor-test-and-yield-data-visualization/)

## Good-versus-bad wafer workflow

For your first requirement, avoid comparing only “good wafer” versus “bad wafer” overall yield. Use a controlled baseline and compare at several levels.

1. **Define population and baseline**
   - Good wafer: qualified wafers from the same product, revision, CP program, tester family, temperature, and preferably the same process window.
   - Bad wafer: excursion wafers based on yield threshold, excess bin loss, customer-return correlation, or engineering disposition.
   - Exclude incomplete wafers, engineering wafers, abnormal retest handling, and map/coordinate errors.

2. **Compare bin composition**
   - Calculate per wafer:
     \[
     \text{Bin Rate}_{b} = \frac{N_{\text{dies in bin }b}}{N_{\text{tested dies}}}
     \]
   - Rank bins by:
     - Bad-wafer bin rate
     - Delta versus baseline
     - Yield-loss contribution
     - Number of affected wafers
     - Spatial systematicity score

3. **Drill down from bin to fail tests**
   - For each dominant hard/soft bin, show:
     - Associated failing test item(s)
     - First failing test versus all failing tests
     - High-side / low-side fail direction
     - Test limit, guardband, and distance-to-limit
     - Fail count and unique-die count
   - Preserve both **hard bin** and **soft bin** logic. A hard bin is usually disposition-oriented; soft bins preserve more diagnostic detail.

4. **Compare parameters statistically**
   - For every one of the 100+ CP parameters, compute baseline-versus-bad metrics:
     - Mean and median delta
     - Standard deviation / variance ratio
     - P1, P5, P50, P95, P99 movement
     - Out-of-spec rate
     - Cpk/Ppk change where limits are meaningful
     - KS statistic for distribution shift
     - Effect size, such as standardized mean difference
     - Correlation with fail label or selected fail bin
   - Rank parameters using a composite score instead of p-value alone. With millions of dies, even tiny, irrelevant differences can become statistically significant.

5. **Check confounders before declaring root cause**
   - Tester, test head, probe card, site, program revision
   - Probe temperature, contact count, retest status
   - Lot, wafer sequence, process layer, fab route, recipe revision
   - Wafer orientation, coordinate transform, die/reticle position

A practical ranking score can be:

\[
\text{Priority} =
(\text{Yield Loss Contribution})
\times
(\text{Bad-vs-Good Effect Size})
\times
(\text{Spatial/Systematic Score})
\]

This prioritizes a parameter that both moves electrically and explains meaningful yield loss—not merely one with a small statistical shift.

## Wafer-map visualizations

Provide an interactive wafer map as the engineering “control center,” not merely an image. Bin maps are categorical maps of die performance, and zonal analyses commonly divide wafers by circle/ring, quadrant, column, or custom zones. [pdf](https://www.pdf.com/understanding-semiconductor-data-visualization-with-wafer-maps-an-introduction/)

### Essential map modes

| Map mode | Die color | Primary use |
|---|---|---|
| Overall pass/fail map | Good / bad / untested | Rapid yield-loss localization |
| Hard-bin map | One color per hard bin | Dominant failure-mode distribution |
| Soft-bin / fail-category map | One color per soft bin or category | Fine-grain diagnosis |
| Selected-test fail map | Pass / high fail / low fail / invalid | Test-specific failure localization |
| Parametric heat map | Continuous value scale | Gradients, center/edge shifts, local excursions |
| Delta map | Deviation from good-wafer baseline | Shows abnormal magnitude by die |
| Bin-overlay map | Selected bin over pass die | Quickly inspect one failure category |
| Fail-stack map | Pie/stack or priority color | Multi-fail die behavior |
| Retest/fail-flip map | Stable pass/fail, recovered, new fail | Probe/contact/retest diagnosis |
| Site / touchdown map | Site or sequence identifier | Tester/prober/probe-card signatures |
| Zone map | Radial ring, quadrant, reticle, custom polygon | Quantified spatial comparison |

### Key interaction behavior

- Click a die to expose: `LOT_ID, WAFER_ID, X, Y, HBIN, SBIN, all failing tests, selected parameter values, limits, site, touchdown, retest status`.
- Lasso or box-select dies, then automatically update:
  - Bin Pareto
  - Parameter distribution
  - Fail-test Pareto
  - Die table/export
- Select a bin or test item from the Pareto and immediately highlight only relevant dies.
- Synchronize maps: **bad wafer**, **good reference wafer**, and **delta map** side by side.
- Provide fixed, version-controlled bin definitions and color dictionaries. The same bin must never mean different failure modes across program revisions without an explicit mapping version.

### Spatial metrics to calculate

Do not rely only on visual judgement. Calculate quantitative features per wafer and per bin/test:

- Edge loss versus center loss
- Radial-ring yield profile
- Quadrant / half-wafer / row / column yield
- Spatial autocorrelation, such as Moran’s \(I\)
- Local clustering/hotspot score
- Largest connected fail cluster
- Cluster count, area, density, elongation, orientation
- Nearest-neighbor distance
- Reticle/shot periodicity score
- Site/touchdown periodicity score
- Random-versus-systematic classification confidence

Typical wafer-map categories include center, edge-ring, edge-local, local cluster, scratch, donut/ring, near-full, and random patterns.  Spatial analysis normally includes separating systematic patterns from random failures, then clustering the systematic regions. [arxiv](https://arxiv.org/html/2404.15436v1)

## Robust 680k-die architecture

At 680,000 dies × 100+ parameters, a raw wide table can exceed 68 million parameter values per wafer. The web application must not attempt to send all raw data to the browser.

### Recommended data model

Use a layered schema:

| Layer | Grain | Main contents |
|---|---|---|
| `wafer_summary` | One row per wafer | Yield, tested count, pass count, bin counts, spatial features, top excursion scores |
| `wafer_bin_summary` | Wafer × bin | Count, rate, contribution, zone/site statistics |
| `wafer_param_summary` | Wafer × parameter | Mean, sigma, quantiles, fail rate, Cpk/Ppk, distribution-shift metrics |
| `die_core` | One row per die | Wafer ID, X/Y, pass/fail, hard/soft bin, site, flags, compact categorical IDs |
| `die_measurement_long` | Die × parameter only as needed | Parameter ID, value, pass/fail direction, test-number metadata |
| `wafer_spatial_tiles` | Wafer × zoom tile × map mode | Pre-aggregated count/ratio/value tiles for rendering |
| `analysis_result` | Analysis run × entity | Pareto, ranked parameters, zones, clusters, anomaly score, model version |

### Performance design principles

- **Use columnar storage**: Parquet/Delta/Iceberg with compression and partitioning by product, test program, lot, wafer, and test date.
- **Keep raw measurements in long form** for scalable filtering and calculation; create a curated wide representation only for selected parameters or model training.
- **Precompute summaries** after CP ingestion: yield, bins, parameter quantiles, test fail counts, spatial features, and zone metrics.
- **Use a server-side query engine**: DuckDB for smaller departmental systems, or Spark/Databricks, ClickHouse, Trino, Snowflake, or SAS Viya/CAS for larger enterprise implementations.
- **Return only visible map data**: use level-of-detail rendering. At low zoom, return aggregate tiles; fetch individual dies only at high zoom or after a selection.
- **Avoid SVG for 680k dies**. Use WebGL/canvas point rendering or raster/vector tiles. SVG DOM objects at this scale will be slow and memory-heavy.
- **Represent bin and state as integers**, not repeated strings. Encode map colors client-side from a controlled lookup table.
- **Cache by wafer and analysis version**, especially static CP maps, spatial feature results, and parameter summaries.
- **Use asynchronous jobs** for expensive tasks such as all-parameter comparisons, clustering, reticle-periodicity analysis, or model inference.
- **Adopt progressive drill-down**: dashboard summaries first; then selected wafer/map; then selected bin/test; finally selected die measurements.

### Practical map rendering levels

| Zoom level | Returned content |
|---|---|
| Lot/product overview | Wafer-level yield and dominant-bin tiles |
| Wafer overview | Aggregated grid or radial/zone tiles |
| Mid zoom | One point per die or cluster representative points |
| High zoom / selected region | Individual dies plus selected test values |
| Die inspection | Full die record and failure sequence |

## Recommended engineering dashboard

Build the page around these coordinated panels:

1. **Filter and baseline panel**: product, device revision, lot, wafer, CP program, tester, prober, probe card, site, temperature, date, good/bad definition.

2. **Yield and bin panel**: wafer-yield trend, wafer ranking, bin Pareto, selected-bin contribution, bin transition/retest chart.

3. **Wafer-map panel**: synchronized bad/reference/delta maps, map-mode selector, die selection, zone overlays, coordinate and orientation controls.

4. **Parameter panel**: ranked bad-versus-good parameter table, histogram with LSL/USL, box/violin plot by wafer group, scatter plot with selected correlated parameter, parameter heat map.

5. **Root-cause evidence panel**: spatial-pattern score, site/touchdown association, process/tester stratification, lot chronology, matched CP-to-FT results if available.

## Governance and validation

For production engineering use, make the analytics results reproducible:

- Version CP test programs, limit tables, bin definitions, failure-category rules, baseline populations, and map coordinate transformations.
- Store the exact analysis query, data snapshot, parameter list, and code/model version with every engineering result.
- Include data-quality rules: duplicate die detection, illegal X/Y coordinates, invalid bin mapping, incomplete test records, missing/zero values, mixed test-program versions, and inconsistent wafer orientation.
- Validate each map against known-good raw STDF/ATDF or tester export records before operational deployment.
- Treat automated root-cause outputs as **prioritized hypotheses**, not proof. Corroborate dominant CP signatures with WAT, inline defect/inspection, process history, and final-test correlation where available. CP data is especially useful for cross-stage correlation when merged consistently with other manufacturing sources. [yieldwerx](https://yieldwerx.com/blog/complete-guide-to-test-data-manipulation-in-semiconductor-manufacturing/)

A robust minimum viable release should include: **yield trend, bin Pareto, interactive hard/soft-bin wafer map, selected-test fail map, parameter histogram with limits, good-vs-bad ranking, radial/quadrant analysis, site analysis, and exportable die lists**. After that, add automated pattern classification, CP-to-FT correlation, and anomaly detection.

# CP (Circuit Probe) Data Analysis System
### Functional & Technical Requirements Document
*For IT System Design & Development Reference*

**Document Version:** 1.0 &nbsp;|&nbsp; **Status:** Draft for IT System Design &nbsp;|&nbsp; **Prepared:** 2026

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [System Design Philosophy](#2-system-design-philosophy)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements — Performance & Scalability](#4-non-functional-requirements--performance--scalability)
5. [Dashboard / UI Panel Requirements](#5-dashboard--ui-panel-requirements)
6. [Data Governance & Validation Requirements](#6-data-governance--validation-requirements)
7. [Glossary of Key Terms](#7-glossary-of-key-terms)
8. [Recommended MVP Scope](#8-recommended-mvp-scope)

---

## 1. Purpose & Scope

This document defines the functional and non-functional requirements for a CP (Circuit Probe) / EWS engineering data analysis system. It is intended as the baseline specification for IT system design, covering:

1. CP data analysis logic for good/bad wafer and failure-category drill-down
2. Interactive wafer map visualization with bin and die-location detail
3. Performance/scalability requirements for datasets up to **680,000 dies across 100+ CP test parameters** per analysis session

The intended audience is the IT/software architecture team, data engineering team, and yield/product engineering stakeholders who will jointly design the ingestion pipeline, data model, analytics services, and web front end.

### 1.1 In Scope

- CP/EWS bin and parametric data ingestion from STDF/ATDF/tester export sources
- Yield, bin, and parametric analytics with progressive drill-down (Product → Lot → Wafer → Bin → Test → Die)
- Good-vs-bad wafer statistical comparison workflow
- Interactive, multi-mode wafer map visualization (web-based) with die-level click-through
- Spatial pattern detection (systematic vs. random loss)
- System architecture for large-scale (680K-die, 100+ parameter) datasets

### 1.2 Out of Scope (this phase)

- Final Test (FT) and reliability data analysis, except where used as cross-stage corroboration
- Automated fab equipment control / APC actioning
- Physical failure analysis (FA) lab workflows

---

## 2. System Design Philosophy

The system is built around a single organizing principle: **progressive drill-down analytics**, moving from a high-level yield signal down to individual die evidence without forcing the engineer to pre-select a failure hypothesis.

### 2.1 Drill-Down Hierarchy

> **Product → Lot → Wafer → Bin / Failure Category → Test Parameter → Die Location (X, Y) → Site / Touchdown**

Every analytic view (dashboard, table, or map) must support "click to filter downstream" and "click to see upstream context," so that a die selected on a wafer map immediately filters the Pareto and parameter panels, and a bin selected on a Pareto chart immediately filters the wafer map.

### 2.2 Three Analytical Pillars

- **Yield / bin analytics** — "which failure category, and how much yield does it cost?"
- **Die-level parametric analytics** — "which of the 100+ electrical parameters explains the failure?"
- **Spatial wafer-map pattern detection** — "is the loss random or systematic, and where?"

CP bin maps are central to this because each die carries a pass/fail or bin classification, and spatial patterns can distinguish random loss from systematic excursions such as edge rings, center clusters, scratches, donuts, or localized regions.

---

## 3. Functional Requirements

### 3.1 Core CP Analysis Functions

The table below defines the mandatory analytic functions the system must provide, the engineering question each answers, and the expected output artifact.

| Function | Engineering Question Answered | Typical Output |
|---|---|---|
| Yield dashboard | Which product, lot, wafer, tester, program, or probe card has yield loss? | Wafer yield trend, lot trend, yield distribution, excursion flags |
| Hard-bin / soft-bin Pareto | Which failure category causes the largest yield loss? | Pareto bars, cumulative yield-loss %, top bins |
| Bin contribution drill-down | Which wafers and dies contribute to a selected bad bin? | Wafer ranking, bin count, bin %, die list/map |
| Wafer-to-wafer comparison | How does a bad wafer differ from baseline good wafers? | Delta yield, delta bin mix, parameter-shift table |
| Parametric yield analysis | Which CP test parameter is failing or becoming marginal? | Pass/fail rate by test, LSL/USL violations, test Pareto |
| Distribution analysis | Has the electrical distribution shifted, widened, become bimodal, or developed tails? | Histogram, KDE/density, box plot, violin plot, percentile trend |
| Specification / capability analysis | Is the parameter centered and capable relative to test limits? | Mean, sigma, Cp/Cpk or Pp/Ppk, distance-to-limit |
| Good-vs-bad discrimination | Which parameters best separate good and bad die/wafer populations? | Effect size, KS statistic, AUC, logistic-regression importance |
| Parameter correlation | Which electrical parameters move together or explain a fail bin? | Correlation matrix, scatter matrix, regression / feature importance |
| Spatial wafer-map analysis | Is loss random or systematic by die position? | Bin map, parametric heat map, clustering score, spatial pattern label |
| Zone / radial analysis | Is yield loss concentrated at edge, center, quadrant, row, column, or custom region? | Yield by radial ring, quadrant, reticle field, custom zone |
| Site / touchdown analysis | Is the issue tied to a probe site, touchdown sequence, or test cell? | Site-to-site yield, site bin Pareto, first/last-touchdown trend |
| Retest / fail-flip analysis | Which dies change status across CP passes or re-probes? | First-pass vs. final-pass yield, fail-flip map, recoverable-yield estimate |
| Cross-stage correlation | Does CP predict FT, reliability, inline, WAT, or inspection behavior? | CP-to-FT correlation, bin transition matrix, matched-die map |
| Outlier detection (PAT / GDBN / SBL / SYL / OOF) | Which passing die are statistically abnormal or spatially at-risk despite passing? | Outlier die flags, dynamic/static/gap-based PAT limits, GDBN risk map |

> Bin Pareto, pass/fail maps, fail-mode maps, soft/hard-bin views, fail-flip maps, defect overlays, trends, histograms, and correlation charts are established yield-analytics functions across the industry. Outlier-detection methods commonly deployed alongside these include Part Average Testing (PAT — static, dynamic, and gap-based variants), Good Die in Bad Neighborhood (GDBN), Statistical Bin Limits (SBL), Statistical Yield Limits (SYL), and Out-of-Family (OOF) screening.

### 3.2 Good-vs-Bad Wafer Analysis Workflow

**Requirement:** the system must not compare only overall good-wafer vs. bad-wafer yield. It must support a controlled, multi-level comparison workflow as follows.

#### Step 1 — Define Population and Baseline

- **Good wafer population:** qualified wafers from the same product, revision, CP program, tester family, and test temperature — preferably the same process window.
- **Bad wafer population:** excursion wafers selected by yield threshold, excess bin loss, customer-return correlation, or engineering disposition.
- **Exclusion rules:** incomplete wafers, engineering wafers, abnormal retest handling, and map/coordinate errors must be filterable and excluded by default.

#### Step 2 — Compare Bin Composition

For every wafer, compute the bin rate:

```
Bin Rate(b) = N(dies in bin b) / N(tested dies)
```

Rank bins by: bad-wafer bin rate, delta versus baseline, yield-loss contribution, number of affected wafers, and spatial systematicity score.

#### Step 3 — Drill Down From Bin to Failing Tests

For each dominant hard/soft bin, the system must expose:

- Associated failing test item(s)
- First failing test vs. all failing tests
- High-side / low-side fail direction
- Test limit, guardband, and distance-to-limit
- Fail count and unique-die count

Hard bin (disposition-oriented) and soft bin (diagnostic detail) logic must both be preserved as distinct, independently queryable fields — never collapsed into one at ingestion.

#### Step 4 — Compare Parameters Statistically

For every one of the 100+ CP parameters, the system must compute baseline-vs-bad metrics:

- Mean and median delta
- Standard deviation / variance ratio
- P1, P5, P50, P95, P99 percentile movement
- Out-of-spec rate
- Cpk/Ppk change where limits are meaningful
- KS statistic for distribution shift
- Effect size (e.g., standardized mean difference)
- Correlation with fail label or selected fail bin

> Parameters must be ranked by a composite score rather than by p-value alone, since at 680K-die scale even tiny, practically irrelevant differences can become statistically significant.

#### Step 5 — Check Confounders Before Declaring Root Cause

- Tester, test head, probe card, site, program revision
- Probe temperature, contact count, retest status
- Lot, wafer sequence, process layer, fab route, recipe revision
- Wafer orientation, coordinate transform, die/reticle position

#### Priority Scoring Formula

```
Priority = (Yield Loss Contribution) × (Bad-vs-Good Effect Size) × (Spatial / Systematic Score)
```

This formula must be implemented as a first-class, configurable ranking function so the system surfaces parameters that both move electrically and explain meaningful yield loss — not merely parameters with a small but statistically detectable shift.

### 3.3 Wafer Map Visualization Requirements

The wafer map is the engineering "control center" of the system — an interactive analytic surface, not a static image.

#### Required Map Modes

| Map Mode | Die Color Encoding | Primary Use |
|---|---|---|
| Overall pass/fail map | Good / bad / untested | Rapid yield-loss localization |
| Hard-bin map | One color per hard bin | Dominant failure-mode distribution |
| Soft-bin / fail-category map | One color per soft bin or category | Fine-grain diagnosis |
| Selected-test fail map | Pass / high fail / low fail / invalid | Test-specific failure localization |
| Parametric heat map | Continuous value scale | Gradients, center/edge shifts, local excursions |
| Delta map | Deviation from good-wafer baseline | Shows abnormal magnitude by die |
| Bin-overlay map | Selected bin over pass die | Quickly inspect one failure category |
| Fail-stack map | Pie/stack or priority color | Multi-fail die behavior |
| Retest / fail-flip map | Stable pass/fail, recovered, new fail | Probe/contact/retest diagnosis |
| Site / touchdown map | Site or sequence identifier | Tester/prober/probe-card signatures |
| Zone map | Radial ring, quadrant, reticle, custom polygon | Quantified spatial comparison |
| GDBN / PAT risk overlay | Flagged outlier / at-risk die highlight | Reliability-risk screening on passing die |

#### Required Interaction Behavior

- Click a die to expose: `LOT_ID, WAFER_ID, X, Y, HBIN, SBIN`, all failing tests, selected parameter values, limits, site, touchdown, retest status.
- Lasso or box-select dies, then automatically update: Bin Pareto, Parameter distribution, Fail-test Pareto, and Die table/export.
- Select a bin or test item from a Pareto chart and immediately highlight only the relevant dies on the map.
- Synchronize maps side by side: bad wafer, good reference wafer, and delta map.
- Maintain fixed, version-controlled bin definitions and color dictionaries — the same bin code must never mean different failure modes across program revisions without an explicit mapping version.

### 3.4 Spatial Metrics Requirements

Spatial judgement must not rely on visual inspection alone. The system must calculate the following quantitative features per wafer and per bin/test:

- Edge loss vs. center loss
- Radial-ring yield profile
- Quadrant / half-wafer / row / column yield
- Spatial autocorrelation (e.g., Moran's I)
- Local clustering / hotspot score
- Largest connected fail cluster
- Cluster count, area, density, elongation, orientation
- Nearest-neighbor distance
- Reticle / shot periodicity score
- Site / touchdown periodicity score
- Random-vs-systematic classification confidence

> Typical wafer-map pattern categories to classify against: center, edge-ring, edge-local, local cluster, scratch, donut/ring, near-full, and random. Spatial analysis should first separate systematic patterns from random failure, then cluster and label the systematic regions.

---

## 4. Non-Functional Requirements — Performance & Scalability

**Target scale:** up to 680,000 dies × 100+ CP parameters per analysis session (68M+ parameter values). A raw wide table at this scale must never be sent to the browser in full, and must never be the default processing shape server-side.

### 4.1 Layered Data Model

| Layer | Grain | Main Contents |
|---|---|---|
| `wafer_summary` | One row per wafer | Yield, tested count, pass count, bin counts, spatial features, top excursion scores |
| `wafer_bin_summary` | Wafer × bin | Count, rate, contribution, zone/site statistics |
| `wafer_param_summary` | Wafer × parameter | Mean, sigma, quantiles, fail rate, Cpk/Ppk, distribution-shift metrics |
| `die_core` | One row per die | Wafer ID, X/Y, pass/fail, hard/soft bin, site, flags, compact categorical IDs |
| `die_measurement_long` | Die × parameter (as needed) | Parameter ID, value, pass/fail direction, test-number metadata |
| `wafer_spatial_tiles` | Wafer × zoom tile × map mode | Pre-aggregated count/ratio/value tiles for rendering |
| `analysis_result` | Analysis run × entity | Pareto, ranked parameters, zones, clusters, anomaly score, model version |

### 4.2 Performance Design Principles

| ID | Requirement |
|---|---|
| **PERF-01** | Use columnar storage (Parquet/Delta/Iceberg) with compression, partitioned by product, test program, lot, wafer, and test date. |
| **PERF-02** | Keep raw measurements in long form for scalable filtering; materialize a curated wide table only for a selected parameter subset or model training. |
| **PERF-03** | Precompute summaries at ingestion time: yield, bins, parameter quantiles, test fail counts, spatial features, and zone metrics. |
| **PERF-04** | Use a server-side query engine sized to deployment scale — DuckDB for smaller departmental systems; Spark/Databricks, ClickHouse, Trino, Snowflake, or SAS Viya/CAS for enterprise scale. |
| **PERF-05** | Return only visible map data using level-of-detail (LOD) rendering: aggregate tiles at low zoom; individual dies only at high zoom or after explicit selection. |
| **PERF-06** | Never render 680K dies as SVG DOM elements. Use WebGL/canvas point rendering (e.g., deck.gl, regl) or raster/vector map tiles. |
| **PERF-07** | Encode bin/state as integers, not repeated strings; resolve map colors client-side from a controlled lookup table. |
| **PERF-08** | Cache by wafer and analysis version — especially static CP maps, spatial feature results, and parameter summaries. |
| **PERF-09** | Run expensive computations (all-parameter comparison, clustering, reticle-periodicity analysis, model inference) as asynchronous background jobs, not inline request/response. |
| **PERF-10** | Implement progressive drill-down loading: dashboard summaries first → selected wafer/map → selected bin/test → selected die measurements. |

### 4.3 Map Rendering / Zoom Levels

| Zoom Level | Returned Content |
|---|---|
| Lot / product overview | Wafer-level yield and dominant-bin tiles |
| Wafer overview | Aggregated grid or radial/zone tiles |
| Mid zoom | One point per die, or cluster representative points |
| High zoom / selected region | Individual dies plus selected test values |
| Die inspection | Full die record and failure sequence |

---

## 5. Dashboard / UI Panel Requirements

The engineering web page must be organized as five coordinated, cross-filtering panels:

1. **Filter and baseline panel** — product, device revision, lot, wafer, CP program, tester, prober, probe card, site, temperature, date, and good/bad population definition.
2. **Yield and bin panel** — wafer-yield trend, wafer ranking, bin Pareto, selected-bin contribution, bin transition/retest chart.
3. **Wafer-map panel** — synchronized bad/reference/delta maps, map-mode selector, die selection, zone overlays, coordinate and orientation controls.
4. **Parameter panel** — ranked bad-vs-good parameter table, histogram with LSL/USL, box/violin plot by wafer group, scatter plot with correlated parameter, parameter heat map.
5. **Root-cause evidence panel** — spatial-pattern score, site/touchdown association, process/tester stratification, lot chronology, matched CP-to-FT results where available.

---

## 6. Data Governance & Validation Requirements

- Version CP test programs, limit tables, bin definitions, failure-category rules, baseline populations, and map coordinate transformations.
- Store the exact analysis query, data snapshot, parameter list, and code/model version alongside every engineering result (reproducibility requirement).
- Enforce data-quality rules: duplicate die detection, illegal X/Y coordinates, invalid bin mapping, incomplete test records, missing/zero values, mixed test-program versions, and inconsistent wafer orientation.
- Validate every rendered map against known-good raw STDF/ATDF or tester export records before operational deployment.
- Treat automated root-cause outputs as **prioritized hypotheses, not proof** — corroborate dominant CP signatures with WAT, inline defect/inspection, process history, and final-test correlation where available.
- Standardize test units, coordinates, test conditions, and metadata across all data sources before any cross-wafer or cross-lot comparison is performed.

---

## 7. Glossary of Key Terms

| Term | Definition |
|---|---|
| **STDF** | Standard Test Data Format — de facto industry-standard binary file format for semiconductor ATE test data (originally developed by Teradyne). |
| **GDBN** | Good Die in Bad Neighborhood — flags electrically-passing die that are spatially surrounded by a high concentration of failing die, as a reliability risk screen. |
| **PAT** | Part Average Testing — statistically derived per-parameter limits (beyond nominal spec) used to screen out electrical outliers. Variants: static (fixed for N lots/days), dynamic (recalculated per wafer), gap-based (screens deviation from bulk distribution, robust to non-Gaussian data). |
| **SBL / SYL** | Statistical Bin Limits / Statistical Yield Limits — flag wafers or lots whose bin distribution or yield deviates from historical norms. |
| **OOF** | Out-of-Family — screening method identifying wafers/lots/parameters that deviate from the established population "family." |
| **Hard Bin** | Final disposition-oriented bin code (e.g., pass / scrap category) assigned to a die. |
| **Soft Bin** | Finer-grained diagnostic bin preserving more detail than hard bin, used for failure-mode analysis. |
| **Cpk / Ppk** | Process capability / performance indices measuring how well a parameter's distribution fits within its test limits. |
| **WAT** | Wafer Acceptance Test — inline electrical test data used to corroborate CP-level findings against process/parametric drift. |

---

## 8. Recommended MVP Scope

For initial IT system design and phased delivery, the minimum viable release should include:

- Yield trend and bin Pareto
- Interactive hard/soft-bin wafer map
- Selected-test fail map
- Parameter histogram with limits
- Good-vs-bad parameter ranking (Section 3.2, Steps 1–4)
- Radial / quadrant spatial analysis
- Site analysis
- Exportable die lists

> **Phase 2 additions:** automated spatial pattern classification (ML-based), CP-to-FT correlation, GDBN/PAT outlier modules, and anomaly detection.
