# Production Wafer Map / Reticle Die-Grid Drawing System
## Technical & Architectural Reference for Development

---

## 1. Scope

This reference covers everything needed to design a production-grade system that draws:
- The **reticle field** (the array of dies exposed in a single stepper shot)
- The **shot/stepping map** (how the reticle field repeats across the wafer)
- The full **die grid on a 300mm wafer**, including notch/flat orientation, edge exclusion, and partial (non-printable) die

It is organized as: (1) geometry & math foundations, (2) shot-map/reticle-placement algorithms, (3) data models & industry file formats, (4) rendering architecture, (5) existing open-source tools you can study or reuse, (6) a recommended system architecture.

---

## 2. Geometry & Math Foundations

### 2.1 Wafer Coordinate System (SEMI M20 / M1)
Production tools don't just draw a circle — they anchor everything to a standardized coordinate system:
- **SEMI M20** ("Practice for Establishing a Wafer Coordinate System") defines how the wafer's physical geometry (center, flat, or notch) is used to build an x‑y (or r‑θ) coordinate system that every other array (die grid, site map, defect map) is referenced against. <cite index="54-1">This system lets characterization and inspection equipment report the precise location of any point, and lets die/site/map arrays be tied back to the physical wafer geometry.</cite>
- **Notch vs. flat**: <cite index="56-1">Notches are the standard for 200mm/300mm wafers because they allow faster, more precise automated alignment and reduce edge stress compared to the older flat-edge convention used on smaller wafers.</cite> Your renderer needs a notch-orientation parameter (commonly 0°/90°/180°/270°, "down" being most common for 300mm) because it determines the rotation of the die-row/column axes relative to the physical wafer, not just a cosmetic marking.
- **Practical implication for your drawing engine**: build the die grid in wafer-relative Cartesian coordinates (origin at wafer center, or at a defined reference die per the format you import), then rotate the whole grid based on notch angle only at render time — never bake notch rotation into the underlying die coordinates, or you'll fight yourself when reading/writing SEMI G85/E142 files (see §4).

### 2.2 Gross Die Per Wafer (GDPW) — the sizing formula
For a first-pass estimate (not final placement), the closed-form approximation is:

```
GDPW = (π × (D/2 − E)²) / S  −  (π × (D/2 − E)) / √(2 × S)
```
where `D` = wafer diameter (300mm), `E` = edge exclusion (typically 2–3mm), `S` = die area (die width × die height, mm²). <cite index="8-1">The first term estimates how many rectangles fit into the usable circular area; the second term subtracts the partial dies lost at the wafer edge.</cite> <cite index="8-1">This is only an approximation — actual die placement uses stepping algorithms that can recover a few extra dies depending on die aspect ratio and orientation</cite>, so use GDPW for quick UI feedback/cost modeling, not as the authoritative die count. Implement the real count via the stepping/placement algorithm in §3.

### 2.3 Precision requirements
Real optimal-layout search needs much finer resolution than you'd expect: <cite index="7-1">with small die sizes, sub-hundred-micron step values are needed to reach the true optimum, and a brute-force search at 0.1 µm resolution for a ~4.66×4.43mm die requires well over a billion permutation evaluations</cite> — which is why production tools use smarter search strategies rather than exhaustive brute force (see §3.2–3.3).

---

## 3. Reticle Shot-Map & Die-Placement Algorithms

### 3.1 What a shot map actually encodes
<cite index="32-1">A shot map is the diagram of every reticle shot printed on the wafer; it captures the offset of the center shot plus the stepping (pitch) distance in X and Y.</cite> <cite index="27-1">Stepper software historically computed the maximum printable die count and the corresponding die map from the stepper's own performance parameters — shot size, die-per-shot count, and the number of row/column shots needed to cover the wafer.</cite>

Your data model should separate two layers cleanly:
1. **Reticle layer** — an N×M array of individual die cells inside one reticle field (a reticle can and often does hold multiple distinct die designs, e.g., multi-project wafer / shuttle runs).
2. **Shot/step layer** — how that reticle-field rectangle repeats (steps) across the wafer in a regular row/column grid, with a single global (x-offset, y-offset) controlling where the shot grid sits relative to the wafer center.

### 3.2 Center-offset optimization (the core placement algorithm)
This is the algorithm actually used in production stepper/wafer-layout software, and it's the one you should implement first:

- <cite index="9-1">Starting from the shot map centered on the wafer, offset the shot-grid center in small increments in +X, −X, +Y, and −Y (increments limited by stepper resolution, i.e., a fraction of one shot width), and for each offset compute the number of fully printable die.</cite>
- <cite index="9-1">The offset that yields the maximum printable-die count is selected as the shot map's placement; production layout software additionally keeps the bottom shot row clear of the scribe/notch title area so wafer ID text doesn't collide with printed shots.</cite>
- A more general (non-integer-grid) formulation: <cite index="34-1">for each candidate chip-to-wafer offset, construct the full die matrix, apply edge/notch/flat exclusion and any known yield weighting, then generate a set of candidate shot maps (reticle exposure groupings) that cover that die matrix, score each on the fab's chosen metric (chips-per-wafer, chips-per-hour, etc.), and output the best-scoring shot map including its offset.</cite>

**Recommended implementation for your system**: a 2D grid search over (dx, dy) ∈ [0, pitch_x) × [0, pitch_y) at a fine step (e.g., pitch/1000), vectorized: for each candidate offset, compute die-row/column index bounds analytically (not per-die iteration) using `floor()/ceil()` against the wafer's circular boundary equation, then just count. This turns "billions of brute-force die-by-die checks" into a fast vectorized sweep over offsets only — get the count analytically per offset rather than looping per die per offset.

### 3.3 Faster-than-brute-force strategies (for large search spaces / irregular die placement)
When you need optimum accuracy at fine step resolution, or you're supporting non-regular ("irregular") die placement schemes for cost-sensitive substrates:
- **Hierarchical/coarse-to-fine grid search**: <cite index="30-1">divide the feasible offset space into a coarse grid, evaluate a placement-optimization routine at each grid center, pick the best-performing grid cell, then refine by recursing into that cell at finer resolution</cite> — this is the standard way to avoid the billions-of-evaluations brute-force trap from §2.3.
- **Boundary-tracing search**: <cite index="6-1">divide the feasible center-location region into grid cells sized to the step resolution; from an initial cell, move the wafer-center candidate through the grid using a boundary-tracing walk (e.g., Moore-neighbor tracing), and at each step recompute the coverage region and the count of fully-contained die, keeping whichever offset improves on the current best.</cite>
- **Closed-form GDW approximation for irregular layouts**: for research-grade cost estimation on irregular (non-rectangular-grid) die placement, <cite index="30-1">a simple analytic GDW approximation formula has been shown to land within 5% of the true optimum in general, and within roughly 1% when the die's long-side-to-wafer-radius ratio is under about 1%.</cite>
- **Combinatorial optimization (multi-project/shuttle reticles)**: when the reticle itself must be floorplanned (multiple distinct chip designs packed into one reticle to share mask cost), this becomes a 2D packing/floorplanning problem historically solved with <cite index="30-1">quadrisection-based simulated annealing, or formulated and solved as a mixed integer linear program (mixed-ILP)</cite>. Relevant only if your system needs to support MPW/shuttle reticle layout, not simple single-product wafers.

### 3.4 Edge-yield "sweet spot" analysis
For a production yield-aware system, once you have die placement, you often want a yield-weighted view, not just a binary printable/not-printable map: <cite index="9-1">using a shot map, an initial best-fit offset, usable wafer diameter, a target yield margin, and per-location historical yield data, compute an estimated-yield curve as a function of radius from wafer center, then select a "sweet spot" radius where estimated yield crosses the target margin</cite> — this radius becomes a secondary visual ring/overlay on your wafer map, distinct from the hard edge-exclusion boundary.

---

## 4. Data Model & Industry-Standard File Formats

Building to these standards (even partially) makes your tool interoperable with real fab/test-house data instead of a closed toy format.

### 4.1 SEMI G85 — Map Data Format (die/bin grid, plain array)
<cite index="10-1">SEMI G85 defines, in file-format detail, how map data items relating to electronic mapping (originally wafer-focused, but applicable to any substrate — wafer, tray, strip, tape) are represented</cite>; note it only specifies the *format*, not the semantic meaning of each value. Structurally, a G85 file is XML with: a `Device` header (rows/columns, orientation, origin location, bin type, null-bin code), a set of `Bin` definitions (code → pass/fail quality + description), and a `Data` block containing one `Row` per wafer row as a delimited string of bin codes. This is the simplest format to implement first — a straightforward 2D array with a legend.

### 4.2 SEMI E142 — Substrate Mapping (modern XML, richer)
<cite index="14-1">SEMI E142 defines the data items required to report, store, and transmit map data for substrates including wafers, frames, strips, and trays, and explicitly covers assembly/packaging and device test.</cite> <cite index="12-1">It's a considerably richer XML schema than G85: it defines nested `Layout` elements (device size, step size, product ID), a `Substrate`/`SubstrateMap` pairing, explicit `Orientation` and axis-direction fields, and requires at least one named reference die per map so that row/column origin is unambiguous — because the "origin corner" convention is otherwise inconsistently interpreted across tools.</cite> <cite index="11-1">E142 also supports multiple devices per map and multiple maps per file, plus richer bin-code metadata than G85.</cite> If your system needs to talk to modern MES/test-cell software, target E142 as your primary interchange format and G85 as a simpler fallback/export option.

### 4.3 KLARF — inspection/defect data (KLA-Tencor)
<cite index="49-1">KLARF is KLA-Tencor's defect-reporting format; it contains standard wafer-map elements plus per-defect location-within-die data (something plain map files don't carry) and pointers to bitmap defect images, making it effectively a hybrid database rather than a pure map.</cite> <cite index="49-1">It's organized into five major sections, with a general/substrate-info section carrying the sample center location (the reference coordinate for the (0,0) die) and the die pitch.</cite> If your system will overlay real inspection/defect data (not just bin/pass-fail), plan a KLARF importer — this is the de facto standard defect-data format across the industry, not just KLA tools. `klarfkit` (Python, see §5.2) is a good reference implementation to study for parsing edge cases across KLARF versions.

### 4.4 Why there's no single universal format
<cite index="15-1">The industry never converged on one or even a few wafer-map formats: large manufacturers built their own proprietary formats, SEMI produced several standards over the years, and probe/assembly-equipment vendors each added their own — leaving a landscape of formats, some undocumented, spanning plain text, XML, and binary.</cite> **Design implication**: architect your internal die/bin data model as format-agnostic (a normalized `{row, col, x_mm, y_mm, bin_code, value?}` record set + wafer/reticle metadata), with pluggable importer/exporter modules per format (G85, E142, KLARF, STDF-derived bin maps, vendor CSV). Don't let any one file format's quirks leak into your core data model.

### 4.5 Practical example schema (informed by the above)
```
Wafer {
  wafer_id, lot_id, diameter_mm (300),
  notch_orientation_deg,          // 0/90/180/270
  edge_exclusion_mm,
  flat_or_notch: "notch"|"flat",
  reference_die: {row, col},      // per SEMI E142 requirement
  origin_location: "lower_left"|"upper_left"|...,
  axis_direction: "up_right"|...
}

Reticle {
  reticle_id,
  field_size_x_mm, field_size_y_mm,
  dies: [ {local_row, local_col, product_id, is_reference_mark} ]
}

ShotMap {
  step_x_mm, step_y_mm,           // stepping pitch
  center_offset_x_mm, center_offset_y_mm,   // from §3.2 optimization
  shots: [ {shot_row, shot_col, reticle_id} ]
}

DieRecord {
  grid_row, grid_col,             // integer die-grid coordinate
  x_mm, y_mm,                     // absolute wafer coordinate (die center or corner)
  bin_code, bin_quality: "pass"|"fail"|"null",
  parametric_value?: float,       // for parametric maps
  shot_row, shot_col,             // links die back to its reticle shot
  is_partial: bool                // edge/partial die flag
}
```

---

## 5. Rendering Architecture

### 5.1 Choosing SVG vs Canvas vs WebGL
This is the single biggest architecture decision and it's driven almost entirely by die count:

| Approach | Practical ceiling | Notes |
|---|---|---|
| **SVG** (DOM-based) | <cite index="42-1">works well up to a few thousand elements, then degrades quickly because every shape is a DOM node with real overhead</cite> | Best for interactivity/accessibility (native hover, click, CSS styling, screen-reader friendly) on lower-density wafers or zoomed-in reticle views |
| **Canvas 2D** | <cite index="44-1">tens of thousands of points comfortably, since it renders as a bitmap and avoids DOM overhead — but has no native per-element interactivity</cite> | Right default for a full 300mm wafer with hundreds to low-thousands of die at production die sizes |
| **WebGL** | <cite index="41-1">hundreds of thousands to millions of points with GPU-accelerated parallel rendering and minimal latency</cite> | Overkill for a single wafer map, but relevant if you're rendering many wafers in a lot/batch view simultaneously, or animating yield trends across a full lot |

**Recommended hybrid pattern** (used broadly in high-density interactive dashboards): <cite index="42-1">use Canvas or WebGL for the bulk die-grid rendering (the "heavy" layer), and overlay a thin SVG (or DOM) layer on top purely for tooltips, selection highlighting, and labels — this gets you Canvas/WebGL performance with SVG-level interactivity and accessibility where it matters.</cite> For very heavy scenes, <cite index="42-1">offload the rendering itself to a Web Worker via OffscreenCanvas so the main thread stays responsive for UI interaction.</cite>

### 5.2 Practical die-count guidance for your renderer
- Full 300mm wafer at typical die sizes (2–10mm square) → **hundreds to a few thousand die** → Canvas 2D is the safe default; SVG is viable if you keep per-die DOM nodes off (e.g., render bulk grid to canvas, use SVG only for the handful of currently-hovered/selected die).
- Reticle-only view (zoomed into one shot, a handful to dozens of die) → SVG is ideal; gives you free hover/click/tooltip wiring and crisp vector scaling for print/export.
- Lot-level (25 wafers) or fleet-level dashboards → WebGL or virtualized/tiled Canvas.

### 5.3 Color encoding conventions
- **Bin maps** (categorical): use a fixed, high-contrast categorical palette per bin code, always render "null"/untested die as a distinct neutral (e.g., light gray) so it's visually separable from real fail bins — this mirrors the G85 `NullBin` concept (§4.1).
- **Parametric maps** (continuous): a perceptually-uniform sequential or diverging colormap (avoid rainbow/jet for anything meant for engineering decisions — it visually distorts gradient perception); support explicit min/max clamping since a single outlier die shouldn't wash out the whole map's dynamic range.
- Libraries: Plotly's `imshow`/heatmap and D3's sequential/diverging scales both cover this well if you go the JS/Python-notebook route (see tools list below); D3-hexbin is relevant only if you intentionally want a hex-binned *aggregate* density view rather than per-die squares (not typical for wafer maps, but occasionally used for very high-die-count reticles or defect-density overlays).

---

## 6. Existing Open-Source Tools & Libraries (study or reuse)

### Python
| Project | Relevance |
|---|---|
| `dougthor42/wafer_map` (PyPI: `wafer-map`) | <cite index="23-1">Purpose-built wafer-map plotting with continuous or discrete data, zoom/pan, usable standalone or embedded as a panel in your own wxPython app; distinguishes die grid-coordinates from absolute floating-point wafer coordinates</cite> — <cite index="24-1">also natively knows SEMI M1‑0302 wafer size standards and lets you re-center the map on the wafer</cite>. Best single reference for correct grid-vs-absolute-coordinate modeling. |
| `cap1tan/wafermap` | <cite index="21-1">A general-purpose Python package specifically for plotting semiconductor wafer maps</cite>, useful as a second implementation reference for API design. |
| `xlhaw/wfmap` (PyPI: `wfmap`) | <cite index="26-1">Built on matplotlib/seaborn; produces both basic numeric/categorical wafer heatmaps and highly customizable trend charts tied to different shot-map definitions</cite> — good reference for combining shot-map awareness with the render layer. |
| `CozumelDiver/stdf2map` | <cite index="22-1">Converts industry-standard STDF (ATE test output) files into bin wafer maps; notably does not assume wafer/die size up front — it lets the tested-die data itself define the map shape, which can end up non-circular for partial-wafer test runs</cite>, and <cite index="22-1">is built on a lightweight STDF parser plus PIL for image generation, with per-part TOML config for bin colors/labels and auto-sizing for legible output from thumbnail to poster scale</cite>. Good reference if you need STDF ingestion. |
| `MichaelHotaling/klarfkit` | <cite index="46-1">Utilities for loading, plotting, and editing KLARF defect files — including overlaying multiple KLARF files into a single composite wafer map to reveal recurring process issues, coloring defects by time/process/class/size/inspector, and round-tripping KLARF data through CSV/Excel for editing</cite>. Best reference for defect-map (as opposed to bin-map) handling. |
| `wafer-view` (PyPI) | <cite index="25-1">An open-source viewer specifically for SEMI.org XML wafer-map standards; parses the XML and renders per-die-status bitmaps with per-status color/visibility toggles, plus computed total/pass/fail/yield summary stats</cite>. Good reference implementation for SEMI E142/G85 parsing. |

### JavaScript / Web
- GitHub topic `wafermap` includes a **Vue 3 + Canvas** component library <cite index="19-1">built specifically for visualizing wafer maps in a customizable, reusable way for semiconductor manufacturing use cases</cite> — worth reviewing for component API design.
- A more recent JS entry (dependency-free, ES modules) is described as <cite index="19-1">an interactive wafer-map visualization and yield-analysis tool covering rendering, spatial statistics, failure clustering, lot-level trends, and reticle analysis</cite> — directly relevant since it explicitly covers reticle-level analysis, not just flat die maps; worth reading its source for how it structures the reticle/shot/die relationship.
- `d3-hexbin`: <cite index="66-1">groups 2D points into hexagonal bins, useful for aggregating large point sets into a coarser visual representation with color and/or size encoding</cite> — only relevant if you add a defect-density aggregate overlay mode.

### Commercial / Industry-Standard Tools (for feature-parity reference, not integration)
- **KLA inspection/review tools** — industry-standard for defect inspection with <cite index="50-1">native KLARF support, 2D review, haze/uniformity mapping, and offline recipe/analysis tooling across 150/200/300mm wafers</cite>. Useful as the UX bar for defect-overlay features.
- **yieldWerx** and similar YMS (yield management systems) platforms — good references for how production tools structure defect wafer maps, zonal maps, trend charts, and histograms as a connected suite rather than isolated views.

---

## 7. Recommended System Architecture (putting it together)

1. **Core geometry engine** (language-agnostic logic, unit-testable): wafer boundary math, notch/flat rotation, edge-exclusion boundary, die-grid generation from `(die_w, die_h, scribe_x, scribe_y)`, and the shot-map center-offset optimizer (§3.2–3.3). This should have zero rendering dependencies.
2. **Normalized data model** (§4.5) with pluggable importers/exporters for SEMI G85, SEMI E142, KLARF, and STDF-derived bin maps — treat all of these as adapters into/out of your one internal schema.
3. **Rendering layer**, split by zoom level:
   - Full-wafer view → Canvas 2D (or WebGL if you'll animate/compare many wafers), thin SVG/DOM overlay for interaction (§5.1).
   - Reticle/shot detail view → SVG, since die counts are small and you want native interactivity + crisp export.
4. **Analysis layer** (optional but high-value for "production" positioning): yield-vs-radius "sweet spot" computation (§3.4), bin-count/yield summary stats (mirroring what `wafer-view` computes), defect clustering/spatial-signature detection for pattern recognition (scratch, edge roll-off, center-to-edge trends).
5. **Export layer**: SVG/PNG/PDF for reports, plus round-trip export back to G85/E142 so your tool composes with the rest of a fab's toolchain rather than being a dead end.

---

## 8. Key Sources for Further Reading

- SEMI G85 — Specification for Map Data Format: https://store-us.semi.org/products/g08500-semi-g85-specification-for-map-data-format
- SEMI E142 — Specification for Substrate Mapping: https://store-us.semi.org/products/e14200-semi-e142-specification-for-substrate-mapping
- SEMI M20 — Practice for Establishing a Wafer Coordinate System: https://store-us.semi.org/products/m02000-semi-m20-practice-for-establishing-a-wafer-coordinate-system
- SEMI E142 XML worked example: https://artwork.com/package/wmapconvert/map_formats/semi_E142/index.htm
- SEMI G85 XML worked example: https://www.artwork.com/package/wmapconvert/G85/index.htm
- KLARF format explainer: https://artwork.com/package/wmapconvert/map_formats/KLARF/index.htm
- Die-per-wafer (GDPW) calculator/guide: https://siliconanalysts.com/guide/chips-per-wafer
- `dougthor42/wafer_map`: https://github.com/dougthor42/wafer_map
- `cap1tan/wafermap`: https://github.com/cap1tan/wafermap
- `xlhaw/wfmap`: https://pypi.org/project/wfmap
- `CozumelDiver/stdf2map`: https://github.com/CozumelDiver/stdf2map
- `MichaelHotaling/klarfkit`: https://github.com/MichaelHotaling/klarfkit
- GitHub topic pages: https://github.com/topics/wafer-map and https://github.com/topics/wafermap
- SVG vs Canvas vs WebGL performance comparison: https://www.svggenie.com/blog/svg-vs-canvas-vs-webgl-performance-2025
