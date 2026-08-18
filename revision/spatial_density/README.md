# Spatial activity-density hotspots

Where is a cCRE more active than expected, and is that region real?

`revision/Bayes_OldData/` estimates a per-cell Gamma-conjugate activity
posterior mean for every (cell, cCRE) pair;
`revision/Bayes_OldData/visualization/plot_all_ccre_spatial.py` plots it.
`revision/spatial_correlation/` asks whether that map has *any* spatial
structure (Moran's I / Geary's C). This workflow asks **where**, and returns the
region itself — an explicit boundary, a cell membership list, and a family-wise
guarantee that the whole region is enriched.

## The test

Cells sit at fixed positions and carry a mark `a_i`, the estimated activity. The
tested surface is the kernel-weighted **local mean** of the mark,

```
R(u) = sum_i w_i(u) a_i,      w_i(u) = K_h(u - x_i) / sum_j K_h(u - x_j)
```

not the raw activity density `sum_i K_h(u - x_i) a_i`. The denominator is
invariant under the null, so the two give identical p-values, but `R` is on the
same 1.0-is-baseline scale as the plotted activity instead of tracking where
cells happen to be dense.

The null is **random labelling**: reassign the observed activities to cells,
positions fixed — exactly the "randomly assign the estimates to the cells and
see what density you get" expectation. Under it the finite-population moments of
`R(u)` are exact,

```
E0[R(u)]   = mean(a)
Var0[R(u)] = var(a) * n/(n-1) * (sum_i w_i(u)^2 - 1/n)
```

so the standardised surface `z = (R - E0)/sqrt(Var0)` — Getis-Ord `Gi*` with
Gaussian weights — costs no permutations. `sum_i w_i(u)^2` is one extra
convolution, because the square of a normalised Gaussian of width `sigma` is a
normalised Gaussian of width `sigma/sqrt(2)` scaled by `1/(4 pi sigma^2)`.
Permutations are still needed for p-values (see *Mark transform* below).

**The region and the test are one object.** With `c_alpha` the `(1-alpha)`
quantile of the permutation maxima taken over every valid pixel *and* every
bandwidth,

```
E(alpha) = { u : z(u) >= c_alpha }
```

contains no null pixel with probability at least `1 - alpha`. Every pixel in it
is significant *simultaneously*, so the whole region can be asserted without a
per-pixel caveat and without any further correction for having searched — and
`E(alpha)` is empty exactly when the global max-test fails.

Secondary regions come from a **step-down**: peel the connected component holding
the peak, remove it together with one kernel footprint (its neighbours are driven
by the same cells), re-maximise every permuted surface over what is left, and
repeat. Each round derives its own threshold from the restricted null, so the
p-values need no extra multiplicity correction.

The boundary is a **band, not a line**. A cell-level bootstrap (resample cells
with replacement, rebuild the geometry, re-threshold) gives a per-pixel inclusion
probability; the 5% and 95% contours bound the boundary and
`boundary_band_units` reports its width.

## Two things that are easy to get wrong

**The tissue mask cannot be an `n_eff` threshold.** Far outside the tissue every
cell is roughly equidistant, so the kernel weights are near-uniform and the
effective sample size `n_eff = (sum k)^2 / sum k^2` grows toward the whole
section rather than shrinking. A pixel is inside the tissue when a real cell
lies within one bandwidth of it; `n_eff >= --min-effective-cells` is a separate
condition, bounding how few cells the local mean rests on. Both are required.

**The raw activity is too heavy-tailed to average.** Median 1.45 against a 99th
percentile of 23.6, with no upper bound. A raw local mean is whichever extreme
cell is nearby, and this costs *power*, not calibration: the permutation null
inherits the same tail, its maximum is inflated wherever the outlier lands, and a
broad but moderate elevation cannot beat it. On a 3-cCRE pilot, raw marks gave
`max_z = 18.1` at `p = 0.70` and produced a "region" that was a single edge pixel
containing no cells. The default `--mark-transform rank` replaces each activity
with its van der Waerden normal score, making the statistic a spatial rank test:
a region is enriched when its cells are *consistently* high, not when one of them
is enormous. Effect sizes (`mean_activity_in`, `rate_ratio`) are always reported
on the raw activity scale regardless.

## Choose the null deliberately — they disagree

| | `--null global` | `--null within_subclass` |
|---|---|---|
| activity reassigned to | any cell | cells of the same subclass |
| holds fixed | nothing | local cell-type composition |
| pilot: negative controls | **significant** | not significant |
| pilot: region size | most of the tissue | one compact 139-cell domain |
| a region means | "this area is active" | "more active than the same cell types elsewhere" |

`global` is the literal random-reassignment expectation and is the CLI default.
On this dataset it fails its own negative-control check: cell types are
spatially organised and their activities differ, so nearly any region beats a
null that ignores composition. `within_subclass` is the one whose regions
support a positional claim. The SLURM job runs both; read them together, because
the difference between them is itself the answer to "is this a cell-type domain
or a spatial one".

## Running it

```bash
# smoke test: 3 cCREs, one section, one bandwidth (~20 s)
/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python \
  revision/spatial_density/code/run_activity_density_hotspots.py \
  --cres CRE002 CRE003 CRE328 --sections sec1 --bandwidths 200 \
  --permutations 199 --bootstrap 50 --jobs 3 --outdir /tmp/sd_smoke

# the full thing, both nulls, plus characterisation, figures and calibration
sbatch revision/spatial_density/code/submit_activity_density_hotspots.slurm
```

Overridable via the environment: `PERMUTATIONS`, `BOOTSTRAP`, `MAX_REGIONS`,
`BANDWIDTHS`, `JOBS`, `REGION_JOBS`, `NULLS`.

### Scripts

| file | does |
|---|---|
| `code/activity_density.py` | the statistics: grid, geometry, surfaces, null, step-down regions. No I/O. |
| `code/run_activity_density_hotspots.py` | screen every (cCRE, section), delineate the ones that pass, write regions. |
| `code/characterise_regions.py` | subclass enrichment, marker genes, cross-cCRE Jaccard clustering. |
| `code/plot_activity_density_hotspots.py` | 4-panel figure per region, plus calibration QQ and overview. |

### Outputs (per null, under `results/<null>/`)

| file | one row per |
|---|---|
| `activity_density_summary.csv` | (cre, section): `max_z`, `p_fwer`, `q_global_bh`, `significant` |
| `activity_density_scales.csv` | (cre, section, bandwidth): scale stability |
| `activity_density_regions.csv` | region: area, centroid, peak, `n_cells`, `rate_ratio`, `boundary_band_units` |
| `region_cell_membership.csv.gz` | cell: `obs_name -> region_id`, with inclusion probability |
| `region_subclass_enrichment.csv` | (region, subclass): hypergeometric fold enrichment |
| `region_marker_genes.csv` | (region, gene): Mann-Whitney inside vs outside |
| `region_jaccard.csv`, `region_clusters.csv` | cCRE pairs sharing a region, and their clusters |
| `regions/{cre}_{section}.geojson` | region polygons in tissue coordinates (not WGS84) |
| `surfaces/{cre}_{section}.npz` | `z`, `activity_density`, `cell_density`, `valid`, `inclusion_probability`, `region_labels` |

`region_cell_membership.csv.gz` is the file to join against for anything
downstream — DE, TF motif, composition — so a region is reusable rather than
just a picture.

## Verification

1. **Synthetic recovery.** A planted circular hotspot at (1200, 2800) in uniform
   heavy-tailed noise is recovered at centroid (1196, 2777), `p = 0.005`; the
   same field with no hotspot gives `p = 0.56` and zero regions.
2. **Calibration.** `--permute-observed` shuffles the marks once up front.
   `p_fwer` must come out uniform and **no region may be emitted**. Stage 4 of
   the SLURM job runs this.
3. **Negative controls.** `CRE328/330/331/332/333/336/337` define the 1.0
   baseline and must not produce regions. They do not under `within_subclass`;
   they do under `global`, which is the finding described above.
4. **Anatomical sanity.** The pilot's one `within_subclass` region resolved to
   the olfactory bulb — OB-out Frmd7 Gaba 19.7x enriched, markers Grem1, Sp8,
   Eomes, Calb2, Slc6a3, Sp9. A region that characterises as a coherent known
   structure is the strongest end-to-end check available without an atlas.
5. **Cross-check.** cCREs with a region should overlap those flagged
   `spatially_autocorrelated` in
   `revision/spatial_correlation/results/activity_normalized/`; a large
   "region but no Moran's I" corner would be worth chasing.

## Density-stabilised maps (visualisation, no test)

`code/plot_stabilised_density.py` is the descriptive counterpart to the
permutation workflow: a per-cCRE activity map with the cell-density artefact
removed in closed form, no permutations and no p-values.

**What was already handled.** `R = D/N` is a kernel-weighted *mean*, not a sum,
so it is unbiased for local mean activity at any cell density. Dividing by `N`
*is* the density normalisation and it was always there.

**What this adds.** The residual density artefact is noise, not bias: the
sampling variance of `R(u)` falls with local effective sample size, so thin
areas throw extreme values by chance and dominate the eye. That variance is
exactly `var(a) * variance_factor(u)`, already computed by `build_geometry`, so
`activity_density.stabilise` needs no new estimation -- only empirical-Bayes
shrinkage against it:

```
sigma2(u) = var(a) * variance_factor(u)
tau2      = max(0, Var_pixels(R) - mean_pixels(sigma2))
w(u)      = tau2 / (tau2 + sigma2(u))
shrunk(u) = mu + w(u) * (R(u) - mu)
```

Verified on synthetic fields: `w -> 1` reproduces `R` exactly; `tau2 = 0` gives a
flat map; a planted hotspot keeps weight 0.96 at its peak; pure noise collapses
from sd 0.0925 to 0.0000; and on a **62.6x cell-density gradient with constant
true activity** the raw range of 1.41 becomes **0.0000** -- the property the
whole exercise exists for.

On real data the correction scales with bandwidth exactly as it should, biting
where sampling is thin and stepping aside where it is not (CRE138 sec2):

| h | mean weight | peak, raw -> stabilised |
|---|---|---|
| 50 | 0.825 | 5.17x -> **3.79x** |
| 100 | 0.914 | 3.90x -> 2.61x |
| 200 | 0.972 | 2.47x -> 2.37x |
| 400 | 0.994 | 2.15x -> 2.15x |

**Read the colour scale as relative.** Panels 3 and 4 show activity as a
multiple of that cCRE's own section baseline, which is what the shrinkage
targets. That is the only self-consistent choice: a pixel with no local evidence
lands exactly on 1.0 and reads as "nothing to say here". Absolute baselines vary
enormously between cCREs -- 11.7x the negative-control level for CRE138, 0.33x
for CRE174 -- and are recorded as `baseline_mean`.

**Limitation, stated on every figure.** This normalises out cell density only,
not cell-type composition. Activity differs by subclass and subclasses are
spatially organised, so bright areas may still be cell-type anatomy rather than
position. `stabilised_density_summary.csv` ranks cCREs by
`max_shrunk_relative`; that ranking is descriptive and is **not** a significance
test.

```bash
sbatch revision/spatial_density/code/submit_stabilised_density.slurm   # h = 100, 200, 400
```

Outputs: `figures/stabilised/h{H}/{cre}_{section}_stabilised.{png,pdf}` and
`results/stabilised/h{H}/stabilised_density_summary.csv`.

## What the full run found (job 19989029, `within_subclass`)

389 cCREs x 2 sections x 999 permutations x 3 bandwidths, screened in ~9 min on
16 cores. **27 of 778 pairs significant** at `q_global_bh <= 0.05`, yielding
**20 regions** (9 further components dropped under `--min-region-cells 50`).

The result is dominated by a single recurrent domain. Eighteen of the twenty
regions sit on the same ~510-cell patch of sec1 olfactory bulb, centred near
`(-3000, -6800)`, all detected at `h = 400`; the two sec2 regions sit on one
corresponding patch near `(3850, -5600)`. 120 cells are shared by ten or more
cCREs. Every one of these regions characterises identically: `OB-out Frmd7 Gaba`
enriched ~20x, with `Grem1` / `Sp8` / `Sp9` / `Th` as top markers. The
cross-cCRE Jaccard clustering collapses them into two clusters per section --
which is the point of that step: 389 separate maps, a handful of real domains.

**The patch is not a global artefact.** Across all 389 cCREs its inside/outside
activity ratio has median 0.63 -- most cCREs are *less* active there, not more --
and the inferred virus copy number is flat across the boundary (ratio 1.06,
79.5 vs 74.7 copies per cell). The 18 significant cCREs are a specific top
decile (median percentile 96.5), not a blanket elevation.

**But the domain is not cleanly enhancer-specific, and this needs stating.**
Two of the seven annotated negative controls sit in the same top decile of that
patch -- `CRE333` at ratio 2.33 (97.2nd percentile, and formally significant)
and `CRE336` at 2.24 (96.1st, just under). A negative control has no enhancer
activity by construction, so part of the elevation there is driven by something
common to the constructs rather than by regulatory function. Three of the 18
significant cCREs (`CRE288` 1.73, `CRE309` 1.79, `CRE303` 1.81) have *lower*
ratios than `CRE333` does. Read the strong end of the list (`CRE294`, `CRE012`,
`CRE348`, `CRE087`, `CRE003`, all above 2.7) with more confidence than the weak
end, and treat `CRE333`'s appearance as the calibration warning it is rather
than as a discovery.

The natural follow-up, not implemented here, is to calibrate effect sizes
against the negative-control distribution in the same region instead of against
a spatial permutation null -- the permutation null asks "is this location
special", which is a different question from "is this cCRE special at this
location".

## Caveats

- **A region is inflated by roughly one bandwidth.** Kernel smoothing spreads a
  true boundary, so the excursion set is systematically larger than the
  underlying domain — the synthetic recovery returned 930,000 units² for a true
  502,655 units² disk at `h = 200`. Compare regions at matched bandwidths and
  read `boundary_band_units` before treating an edge as precise.
- **`activity_normalized` is NaN for 118,819 of 408,621 cells**, whose subclasses
  have no pooled negative-control reference (only 44 of 328 do). Those cells are
  excluded from the analysis and drawn as grey background. `--matrix activity`
  keeps every cell on an arbitrary, non-baselined scale.
- The two sections are separate coordinate frames and separate tissue; nothing
  is ever pooled across them.
