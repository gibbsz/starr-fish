# HOCOMOCO TF motifs in whole-dataset joint-dropout significant cCREs

This folder compares significant cCREs with non-significant cCREs in each cell
type and identifies HOCOMOCO TF motifs enriched among the significant set. It
uses the whole-dataset joint-dropout mean-negative-control test, with no split
or combination by anatomical section.

The workflow preserves the HOCOMOCO motif-to-mouse-gene mapping and joins it to
TF expression in the same cell types. This makes it possible to distinguish a
sequence motif lead from a candidate TF that is actually expressed in that
cell type.

## Bayesian input and the two backgrounds

The input Bayesian table is:

```text
revision/bayesian_vs_fold_change/results/tables/
joint_dropout_mean_negative_control_tests.csv.gz
```

It was produced from the joint-dropout model by comparing each target's
posterior `log_gamma` with the draw-wise mean of the seven ordinary negative
controls. It contains only eligible, non-control, non-blacklisted
cCRE–cell-type pairs with target T7 >= 50 and sufficient negative-control T7.
Its `q_right` values are BH-adjusted across all eligible cCRE–cell-type pairs.

For every cell type:

```text
significant     = eligible cCREs with q_right <= 0.05
non-significant = eligible cCREs with finite q_right > 0.05
```

The non-significant set is the matched **enrichment comparison** background. It
has passed the same model and coverage filters as the significant set.

FIMO uses a different kind of background: a nucleotide null model. For each
cell type, the workflow writes a FASTA containing **all valid finite-q cCREs in
that cell type**, including both significant and non-significant cCREs. MEME
Suite `fasta-get-markov -m 0 -dna` estimates A/C/G/T frequencies from that
entire FASTA, combining reverse complements and using its 0.1 pseudocount. The
resulting cell-type-specific file is passed to FIMO with `--bgfile`.

Thus the design is:

```text
FIMO sequence background = all valid cCREs in that cell type
enrichment target         = significant valid cCREs
enrichment comparison     = non-significant valid cCREs
```

The significance label is never used to estimate FIMO's background, preventing
the sequence null model from being fitted to only one side of the enrichment
comparison.

## HOCOMOCO motif space

The workflow uses the current HOCOMOCO v14 CORE collection:

```text
resources/H14CORE_meme_format.meme
resources/H14CORE-MOUSE_annotation.jsonl
```

The mouse annotation contains 1,245 motif models representing 809 mouse TF
genes, including gene symbol, synonyms, MGI, Entrez, UniProt, and TFClass
metadata. Source URLs and licensing are recorded in `resources/README.md`.

For every cell type, FIMO scans the actual ~200-bp cloned insert sequences from:

```text
STARRFISH_in_vivo/results/cre_info.csv
```

This is preferable to scanning an arbitrary genomic window around the cCRE.
The same cCRE can have slightly different FIMO scores between cell types
because each cell type has its own all-valid-cCRE nucleotide background.

FIMO converts each motif-position match to a log-odds score using the HOCOMOCO
PWM and that cell type's zero-order A/C/G/T background. Its p-value is the
probability, under that background model, of a random sequence receiving a
score at least as high as the observed score; FIMO computes the score-tail
distribution by dynamic programming. A motif is called present when at least
one position on either strand has `p <= 1e-4`. The `1e-4` value is the
preselected per-site FIMO reporting threshold, not an adjusted p-value.
Downstream motif enrichment has its own Fisher test and BH corrections. The
stored cell-type/cCRE/motif score is `-log10(best FIMO p-value)`.

Alternative HOCOMOCO models for the same mouse TF are also collapsed into a
gene-level any-hit matrix. Therefore the results include both:

- motif-model tests, which preserve alternative binding models;
- TF-gene tests, where a cCRE is positive if any HOCOMOCO model mapped to that
  mouse gene is present.

## Cell-type × motif activity matrix

The workflow also calculates an activity matrix using **all valid cCREs that
passed `target_t7_total >= 50`**, without filtering on significance. For cCRE
`i`, cell type `c`, and motif `m`:

```text
cCRE effect(i,c) = effect_vs_mean_control_mean
motif score(i,c,m) = -log10(best cell-type-specific FIMO p)
                      or 0 when the motif is absent
motif contribution(i,c,m) = cCRE effect(i,c) * motif score(i,c,m)
```

`effect_vs_mean_control_mean` is the posterior mean cCRE activity minus the
draw-wise mean activity of the ordinary negative controls. Negative effects
are retained, so negative matrix entries indicate motif-bearing cCREs with
activity below the negative-control mean.

The primary cell-type × motif matrix is the mean contribution across all valid
cCREs in that cell type:

```text
motif activity(c,m) = mean_i motif contribution(i,c,m)
```

The mean is primary because the number of valid cCREs differs across cell
types. A summed-contribution matrix is also exported, along with occurrence
fractions, hit counts, and matching-score summaries. This activity matrix is a
descriptive aggregation; it is separate from the significant-versus-
non-significant Fisher enrichment test.

For direct comparison of relative motif patterns between cell types, each row
of the primary matrix is also standardized across all 1,206 detected motif
models:

```text
z(c,m) = [motif activity(c,m) - mean across motifs in cell type c]
         / population SD across motifs in cell type c
```

Consequently, every cell-type row in the z-score matrix has mean 0 and
population standard deviation 1. Positive values identify motifs above that
cell type's overall motif-activity level; they do not by themselves imply
positive raw activity or statistical significance. In the activity heatmaps,
cell types with at least 20 valid cCREs are rows ordered by their original
numbered h5ad subclass prefix. This retains 43 cell types and excludes only
`LSX Nkx2-1 Gaba` (16 valid cCREs). The full matrices still retain all 44 cell
types. The selected motif columns are ordered by average-linkage hierarchical
clustering with correlation distance and displayed with a column dendrogram.
Motif selection for visualization is rank-based rather than a statistical
threshold: the default figure includes the 100 motifs with the largest standard
deviation across the retained cell types in the matrix being plotted. In the
z-score matrix, these 100 motifs account for 31.9% of total across-cell-type
motif variance and represent 89 unique TF genes. The complete matrix retains all
1,206 detected motif models.

## cCRE activity–motif score correlations

As a separate analysis, the workflow correlates cCRE activity with motif
matching strength within each cell type. For every HOCOMOCO motif it uses all
finite-q cCREs passing `target_t7_total >= 50`:

```text
y(i,c) = effect_vs_mean_control_mean
x(i,c,m) = -log10(best cell-type-specific FIMO p)
             or 0 when the motif is absent
```

Two associations are calculated:

- Pearson `r`, measuring a linear relationship between activity and matching
  score;
- Spearman `rho`, measuring a monotonic relationship after average-ranking
  both values, including tied zero motif scores.

A positive correlation means that cCREs with stronger motif matches tend to
have higher activity relative to the negative-control mean. A negative
correlation means that stronger matches tend to occur in cCREs with lower
activity. Correlations are undefined when a motif has no score variance in a
cell type, most commonly because it has no qualifying FIMO hits.

Two-sided correlation p-values are BH-adjusted across all nonconstant motifs
separately within each cell type. The output also retains the number of valid
cCREs and motif-positive cCREs, which should be consulted because a strong
correlation supported by very few motif hits can be unstable.

Pearson and Spearman have separate heatmaps. Each displays the 100 motifs with
the largest maximum absolute correlation across the retained cell types.
Rows are cell types with at least 20 valid cCREs, ordered by numbered prefix;
motif columns are clustered independently in each heatmap. Undefined
correlations are gray. The complete exported matrices retain every motif and
all cell types.

## TF expression link

TF expression is aggregated from:

```text
revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad
```

The h5ad subclass labels are standardized exactly as in the Bayesian workflow.
For every HOCOMOCO TF gene present in the 500-gene panel, the output reports:

- `tf_expression_mean`: mean stored `X` value in the cell type;
- `tf_expression_fraction_detected`: fraction of cells with `X > 0`;
- `tf_expression_n_cells`: number of cells in the aggregate;
- `tf_gene_in_expression_panel`: whether the TF was measured.

The panel measures 78 of the 809 HOCOMOCO mouse TF genes. Missing expression is
therefore reported explicitly and is not interpreted as zero expression.

## Statistical comparison

For each cell type and motif (or collapsed TF gene), the workflow builds:

```text
                        motif present   motif absent
significant cCREs             a              b
non-significant cCREs         c              d
```

It runs a one-sided Fisher exact test (`alternative="greater"`). The output
reports motif frequencies, their difference, enrichment and odds ratios,
motif-score means, and the significant cCREs carrying each motif.

- `q_value_cell_type`: BH correction over motifs within one cell type.
- `q_value_global`: BH correction over all cell-type/motif comparisons.

The within-cell-type FDR is the primary result. The global value is a stricter
study-wide reference. HOCOMOCO motifs are correlated, so the TF-gene collapsed
analysis is useful alongside the full model-level results.

## Run

Locally:

```bash
/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python \
  revision/TFMotif/code/run_tf_motif_analysis.py
```

With Slurm:

```bash
sbatch revision/TFMotif/code/submit_tf_motif_analysis.slurm
```

The FIMO scan is cached with hashes of the exact cell-type/cCRE membership,
sequences, motif database, annotation, threshold, and background definition.
Use `--force-rescan` to rebuild it. `--fimo-jobs` controls how many independent
cell-type scans run concurrently (default 2).

Focused tests:

```bash
/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python \
  -m pytest revision/TFMotif/tests
```

## Outputs

Outputs are under `results/`.

- `run_manifest.json`: input hashes, definitions, parameters, and counts.
- `tables/ccre_sets_by_cell_type.csv`: every eligible cCRE and its significant
  status.
- `tables/motif_enrichment_by_cell_type.csv.gz`: all HOCOMOCO motif-model
  Fisher comparisons with mouse TF annotation and expression.
- `tables/top_motifs_by_cell_type.csv`: ten lowest-q motif models per cell type.
- `tables/tf_gene_motif_enrichment_by_cell_type.csv.gz`: any-HOCOMOCO-model
  enrichment collapsed to mouse TF gene.
- `tables/top_tf_genes_by_cell_type.csv`: ten lowest-q TF genes per cell type.
- `tables/top_expression_supported_tf_genes_by_cell_type.csv`: top TF-gene
  comparisons restricted to TFs detected in the 500-gene expression panel.
- `tables/hocomoco_motif_to_mouse_gene.csv.gz`: complete motif-to-gene mapping.
- `tables/hocomoco_mouse_tf_gene_metadata.csv.gz`: collapsed TF-gene metadata
  and the contributing HOCOMOCO motif IDs.
- `tables/hocomoco_tf_expression_by_cell_type.csv.gz`: mean expression and
  detection fraction for the 78 measured HOCOMOCO TF genes.
- `tables/fimo_background_by_cell_type.csv`: all-valid-cCRE counts, sequence
  base counts, A/C/G/T frequencies, and FASTA/background file paths for all 44
  cell types.
- `tables/cell_type_by_motif_activity.csv.gz`: primary 44 cell-type × 1,206
  motif matrix, containing the mean effect-weighted motif contribution across
  all valid T7>=50 cCREs.
- `tables/cell_type_by_motif_activity_sum.csv.gz`: summed rather than mean
  effect-weighted contributions.
- `tables/cell_type_by_motif_activity_zscore.csv.gz`: the primary activity
  matrix z-scored across all motifs separately within every cell type.
- `tables/cell_type_by_motif_occurrence_fraction.csv.gz` and
  `tables/cell_type_by_motif_hit_count.csv.gz`: motif occurrence among all
  valid cCREs.
- `tables/cell_type_by_motif_matching_score_mean_{all,present}.csv.gz`: mean
  FIMO score across all valid cCREs or only motif-positive cCREs.
- `tables/motif_activity_by_cell_type_long.csv.gz`: all matrix metrics in long
  form with HOCOMOCO motif-to-gene annotation and matched TF expression.
- `tables/cell_type_by_motif_activity_pearson_{r,p,q_cell_type}.csv.gz`:
  Pearson correlation, two-sided p-value, and within-cell-type BH FDR matrices.
- `tables/cell_type_by_motif_activity_spearman_{rho,p,q_cell_type}.csv.gz`:
  Spearman correlation, two-sided p-value, and within-cell-type BH FDR
  matrices.
- `tables/motif_activity_correlation_by_cell_type_long.csv.gz`: Pearson and
  Spearman results, motif hit counts, HOCOMOCO gene annotation, and matched TF
  expression in one long table.
- `tables/cell_type_number_prefixes.csv`: mapping between cleaned Bayesian cell
  types and the original numbered h5ad labels used for heatmap row order.
- `tables/hocomoco_fimo_best_hits.csv.gz`: best hit for every detected
  cell-type/cCRE/motif combination, including position, strand, sequence, and
  TF annotation.
- `tables/significant_ccre_motif_profiles.csv.gz`: wide HOCOMOCO score profiles
  for all significant cCRE–cell-type pairs.
- `tables/significant_ccre_motifs_long.csv.gz`: one row per motif present in a
  significant cCRE with Bayesian effect, q-value, HOCOMOCO gene mapping, and TF
  expression.
- `figures/top_motif_enrichments.{png,pdf}`: strongest motif comparisons.
- `figures/weighted_motif_activity_heatmap.{png,pdf}`: the 100 motifs whose
  mean weighted activities vary most across cell types.
- `figures/weighted_motif_activity_zscore_heatmap.{png,pdf}`: the corresponding
  view based on within-cell-type motif z-scores, with numbered cell types as
  rows and hierarchically clustered motifs as columns.
- `figures/motif_activity_pearson_correlation_heatmap.{png,pdf}`: cell type ×
  motif Pearson correlations for the 100 strongest motif associations.
- `figures/motif_activity_spearman_correlation_heatmap.{png,pdf}`: the
  corresponding Spearman rank-correlation view.
- `figures/motif_enrichment_heatmap.{png,pdf}`: positive, within-cell-type
  FDR-significant frequency differences (written only when such results exist).

Sparse cell types can yield large odds ratios from one or two cCREs. Interpret
effect sizes with the target/background counts and FDR. Motif presence also
does not prove TF occupancy; matched TF expression improves prioritization but
is still supporting evidence rather than a binding assay.

## Current default result

The default run analyzed 3,401 eligible cCRE–cell-type pairs in 44 cell types,
including 655 significant pairs in 43 cell types. It estimated 44 FIMO
backgrounds from all 3,401 valid pairs; A/T frequencies ranged from 0.2805 to
0.2875 and C/G frequencies from 0.2125 to 0.2195. FIMO found 212,148 best
cell-type/cCRE/motif hits spanning 1,206 HOCOMOCO mouse motif models (788 TF
genes).

The weighted motif-activity output is a 44 × 1,206 matrix built from all 3,401
valid T7>=50 cCRE–cell-type pairs. Mean motif activities range from -2.641 to
1.214. These signed values are descriptive effect-weighted summaries, not
p-values or formal enrichment calls.

The Pearson and Spearman correlation outputs are also 44 × 1,206. A
correlation is defined for 45,225 of the 53,064 cell-type/motif combinations;
the remainder have constant motif scores. Pearson correlations range from
-0.619 to 0.677 and Spearman correlations range from -0.587 to 0.599. There
are 48 Pearson associations at within-cell-type FDR <= 0.05, but 40 are
supported by only one motif-positive cCRE. No Spearman association passes 5%
within-cell-type FDR. Therefore the Pearson FDR calls should not be interpreted
without their motif hit counts. The strongest Pearson result with broader
support is `FOXH1.H14CORE.0.P.B` in `L2-3 IT CTX Glut` (`r=0.449`, 7 of 119
cCREs with the motif, raw `p=3.05e-7`, within-cell-type `q=3.54e-4`); `Foxh1`
is not measured in the 500-gene expression panel.

No motif-model or TF-gene comparison passed within-cell-type or global 5% FDR.
Accordingly, the tables and top plot should be read as exploratory rankings,
not significant motif discoveries.

The smallest motif-model p-values were:

| Cell type | HOCOMOCO motif | Mouse TF | Significant with motif | Non-significant with motif | Raw p | Within-cell-type q |
|---|---|---|---:|---:|---:|---:|
| L4-5 IT CTX Glut | FOXH1.H14CORE.0.P.B | Foxh1 | 5/16 | 1/105 | 1.21e-4 | 0.146 |
| L5 ET CTX Glut | NR1D1.H14CORE.1.SM.B | Nr1d1 | 11/67 | 2/110 | 4.90e-4 | 0.591 |

Neither `Nr1d1` nor `Foxh1` is measured in the 500-gene panel. Examples of
expression-supported, gene-collapsed leads include `Rorb` in CBX Purkinje Gaba
(2/6 significant versus 0/51 non-significant cCREs, raw p=0.0094; expression
detected in 37.7% of cells) and `Tcf7l2` in L6 IT CTX Glut (7/19 versus 6/55,
raw p=0.0168; expression detected in 9.0% of cells). Their adjusted q-values
are 1.0, so these are candidates for follow-up rather than formal calls.
