import pandas as pd
import snapatac2 as sa2

# --- Load input files ---
print("Loading AnnData...")
ann = sa2.read('2024701.brain.snapatac2.h3k27ac.h5ad', backed=None)  # NO backed='r'
print("AnnData loaded into memory.")

abc_anno = pd.read_csv('cluster_annotation_term.csv', index_col=0)
starrfish_counts = pd.read_csv('expr1_expr2_counts.csv', index_col=0)
peaks = pd.read_csv('cre_info.csv', index_col=0)

# --- Preprocess annotation info ---
abc_anno['subclass'] = abc_anno['subclass'].str.replace('/', '-')
subclass_map = abc_anno.groupby('subclass_number')['subclass'].first()

# --- Work with obs directly ---
obs = ann.obs
obs['subclass_number'] = obs['subclass'].str[:3].astype(int)
obs['subclass_name'] = obs['subclass_number'].map(subclass_map)

# --- Determine cells and peaks to keep ---
cell_mask = obs['subclass_name'].isin(starrfish_counts.index)
peak_mask = ann.var_names.isin(peaks['enh'])

# --- Subset AnnData ---
print("Subsetting AnnData...")
ann_subset = ann[cell_mask.values, peak_mask].copy()  # .copy() loads subset into memory

# optional: free up memory
del ann

# --- Add subclass_name ---
ann_subset.obs['subclass_name'] = obs.loc[cell_mask, 'subclass_name'].values

# --- Save subset ---
print("Saving subset...")
ann_subset.write('brain.snapatac2.H3K27ac.starrfish_subset.h5ad')
print("Done.")

