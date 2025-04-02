import pandas as pd
import numpy as np
CREs = ['CRE001', 'CRE002', 'CRE003', 'CRE004', 'CRE005', 'CRE006', 'CRE007', 'CRE008',
        'CRE009', 'CRE010', 'CRE011', 'CRE012', 'CRE013', 'CRE014', 'CRE015', 'CRE016',
        'CRE017', 'CRE018', 'CRE019', 'CRE020']

def process_cbgs(enhancer_cbg =r'C:\Users\zgibbs\cellpose\cell_by_gene\SFv4_T7_July_enhancer_cbg.csv',
                 T7_cbg = r'C:\Users\zgibbs\cellpose\cell_by_gene\SFv4_T7_July_T7_cbg.csv',
                 filter_t7 = 49, 
                 
                ):
    # load cbgs
    enhancer_cbg = pd.read_csv(enhancer_cbg)
    T7_cbg = pd.read_csv(T7_cbg)

    T7_filt_cbg = T7_cbg[T7_cbg['total transcripts']>49]
    #merge cbgs
    merged = pd.merge(T7_filt_cbg, enhancer_cbg, how='left', left_on='masks', right_on='masks')


    for cre in CREs:
        idx = str(cre + '_x')
        idy = str(cre + '_y')
        merged[cre] = merged[idy] / merged[idx]
        # fill inf and nan with zeroes
    
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.fillna(0)

    # T7 matrix
    T7_m = merged.iloc[:,1:21]
    T7_m.columns = CREs
    T7_m['total transcripts'] = T7_m.sum(axis=1)
    T7_m['masks'] = merged['masks']
    T7_m['fov'] = pd.cut(merged.masks.astype(int), bins = np.arange(0, 226)*1000, right=False)
    
    # enhancer matrix
    CRE_m = merged.iloc[:,23:43]
    CRE_m.columns = CREs
    CRE_m['total transcripts'] = CRE_m.sum(axis=1)
    CRE_m['masks'] = merged['masks']
    CRE_m['fov'] = pd.cut(merged.masks.astype(int), bins = np.arange(0, 226)*1000, right=False)
    
    # T7-normalized matrix
    norm_m = merged.iloc[:,-20:].fillna(0)
    norm_m.columns = CREs
    norm_m['total transcripts'] = norm_m.sum(axis=1)
    norm_m['masks'] = merged['masks']
    norm_m['fov'] = pd.cut(merged.masks.astype(int), bins = np.arange(0, 226)*1000, right=False)

    return T7_m, CRE_m, norm_m

