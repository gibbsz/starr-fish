import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor']='k'

def plot_gene_scdata(scdata2,gene='SOX9',nmax=20,sz_min=5,sz_max=30,transpose=1,flipx=1,flipy=1,tag='X_spatial'):
    Xcells = scdata2.obsm[tag][:,::transpose]*[flipx,flipy]
    ign = list(scdata2.var.index).index(gene)
    #scdata2.obsm['X_umap']
    if 'X_raw' not in scdata2.obsm:
        Xnorm = (np.exp(scdata2.X)-1)
        ncts = np.sum(Xnorm,axis=1)[0]
        scdata2.obsm['X_raw']=np.round(Xnorm/ncts*np.array(scdata2.obs['total_counts'])[:,np.newaxis])
    cts = scdata2.obsm['X_raw'][:,ign].copy()
    plt.style.use("dark_background")
    cts[np.isnan(cts)]=0
    #cts[cts>20]=0
    ncts = np.clip(cts/nmax,0,1)
    size = sz_min+ncts*(sz_max-sz_min)
    from matplotlib import cm as cmap
    #cols = cmap.coolwarm(ncts)
    cols = cmap.coolwarm(ncts)

    good_cells = slice(None)
    good_cells = np.argsort(cts)

    #blanks = [gn for gn in df.columns if 'blank' in gn]
    #blanks_cts = np.nanmean(df[blanks],axis=-1)
    #good_cells = blanks_cts<th_blank

    XC = -Xcells[good_cells,::-1]
    #viewer = napari.view_points(XC,size=size[good_cells],face_color=cols[good_cells],name=gene)
    fig = plt.figure(facecolor='k',figsize=(15,15))
    plt.title(gene+' - N max '+str(nmax))
    fig.set_facecolor('black')
    plt.scatter(XC[:,0],XC[:,1],c=cols[good_cells],s=size[good_cells])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.tight_layout()
    return fig

def plot_cluster_scdata(scdata,clusters=[1,2],transpose=1,flipx=1,flipy=1,sbig=30,small=5,plot_legend = False):
    import matplotlib.pyplot as plt
    cmap = scdata.uns['cmap']
    fig = plt.figure(figsize=(10, 10), facecolor="black")

    from matplotlib import pylab as plt
    x,y = (scdata.obsm['X_spatial']*[-flipx,-flipy])[:,::-transpose].T
    
    #np.unique(scdata.obs["leiden"].astype(np.int))[::-1]
    plt.scatter(x, y, c='gray', s=small, marker='.')
    for cluster in clusters:
        cluster_ = str(cluster)
        inds = scdata.obs["leiden"] == cluster_
        x_ = x[inds]
        y_ = y[inds]
        col = cmap[int(cluster) % len(cmap)]
        plt.scatter(x_, y_, c=col, s=sbig, marker='.',label = cluster_)
    
    plt.grid(False)
    plt.axis("off")
    plt.axis("equal")
    if plot_legend:
        plt.legend()
    plt.tight_layout()
    return fig

def plot_CRE_scdata(scdata2,CRE='SOX9',nmax=20,sz_min=5,sz_max=30,transpose=1,flipx=1,flipy=1,tag='X_spatial'):
    Xcells = scdata2.obsm[tag][:,::transpose]*[flipx,flipy]
    ign = list(scdata.obsm['CRE'][CRE])
    #scdata2.obsm['X_umap']
    # if 'X_raw' not in scdata.obsm:
    #     Xnorm = (np.exp(scdata.X)-1)
    #     ncts = np.sum(Xnorm,axis=1)[0]
    #     scdata.obsm['X_raw']=np.round(Xnorm/ncts*np.array(scdata.obs['total_counts'])[:,np.newaxis])
    cts = np.array(ign).copy()
    plt.style.use("dark_background")
    cts[np.isnan(cts)]=0
    #cts[cts>20]=0
    ncts = np.clip(cts/nmax,0,1)
    size = sz_min+ncts*(sz_max-sz_min)
    from matplotlib import cm as cmap
    #cols = cmap.coolwarm(ncts)
    cols = cmap.coolwarm(ncts)

    good_cells = slice(None)
    good_cells = np.argsort(cts, axis=0)

    #blanks = [gn for gn in df.columns if 'blank' in gn]
    #blanks_cts = np.nanmean(df[blanks],axis=-1)
    #good_cells = blanks_cts<th_blank

    XC = -Xcells[good_cells,::-1]
    #viewer = napari.view_points(XC,size=size[good_cells],face_color=cols[good_cells],name=gene)
    fig = plt.figure(facecolor='k',figsize=(15,15))
    plt.title(CRE+' - N max '+str(nmax))
    fig.set_facecolor('black')
    plt.scatter(XC[:,0],XC[:,1],c=cols[good_cells],s=size[good_cells])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.tight_layout()

def plot_cell_subclass(scdata, subclass='DG Glut', is_class=False):
    # Define coordinates for all cells and those of the subclass
    X = scdata.obsm['X_spatial']
    
    if is_class == True:
        X_ = X[scdata.obs['class'] == subclass]
    
    else:
        X_ = X[scdata.obs['subclass'] == subclass]

    # Plot the data
    plt.style.use("dark_background")
    fig, ax = plt.subplots(facecolor='k', figsize=(10,7))
    ax.scatter(X[:,0],X[:,1],s=0.05,c='gray', marker='.', alpha=0.50)
    ax.scatter(X_[:,0],X_[:,1],s=0.5,c='#00FF00', marker='.')
    ax.set_facecolor('k')
    ax.axis('off')
    plt.axis('equal')
    plt.title(str(subclass), c='w')

    plt.tight_layout()
    plt.show()

def plot_multiple_CREs(scdata, CREs=['SOX9', 'CRE2', 'CRE3'], nmax=20, sz_min=5, sz_max=30, transpose=1, flipx=1, flipy=1, tag='X_spatial'):
    Xcells = scdata.obsm[tag][:, ::transpose] * [flipx, flipy]
    
    # Initialize the figure
    fig = plt.figure(facecolor='k', figsize=(15, 15))
    fig.set_facecolor('black')
    
    # Iterate over the list of CREs
    for CRE in CREs:
        ign = list(scdata.obsm['CRE'][CRE])  # Get the CRE expression values
        cts = np.array(ign).copy()
        cts[np.isnan(cts)] = 0
        
        # Normalize and scale the size of the points
        ncts = np.clip(cts / nmax, 0, 1)
        size = sz_min + ncts * (sz_max - sz_min)
        
        # Use a color map for the points
        cols = cmap.coolwarm(ncts)
        
        # Sorting cells based on expression levels (optional)
        good_cells = np.argsort(cts, axis=0)
        
        # Plot the CRE data
        XC = -Xcells[good_cells, ::-1]
        
        # Plot the points for the current CRE with a label for identification
        plt.scatter(XC[:, 0], XC[:, 1], c=cols[good_cells], s=size[good_cells], label=CRE)
    
    # Final plot adjustments
    plt.title('Multiple CREs - N max ' + str(nmax))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.tight_layout()
    plt.legend()  # Add a legend to differentiate the CREs
    
    return fig
    

def plot_multiple_CREv2(scdata, CREs=['SOX9', 'CRE2', 'CRE3'], mincounts=0, nmax=20, sz_min=5, sz_max=30, transpose=1, flipx=1, flipy=1, tag='X_spatial'):
    Xcells = scdata.obsm[tag][:, ::transpose] * [flipx, flipy]

    # Initialize the figure
    fig = plt.figure(facecolor='k', figsize=(12, 12))
    fig.set_facecolor('black')

    # Plot all cells as background
    XM = -Xcells
    plt.scatter(XM[:,0],XM[:,1],s=0.2,c='gray', marker='.', alpha=0.2)
    
    # Iterate over the list of CREs
    for idx, CRE in enumerate(CREs):
        ign = list(scdata.obsm['CRE'][CRE])  # Get the CRE expression values
        cts = np.array(ign).copy()
        cts[np.isnan(cts)] = 0
        
        # Normalize and scale the size of the points
        ncts = np.clip(cts / nmax, 0, 1)
        size = sz_min + ncts * (sz_max - sz_min)
        
        # generate a unique color for each CRE
        cols = gl.create_palette(palette_size=len(CREs))
        
        # Get the indices where cts > mincounts
        good_cells = np.where(cts > mincounts)[0]  # Extracting the indices from the tuple
        
        # Plot the CRE data
        XC = -Xcells[good_cells]  # Extracting the corresponding cells for plotting
        
        # Select the color from the colormap based on the index
        color = cols[idx % len(cols)]
        
        # Plot the points for the current CRE with a label for identification
        plt.scatter(XC[:, 0], XC[:, 1], c=[color] * len(XC), s=size[good_cells], label=CRE)
        
    
    # Final plot adjustments
    plt.title('Multiple CREs - minimum count per cell of ' + str(mincounts), c='w')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.tight_layout()
    plt.legend(loc='upper right', ncols=len(CREs)/5, labelcolor='w')  # Add a legend to differentiate the CREs
    plt.show()