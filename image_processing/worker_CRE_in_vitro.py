master_analysis_folder = r'path_to_analysis_folder'
lib_fl = r'path_to_codebook.csv' ### codebook file
### Did you compute PSF and median flat field images?
psf_file = r'path_to_psf_file.npy'
master_data_folder = [r'path_to_data_folder'] # raw image data folder
save_folder =r'path_to_save_folder'  ### where to save processed data
flat_field_tag = r'path_to_flat_field_file_tag'
iHm=1 #H iHmin -> H iHmax oly keeps folders of the the form H33,H34...
iHM=16

from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)

from ioMicro import *

def get_intersV2(self,nmin_bits=4,dinstance_th=2,enforce_color=True,enforce_set=None,redo=False):
    """Get an initial intersection of points and save in self.res"""
    self.res_fl = self.decoded_fl.replace('decoded','res')
    if not os.path.exists(self.res_fl) or redo:
        
        res =[]
        if enforce_color and (enforce_set is None):
            icols = self.XH[:,-2].astype(int)
            XH = self.XH
            for icol in tqdm(np.unique(icols)):
                inds = np.where(icols==icol)[0]
                Xs = XH[inds,:3]
                Ts = cKDTree(Xs)
                res_ = Ts.query_ball_tree(Ts,dinstance_th)
                res += [inds[r] for r in res_]
        elif enforce_color and (enforce_set is not None):
            ibits = self.XH[:,-1].astype(int)
            isets = ibits//enforce_set
            icols = self.XH[:,-2].astype(int)
            XH = self.XH
            for icol in np.unique(icols):
                for iset in tqdm(np.unique(isets)):
                    inds = np.where((icols==icol)&(isets==iset))[0]
                    Xs = XH[inds,:3]
                    Ts = cKDTree(Xs)
                    res_ = Ts.query_ball_tree(Ts,dinstance_th)
                    res += [inds[r] for r in res_]
        else:
            XH = self.XH
            Xs = XH[:,:3]
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
        print("Calculating lengths of clusters...")
        lens = np.array(list(map(len,res)))
        Mlen = np.max(lens)
        print("Unfolding indexes...")
        res_unfolder = np.concatenate(res)
        print("Saving to file:",self.res_fl)
        self.res_unfolder=res_unfolder
        self.lens=lens
        
        #np.savez(self.res_fl,res_unfolder=res_unfolder,lens=lens)
    else:
        dic = np.load(self.res_fl)
        self.res_unfolder=dic['res_unfolder']
        self.lens=dic['lens']
        #self.res = res
    lens =self.lens
    self.res_unfolder = self.res_unfolder[np.repeat(lens, lens)>=nmin_bits]
    self.lens = self.lens[lens>=nmin_bits]

def get_icodesV3(dec,nmin_bits=3,iH=-3):
    import time
    start = time.time()
    lens = dec.lens
    res_unfolder = dec.res_unfolder
    Mlen = np.max(lens)
    print("Calculating indexes within cluster...")
    res_is = np.tile(np.arange(Mlen), len(lens))
    res_is = res_is[res_is < np.repeat(lens, Mlen)]
    print("Calculating index of molecule...")
    ires = np.repeat(np.arange(len(lens)), lens)
    #r0 = np.array([r[0] for r in res for r_ in r])
    print("Calculating index of first molecule...")
    r0i = np.concatenate([[0],np.cumsum(lens)])[:-1]
    r0 = res_unfolder[np.repeat(r0i, lens)]
    print("Total time unfolded molecules:",time.time()-start)
    
    ### torch
    ires = torch.from_numpy(ires.astype(np.int64))
    res_unfolder = torch.from_numpy(res_unfolder.astype(np.int64))
    res_is = torch.from_numpy(res_is.astype(np.int64))
    
    import time
    start = time.time()
    print("Computing score...")
    scoreF = torch.from_numpy(dec.XH[:,iH])[res_unfolder]
    print("Total time computing score:",time.time()-start)
    
    
    ### organize molecules in blocks for each cluster
    def get_asort_scores():
        val = torch.max(scoreF)+2
        scoreClu = torch.zeros([len(lens),Mlen],dtype=torch.float64)+val
        scoreClu[ires,res_is]=scoreF
        asort = scoreClu.argsort(-1)
        scoreClu = torch.gather(scoreClu,dim=-1,index=asort)
        scoresF2 = scoreClu[scoreClu<val-1]
        return asort,scoresF2
    def get_reorder(x,val=-1):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(np.array(x))
        xClu = torch.zeros([len(lens),Mlen],dtype=x.dtype)+val
        xClu[ires,res_is] = x
        xClu = torch.gather(xClu,dim=-1,index=asort)
        xf = xClu[xClu>val]
        return xf
    
    
    import time
    start = time.time()
    print("Computing sorting...")
    asort,scoresF2 = get_asort_scores()
    res_unfolder2 = get_reorder(res_unfolder,val=-1)
    del asort
    del scoreF
    print("Total time sorting molecules by score:",time.time()-start)
    
    
    
    import time
    start = time.time()
    print("Finding best bits per molecules...")
    
    Rs = dec.XH[:,-1].astype(np.int64)
    Rs = torch.from_numpy(Rs)
    Rs_U = Rs[res_unfolder2]
    nregs,nbits = dec.codes_01.shape
    score_bits = torch.zeros([len(lens),nbits],dtype=scoresF2.dtype)-1
    score_bits[ires,Rs_U]=scoresF2
    
    
    codes_lib = torch.from_numpy(np.array(dec.codes__))
    
    
    codes_lib_01 = torch.zeros([len(codes_lib),nbits],dtype=score_bits.dtype)
    for icd,cd in enumerate(codes_lib):
        codes_lib_01[icd,cd]=1
    codes_lib_01 = codes_lib_01/torch.norm(codes_lib_01,dim=-1)[:,np.newaxis]
    print("Finding best code...")
    batch = 10000
    icodes_best = torch.zeros(len(score_bits),dtype=torch.int64)
    dists_best = torch.zeros(len(score_bits),dtype=torch.float32)
    from tqdm import tqdm
    for i in tqdm(range((len(score_bits)//batch)+1)):
        score_bits_ = score_bits[i*batch:(i+1)*batch]
        if len(score_bits_)>0:
            score_bits__ = score_bits_.clone()
            score_bits__[score_bits__==-1]=0
            score_bits__ = score_bits__/torch.norm(score_bits__,dim=-1)[:,np.newaxis]
            Mul = torch.matmul(score_bits__,codes_lib_01.T)
            max_ = torch.max(Mul,dim=-1)
            icodes_best[i*batch:(i+1)*batch] = max_.indices
            dists_best[i*batch:(i+1)*batch] = 2-2*max_.values
    
    
    keep_all_bits = torch.sum(score_bits.gather(1,codes_lib[icodes_best])>=0,-1)>=nmin_bits
    dists_best_ = dists_best[keep_all_bits]
    score_bits = score_bits[keep_all_bits]
    icodes_best_ = icodes_best[keep_all_bits]
    icodesN=icodes_best_
    
    indexMols_ = torch.zeros([len(lens),nbits],dtype=res_unfolder2.dtype)-1
    indexMols_[ires,Rs_U]=res_unfolder2
    indexMols_ = indexMols_[keep_all_bits]
    indexMols_ = indexMols_.gather(1,codes_lib[icodes_best_])
    
    # make unique
    indexMols_,rinvMols = get_unique_ordered(indexMols_)
    icodesN = icodesN[rinvMols]
    
    XH = torch.from_numpy(dec.XH)
    XH_pruned = XH[indexMols_]
    XH_pruned[indexMols_==-1]=np.nan
    
    dec.dist_best = dists_best_[rinvMols].numpy()
    dec.XH_pruned=XH_pruned.numpy()
    dec.icodesN=icodesN.numpy()
    np.savez_compressed(dec.decoded_fl,XH_pruned=dec.XH_pruned,icodesN=dec.icodesN,gns_names = np.array(dec.gns_names),dist_best=dec.dist_best)
    print("Total time best bits per molecule:",time.time()-start)

def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file)
                    
def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj.Xh_minus)>0) and  (len(obj_ref.Xh_plus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)

        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
        
        tzxyf = -(tzxy_plus+tzxy_minus)/2
        
    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]

def compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        fl_ref = fls[len(fls)//2]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))

def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method
        im_n = norm_slice(im__,s=30)
        #Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        ### new method
        fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'
        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
            im_med = cv2.blur(im_med,(20,20))
            im__ = im__/im_med*np.median(im_med)
        else:
            print("Did not find flat field")
        try:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
        except:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    np.savez_compressed(save_fl,Xh=Xh)

def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    psf = np.load(psf_file,allow_pickle=True)
    
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            if not os.path.exists(save_fl) or redo:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)  
                                    
def compute_decoding(save_folder,fov,set_,redo=False):
    dec = decoder_simple(save_folder,fov,set_)
    complete = dec.check_is_complete()
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        dec.get_XH(fov,set_,ncols=3,nbits=48,th_h=3600,tag_keep='H')#number of colors match 
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        get_intersV2(dec,nmin_bits=3,dinstance_th=2,enforce_color=True,enforce_set=None,redo=False)
        get_icodesV3(dec,nmin_bits=3,iH=-3)
        #dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        #dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=True,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        #get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=24,is_unique=True)

def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf

def get_files(set_ifov,iHm=iHm,iHM=iHM):
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    all_flds = []
    for master_folder in master_data_folder:

        all_flds += glob.glob(master_folder+r'\H*')
    
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    set_,ifov = set_ifov
    all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
    
    all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
    
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        folder_map_fovs = all_flds[0]#[fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
        fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
        fovs = np.sort([os.path.basename(fl) for fl in fls])
        np.save(fovs_fl,fovs)
    else:
        fovs = np.sort(np.load(fovs_fl))
    fov=None
    if ifov<len(fovs):
        fov = fovs[ifov]
        all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
    return save_folder,all_flds,fov

############### New Code inserted here!!! ##################
def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = None,#flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file)
                    
def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5,th=4): 
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus#[:,:3]
        X_ref = obj_ref.Xh_plus#[:,:3]
        X = X[X[:,-1]>th][:,:3]
        X_ref = X_ref[X_ref[:,-1]>th][:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus#[:,:3]
        X_ref = obj_ref.Xh_minus#[:,:3]
        X = X[X[:,-1]>th][:,:3]
        X_ref = X_ref[X_ref[:,-1]>th][:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]

def compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        fl_ref = fls[len(fls)//2]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))

def compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method):
    print("Computing fitting on: "+str(fov))
    print(len(all_flds),all_flds)
    compute_fits(save_folder,fov,all_flds,redo=redo_fits,try_mode=try_mode,old_method=old_method)
    print("Computing drift on: "+str(fov))
    #compute_drift(save_folder,fov,all_flds,set_,redo=redo_drift)
    compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True)
    compute_drift_V2(save_folder,fov,all_flds,set_,redo=redo_drift,gpu=True)
    compute_decoding(save_folder,fov,set_,redo=redo_decoding)

    
############### End Code inserted here!!! ##################

def main_do_compute_fits_test(save_folder,fld,fov,icol,psf):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)

    ### new method
    fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'
    if os.path.exists(fl_med):
        im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
        im_med = cv2.blur(im_med,(20,20))
        im__ = im__/im_med*np.median(im_med)
    else:
        print("Did not find flat field")
    try:
        Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                deconv={'method':'wiener','beta':0.0001},
                                delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    except:
        Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                deconv={'method':'wiener','beta':0.0001},
                                delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    return Xh

def get_fr(save_folder,all_flds,fov,verbose=False):
    psf = np.load(psf_file)
    
    import time
    start = time.time()
    fld = all_flds[-1]
    #fld = fld.replace('192.168.0.129','192.168.0.130')
    icol=0
    
    
    try:
        save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+os.path.basename(fld)+'--col'+str(icol)+'__Xhfits.npz'
        nels = len(np.load(save_fl)['Xh'])
        Xh = main_do_compute_fits_test(save_folder,fld,fov,icol,psf)
        fr = nels/len(Xh)
    except:
        fr = 0
    if verbose: print("time:",time.time()-start)
    return fr

def main_f(set_ifov,redo_fits = False,redo_drift=False,redo_decoding=True,try_mode=True,old_method=False):
    set_,ifov = set_ifov
    save_folder,all_flds,fov = get_files(set_ifov)
    if try_mode:
        try:
            compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
        except:
            print("Failed within the main analysis:")
    else:
        compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
    
    return set_ifov
    

    
if __name__ == '__main__':
    items = [(set_,ifov) for set_ in ['']
                        for ifov in range(131)]

    inds = np.random.permutation(np.arange(len(items)))###
    items = [items[i] for i in inds]####
       
    # Comment out the next line to run the code in parallel
    # main_f(('',100),redo_drift=False,redo_decoding=True,try_mode=False)

    # Change to True to run in parallel
    if True:
        with Pool(processes=4) as pool:
            print('starting pool')
            result = pool.map(main_f, items)
#python worker_CRE_in_vitro.py