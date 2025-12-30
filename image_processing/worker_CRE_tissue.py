master_analysis_folder = r'path_to_analysis_folder'
lib_fl = r'path_to_codebook.csv' ### codebook file
### Did you compute PSF and median flat field images?
psf_file = r'path_to_psf_file.pkl'
import os
flat_field_tag = r'path_to_flat_field_file_tag'+os.sep
master_data_folder = [r'path_to_data_folder'] # raw image data folder
save_folder =r'path_to_save_folder'  ### where to save processed data

iHm=1
iHM=100
bk_folder = None # optional path to folder for background signal subtraction
bk_frs = [1.15,1.1,1.05]

from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)

from ioMicro import *

def get_best_translation_pointsT(fl,save_folder,fl_ref,save_folder_ref,set_='',resc=5,th=5,
                                 im_med_fl=flat_field_tag+r'med_col_raw3.npz',
                                 psf_fl=psf_file):
    obj = get_dapi_features(fl,save_folder,set_,
    gpu=True,im_med_fl=im_med_fl,psf_fl=psf_fl,redo=False)
    obj_ref = get_dapi_features(fl_ref,save_folder_ref,set_,
    gpu=True,im_med_fl=im_med_fl,psf_fl=psf_fl,redo=False)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus[obj.Xh_plus[:,-1]>th][:,:3]
        X_ref = obj_ref.Xh_plus[obj_ref.Xh_plus[:,-1]>th][:,:3]#obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus[obj.Xh_minus[:,-1]>th][:,:3]
        X_ref = obj_ref.Xh_minus[obj_ref.Xh_minus[:,-1]>th][:,:3]#obj_ref.Xh_plus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]

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

def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    fl = fld+os.sep+fov
    im_ = read_im(fl)
    im__ = np.array(im_[icol],dtype=np.float32)
    

    ### new method
    fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'
    im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
    im_med = cv2.blur(im_med,(20,20))
    im_med = im_med/np.median(im_med)
    im__ = im__/im_med
    Xm = np.array([0,0,0])
    if bk_folder is not None:
        fl_ref = bk_folder+os.sep+fov
        tag_ = os.path.basename(fld)
        set_ =  '_set'+tag.split('_set')[-1] if '_set' in tag_ else ''
        drift = get_best_translation_pointsT(fl,save_folder,fl_ref,bk_folder,set_=set_)
        print(drift)
        from scipy.ndimage import shift
        im_bk = np.array(read_im(fl_ref)[icol],dtype=np.float32)
        im_bk_ = shift(im_bk/im_med,-drift[0],order=0,cval=np.nan)
        if False:
            imr = im__/im_bk_
            fr1p = np.nanpercentile(imr,1)
            print(fr1p)
        fr1p = bk_frs[icol]
        im__ = im__-fr1p*im_bk_
        zb,xb,yb = np.where(~np.isnan(im__))
        xm,xM = np.min(xb),np.max(xb)+1
        ym,yM = np.min(yb),np.max(yb)+1
        zm,zM = np.min(zb),np.max(zb)+1
        im__ = im__[zm:zM,xm:xM,ym:yM]
        Xm = np.array([zm,xm,ym])
    #print(im__.shape,zm,zM,xm,xM,ym,yM,np.sum(np.isnan(im__)))
    
    # Xh = get_local_max_tile(im__,th=2500,s_ = 280,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
    #                         deconv={'method':'wiener','beta':0.0001},
    #                         delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    # print(Xh.shape)
    im__s = norm_slice(im__,100)
    Xh = get_local_maxfast_tensor(im__s,2500,im_raw=im__,dic_psf=None,delta=1,delta_fit=5,sigmaZ=1,sigmaXY=1.5,gpu=False)
    if len(Xh):
        Xh[:,:3]=Xh[:,:3]+Xm
    np.savez_compressed(save_fl,Xh=Xh)

def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    psf = np.load(psf_file,allow_pickle=True)
    
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            redo2 = False
            try:
                np.load(save_fl)['Xh']
            except:
                redo2 = True
            if not os.path.exists(save_fl) or redo or redo2:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)      

def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf    
                    
def compute_decoding(save_folder,fov,set_,redo=False):
    dec = decoder_simple(save_folder,fov,set_)
    complete = dec.check_is_complete()
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        dec.get_XH(fov,set_,ncols=3,th_h=10000)#number of colors match 
        #dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        get_intersV2(dec,nmin_bits=3,dinstance_th=2,enforce_color=True,enforce_set=None,redo=False)
        get_icodesV3(dec,nmin_bits=3,iH=-3)
        #dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        #dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=True,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        #get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=24,is_unique=True)

def get_files(set_ifov,iHm=iHm,iHM=iHM):

    if not os.path.exists(save_folder): os.makedirs(save_folder)
    all_flds = []
    for master_folder in master_data_folder:
        all_flds += glob.glob(master_folder+r'\H*_CREMER*')
        #all_flds += glob.glob(master_folder+r'\H*_TMER*')
    all_flds += glob.glob(master_folder+r'\H*GFP*')

    print(len(all_flds))
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    print(len(all_flds))
    set_,ifov = set_ifov
    all_flds = [fld for fld in all_flds if (set_ in os.path.basename(fld)) and os.path.isdir(fld)]
    print(len(all_flds))
    #all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
    #print(len(all_flds))
    
    
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
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
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
        if len(X) and len(X_ref):
            try:
                tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
            except:
                pass
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus#[:,:3]
        X_ref = obj_ref.Xh_minus#[:,:3]
        X = X[X[:,-1]>th][:,:3]
        X_ref = X_ref[X_ref[:,-1]>th][:,:3]
        if len(X) and len(X_ref):
            try:
                tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
            except:
                pass
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

def main_f(set_ifov,redo_fits = False,redo_drift=False,redo_decoding=False,try_mode=True,old_method=False):
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
    # start 4 worker processes
    ifovs = np.arange(1000)
    items = [(set_,ifov) for set_ in ['_set1','_set2']
                        for ifov in ifovs]
    main_f(['_set1',120],redo_fits =False,redo_drift=False,redo_decoding=False,try_mode=False)
    #227, 613, 639, 677, 712
    #for ifov in [227, 613, 639, 677, 712]:
    #    main_f(['',ifov],redo_fits =False,redo_drift=False,redo_decoding=False,try_mode=False)
    
    if False:
        with Pool(processes=4) as pool:
            print('starting pool')
            result = pool.map(main_f, items)
#python worker_CRE_tissue.py