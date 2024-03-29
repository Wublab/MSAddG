#!/usr/bin/env python

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
import os


alphabet = "ARNDCQEGHILKMFPSTWYV-"
states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
    a2n[a] = n
    
def potts(seq, gup_id):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gup_id)
  alphabet = "ARNDCQEGHILKMFPSTWYV-"
  states = len(alphabet)
  a2n = {}
  for a,n in zip(alphabet,range(states)):
    a2n[a] = n
  ################
  
  def aa2num(aa):
    '''convert aa into num'''
    if aa in a2n: return a2n[aa]
    else: return a2n['-']
  
  # from fasta
  import string
  def parse_fasta(filename, a3m=True):
    '''function to parse fasta file'''
    
    if a3m:
      # for a3m files the lowercase letters are removed
      # as these do not align to the query sequence
      rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
      
    header, sequence = [],[]
    lines = open(filename, "r")
    for line in lines:
      line = line.rstrip()
      if line[0] == ">":
        header.append(line[1:])
        sequence.append([])
      else:
        if a3m: line = line.translate(rm_lc)
        else: line = line.upper()
        sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    
    return np.array(header), np.array(sequence)
  
  def filt_gaps(msa,gap_cutoff=1):
    '''filters alignment to remove gappy positions'''
    tmp = (msa == states-1).astype(np.float32)
    non_gaps = np.where(np.sum(tmp.T,-1).T/msa.shape[0] < gap_cutoff)[0]
    return msa[:,non_gaps],non_gaps
  
  def get_eff(msa,eff_cutoff=0.8):
    '''compute effective weight for each sequence'''
    ncol = msa.shape[1]
    
    # pairwise identity
    msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
  
    # weight for each sequence
    msa_w = (msa_sm >= eff_cutoff).astype(np.float32)
    msa_w = 1/np.sum(msa_w,-1)
    
    return msa_w
  
  def mk_msa(seqs):
    '''converts list of sequences to msa'''
    
    msa_ori = []
    for seq in seqs:
      msa_ori.append([aa2num(aa) for aa in seq])
    #print(msa_ori)
    msa_ori = np.array(msa_ori)
    #print(msa_ori)
    
    # remove positions with more than > 50% gaps
    msa, v_idx = filt_gaps(msa_ori,1)
    
    # compute effective weight for each sequence
    msa_weights = get_eff(msa,0.8)
  
    # compute effective number of sequences
    ncol = msa.shape[1] # length of sequence
    w_idx = v_idx[np.stack(np.triu_indices(ncol,1),-1)]
    
    return {"msa_ori":msa_ori,
            "msa":msa,
            "weights":msa_weights,
            "neff":np.sum(msa_weights),
            "v_idx":v_idx,
            "w_idx":w_idx,
            "nrow":msa.shape[0],
            "ncol":ncol,
            "ncol_ori":msa_ori.shape[1]}
  
  # external functions
  
  def sym_w(w):
    '''symmetrize input matrix of shape (x,y,x,y)'''
    x = w.shape[0]
    w = w * np.reshape(1-np.eye(x),(x,1,x,1))
    w = w + tf.transpose(w,[2,3,0,1])
    return w
  
  def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
    # adam optimizer
    # Note: this is a modified version of adam optimizer. More specifically, we replace "vt"
    # with sum(g*g) instead of (g*g). Furthmore, we find that disabling the bias correction
    # (b_fix=False) speeds up convergence for our case.
    
    if var_list is None: var_list = tf.trainable_variables() 
    gradients = tf.gradients(loss,var_list)
    if b_fix: t = tf.Variable(0.0,"t")
    opt = []
    for n,(x,g) in enumerate(zip(var_list,gradients)):
      if g is not None:
        ini = dict(initializer=tf.zeros_initializer,trainable=False)
        mt = tf.get_variable(name+"_mt_"+str(n),shape=list(x.shape), **ini)
        vt = tf.get_variable(name+"_vt_"+str(n),shape=[], **ini)
        
        mt_tmp = b1*mt+(1-b1)*g
        vt_tmp = b2*vt+(1-b2)*tf.reduce_sum(tf.square(g))
        lr_tmp = lr/(tf.sqrt(vt_tmp) + 1e-8)
  
        if b_fix: lr_tmp = lr_tmp * tf.sqrt(1-tf.pow(b2,t))/(1-tf.pow(b1,t))
  
        opt.append(x.assign_add(-lr_tmp * mt_tmp))
        opt.append(vt.assign(vt_tmp))
        opt.append(mt.assign(mt_tmp))
          
    if b_fix: opt.append(t.assign_add(1.0))
    return(tf.group(opt))
  
  
  def GREMLIN(msa, opt_type="adam", opt_iter=100, opt_rate=1.0, batch_size=None):
    
    ##############################################################
    # SETUP COMPUTE GRAPH
    ##############################################################
    # kill any existing tensorflow graph
    tf.reset_default_graph()
    
    ncol = msa["ncol"] # length of sequence
  
    # msa (multiple sequence alignment) 
    MSA = tf.placeholder(tf.int32,shape=(None,ncol),name="msa")
    
    # one-hot encode msa
    OH_MSA = tf.one_hot(MSA,states)
    
    # msa weights
    MSA_weights = tf.placeholder(tf.float32, shape=(None,), name="msa_weights")
    
    # 1-body-term of the MRF
    V = tf.get_variable(name="V", 
                        shape=[ncol,states],
                        initializer=tf.zeros_initializer)
  
    # 2-body-term of the MRF
    W = tf.get_variable(name="W",
                        shape=[ncol,states,ncol,states],
                        initializer=tf.zeros_initializer)
    
    # symmetrize W
    W = sym_w(W)
    
    def L2(x): return tf.reduce_sum(tf.square(x))
    
    ########################################
    # V + W
    ########################################
    VW = V + tf.tensordot(OH_MSA,W,2)
  
    # hamiltonian
    H = tf.reduce_sum(tf.multiply(OH_MSA,VW),axis=(1,2))
    # local Z (parition function)
    Z = tf.reduce_sum(tf.reduce_logsumexp(VW,axis=2),axis=1)
  
    # Psuedo-Log-Likelihood
    PLL = H - Z
  
    # Regularization
    L2_V = 0.01 * L2(V)
    L2_W = 0.01 * L2(W) * 0.5 * (ncol-1) * (states-1)
  
    # loss function to minimize
    loss = -tf.reduce_sum(PLL*MSA_weights)/tf.reduce_sum(MSA_weights)
    loss = loss + (L2_V + L2_W)/msa["neff"]
  
    ##############################################################
    # MINIMIZE LOSS FUNCTION
    ##############################################################
    if opt_type == "adam":  
      opt = opt_adam(loss,"adam",lr=opt_rate)
  
    # generate input/feed
    def feed(feed_all=False):
      if batch_size is None or feed_all:
        return {MSA:msa["msa"], MSA_weights:msa["weights"]}
      else:
        idx = np.random.randint(0,msa["nrow"],size=batch_size)
        return {MSA:msa["msa"][idx], MSA_weights:msa["weights"][idx]}
    
    # optimize!
    with tf.Session() as sess:
      # initialize variables V and W
      sess.run(tf.global_variables_initializer())
      
      # initialize V
      msa_cat = tf.keras.utils.to_categorical(msa["msa"],states)
      pseudo_count = 0.01 * np.log(msa["neff"])
      V_ini = np.log(np.sum(msa_cat.T * msa["weights"],-1).T + pseudo_count)
      V_ini = V_ini - np.mean(V_ini,-1,keepdims=True)
      sess.run(V.assign(V_ini))
      
      # compute loss across all data
      get_loss = lambda: round(sess.run(loss,feed(feed_all=True)) * msa["neff"],2)
      print("starting",get_loss())
      
      if opt_type == "lbfgs":
        lbfgs = tf.contrib.opt.ScipyOptimizerInterface
        opt = lbfgs(loss,method="L-BFGS-B",options={'maxiter': opt_iter})
        opt.minimize(sess,feed(feed_all=True))
        
      if opt_type == "adam":
        for i in range(opt_iter):
          sess.run(opt,feed())  
          if (i+1) % int(opt_iter/10) == 0:
            print("iter",(i+1),get_loss())
      
      # save the V and W parameters of the MRF
      V_ = sess.run(V)
      W_ = sess.run(W)
      
    # only return upper-right triangle of matrix (since it's symmetric)
    tri = np.triu_indices(ncol,1)
    W_ = W_[tri[0],:,tri[1],:]
    
    mrf = {"v": V_,
           "w": W_,
           "v_idx": msa["v_idx"],
           "w_idx": msa["w_idx"]}
    
    return mrf
  
  names, seqs = parse_fasta(seq)
  
  # process input sequences
  msa = mk_msa(seqs)
  mrf = GREMLIN(msa)
  
  def normalize(x):
    x = stats.boxcox(x - np.amin(x) + 1.0)[0]
    x_mean = np.mean(x)
    x_std = np.std(x)
    return((x-x_mean)/x_std)
  
  def get_mtx(mrf):
    '''get mtx given mrf'''
    
    # l2norm of 20x20 matrices (note: we ignore gaps)
    raw = np.sqrt(np.sum(np.square(mrf["w"][:,:-1,:-1]),(1,2)))
    raw_sq = squareform(raw)
  
    # apc (average product correction)
    ap_sq = np.sum(raw_sq,0,keepdims=True)*np.sum(raw_sq,1,keepdims=True)/np.sum(raw_sq)
    apc = squareform(raw_sq - ap_sq, checks=False)
  
    mtx = {"i": mrf["w_idx"][:,0],
           "j": mrf["w_idx"][:,1],
           "raw": raw,
           "apc": apc,
           "zscore": normalize(apc)}
    return mtx

  mtx = get_mtx(mrf)
  
  w_out = np.zeros((msa["ncol_ori"], msa["ncol_ori"], states * states + 1))
  v_out = np.zeros((msa["ncol_ori"], states))
  
  mrf_ = np.reshape(mrf["w"],(-1, states * states))
  mtx_ = np.expand_dims(mtx["apc"],-1)
  
  w_out[(mtx["i"],mtx["j"])] = np.concatenate((mrf_, mtx_),-1)
  w_out += np.transpose(w_out,(1,0,2))
  v_out[mrf["v_idx"]] = mrf["v"]
  
  return w_out, v_out