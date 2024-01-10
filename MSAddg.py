#!/usr/bin/env python3

from utils import parsermodule
from utils import generate_MSA
from utils import potts
from utils import feature
import tensorflow as tf
import numpy as np
import os


parser = parsermodule.get_parser()
args = parser.parse_args()
seqfilename = args.sequence
a3mfilename = args.a3mfile
path_to_database = args.database
iter_num = args.iteration_number
num_threads = args.num_threads
gpu_id = args.id_of_gpu


outfilename = seqfilename+".scan.txt"


if a3mfilename == None:
    a3mfilename = generate_MSA.hhsearch(seqfilename, iter_num, path_to_database, num_threads)



w_out, v_out = potts.potts(a3mfilename,gpu_id)


seq = feature.getSeq(seqfilename)
msa = feature.get_MSA(a3mfilename)

PATH = os.path.dirname(os.path.abspath(__file__))
path_to_model = os.path.join(PATH,"model",'MSAddg_CNN_regularizers')

model = tf.keras.models.load_model(path_to_model)

def run(wpm,msa,v_out,w_out,seq):
    static_features = feature.getStatic(wpm) + [1]
    statical_features = feature.getStatical(msa,v_out,w_out,seq,wpm) + [1]
    feature_map = np.outer(np.array(statical_features),np.array(static_features)).reshape(8,8,1)
    return feature_map

def gen_mut_scan(seq):
    i = 0
    mut_list = []
    AA_list = ["Q",
               "W",
               "E",
               "R",
               "T",
               "Y",
               "I",
               "P",
               "A",
               "S",
               "D",
               "F",
               "G",
               "H",
               "K",
               "L",
               "C",
               "V",
               "N",
               "M"]
    for res in seq:
        i += 1
        for AA in AA_list:
            if AA == res:
                continue
            else:
                mut_list.append(res+'_'+str(i)+'_'+AA)
    return mut_list

mut_list = gen_mut_scan(seq)


with open(outfilename,"w+") as of:
    of.write("mutation\tscore\n")
    of.close()
    for mut in mut_list:
        wpm = mut.split("_")
        #wpm = mutation.split("_")
        feature_maps = []
        if wpm[0] and wpm[2] in "ARNDCQEGHILKMFPSTWYV":
            if seq[int(wpm[1])-1] == wpm[0]:
                feature_map = run(wpm,msa,v_out,w_out,seq)
                # feature_maps.append(feature_map)
                with open(outfilename,"a+") as of:
                    of.write(mut+"\t"+str(model.predict(np.array([feature_map]))[0][0])+"\n")
                    of.close()
