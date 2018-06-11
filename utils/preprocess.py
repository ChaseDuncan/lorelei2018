import json
import logging
import pickle

from os import listdir
from os.path import isfile, join
from sqlitedict import SqliteDict

"""
    This file contains a bunch of helper functions for preprocessing
    data for LORELEI 2018 EDL.
"""
logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)
# directory where parsed Wikipedia is stored in json files
WIKI_DUMP_DIR="/shared/preprocessed/wikiparser/wikiExtractorJsonPages/"
MID2WID="/shared/preprocessed/upadhya3/enwiki-datamachine/mid.wikipedia_en_id"
WID2TITLE="/shared/preprocessed/upadhya3/enwiki-datamachine/idmap/enwiki-20170520.id2t"

"""
    Creates a tab separated file with two columns. The first column
    is Wikipedia title of a given page and the second column is a
    space separated string of outlinks from that page.

    @param: outfile path to which the resulting file should be written
"""
def outlink_counts(outfile):
    out = open(outfile, "w")
    wikifiles = \
        [join(WIKI_DUMP_DIR+"/",f) \
        for f in listdir(WIKI_DUMP_DIR) if isfile(join(WIKI_DUMP_DIR+"/",f))]
    proc_ct = 0
    num_files = len(wikifiles)
    
    for wikifile in wikifiles:
        proc_ct+=1
        if proc_ct % 10 == 0:
            logging.info("%f complete"%(proc_ct/float(num_files)))
        with open(wikifile, "r") as f:
            for json_str in f.readlines():
                try:
                    wiki_page = json.loads(json_str.strip())
                    out.write(wiki_page["wikiTitle"] + "\t" + " ".join(wiki_page["hyperlinks"].values()) + "\n")
                except:
                    logging.info("failed to parse " + wikifile)
                    out.flush()
            out.flush()

def format_mid(mid):
    """
        Formats a string which represents a Freebase id (mid) from
        the format used in FB15K-237 to that which is used in the cogcomp
        Wikipedia data files. Namely,

            "/m/07cw4"->"m.07cw4"
        @param mid, string of mid to be reformatted
    """
    return mid[1::].replace("/",".")

def build_freebase_relations_dict(mid2wid_file, relation_data_dir, outpath):
    """
        Build and pickle a dict which maps tuples of Wikipedia pageids
        to the number of Freebase relations they have according to the data
        in the FB15K-237 dataset.
        
        Requires a tab separated file for Freebase IDs to Wikipedia page IDs.
        Requires the FB15K-237 dataset which can be downloaded from the Web.

        @param: mid2wid_file, a tsv which mps MIDs to Wikipedia page ids
        @param: relation_data_dir, directory containing the FB15K-237 dataset
        @param: outpath, path to serialize dict to
    """
    dict_path = outpath+"freebase_relations.pkl" 
    kb = {} 
    logging.info("Building mid2wid dict.")
    mid2wid = {}

    with open(mid2wid_file, "r") as f:
        for line in f.readlines():
            spline = line.strip().split("\t")
            mid2wid[spline[0]]=spline[1]
    
    files = ["train.txt","valid.txt","test.txt"]
    count = 0
    temp_k = None
    for f in files:
        with open(relation_data_dir+f,"r") as relations:
            logging.info(relation_data_dir+f)
            for relation in relations.readlines():
                count+=1

                m1,rel,m2 = relation.strip().split("\t")
                m1 = format_mid(m1)
                m2 = format_mid(m2)

                w1 = None
                w2 = None

                try:
                    w1 = mid2wid[m1]
                except:
                    logging.info(m1 + " not in dict.")
                    continue

                try:
                    w2 = mid2wid[m2]
                except:
                    logging.info(m2 + " not in dict.")
                    continue

                k = (w1,w2)
                temp_k = k
                if k in kb:
                    kb[k]+=1
                elif k[::-1] in kb:
                    kb[k[::-1]]+=1
                else:
                    kb[k]=1

        f = open(dict_path,"wb")
        pickle.dump(kb,f)

if __name__=="__main__":
    #outlink_counts("/shared/preprocessed/cddunca2/wikipedia/outlinks.t2t")
    mid2wid_file="/shared/preprocessed/upadhya3/enwiki-datamachine/mid.wikipedia_en_id"
    relation_data_dir="data/FB15K-237/"
    outpath="resources/"
    build_freebase_relations_dict(mid2wid_file, relation_data_dir, outpath)
