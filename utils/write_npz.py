import sys
import os
import csv
import shutil
import numpy as np

from io_utils import load_pkl, get_wid_mid_map, outlinks, id_to_title_map
from data_utils import get_num_freebase_rel, count_outlinks

year = sys.argv[2]
npz_dir = sys.argv[1]+"/npz/"+year+"/"
try:
    shutil.rmtree(npz_dir)
except:
    print('no dir to remove')
os.mkdir(npz_dir)


outlinks_file = "/shared/preprocessed/cddunca2/wikipedia/outlinks.t2t"
outlinks_map = outlinks(outlinks_file)

# Create dictionary of documents to list of entity lists --Sameer
# Maps mention id to list of candidates.
ments_cands_dict = {}
# map of mention id to file
ments_files_dict = {}
# Maps a filename to a list of surface forms of all mentions in the file
files_ments_dict = {}
data_dir = "/home/cddunca2/CS546/data/tac"+year+"/"

def populate_maps_from_file(files_entities_mentions):
    """
        There are 3 maps which are used for feature extraction and are listed above.
        This function populates those maps based on the assumption that a representation
        of the corpus that is being processed exists which is formatted as follows:

        Example line for 'file_entities_mentions.txt'
        ENG_NW_001278_20130109_F00011TB4.xml  ('EDL17_EVAL_08161', '24050318,27019,') ('EDL17_EVAL_21559', '38288499,')

        The file is tab separated with two columns. The first column is the name of the
        file which the mentions come from. The second column is a space separated string
        of tuples where the first element in the tuple is a unique identifier for the mention
        and the second element is a comma separated list of Wikipedia page ids. These page
        ids are the candidate pages which have been provided by some EL system.

        @param: file_entities_mentions, the file_entities_mentions.txt file for the corpus
    """
    with open(files_entities_mentions, 'r') as files_entities:
        files_entities_reader = csv.reader(files_entities, delimiter = '\t')
        for row in files_entities_reader:
            # get list of lists of candidates
            if not row:
                continue
            file_name, *mentions = row
            # parses line from file_entities_mentions.txt into mentions and candidates
            # kinda gross... why not parse the line as a list of tuples?
            candidates = [mc.split('\'')[3].split(',')[:-1] for mc in mentions]
            mentions = [mc.split('\'')[1].split(',')[0] for mc in mentions]
            
            if file_name not in files_ments_dict:
                files_ments_dict[file_name] = mentions

            for m,c in zip(mentions, candidates):
                ments_files_dict[m] = file_name
                ments_cands_dict[m] = c

populate_maps_from_file(data_dir+"files_entities_mentions.txt")
wiki_mid_dict = get_wid_mid_map()

unary_features = {}
with open(data_dir+'/preds/preds.txt', 'r', encoding="utf8") as unary_file:
    with open(data_dir+'/mention_candidate.txt', 'r') as ment_cand_file:
        mention_candidate_reader = csv.reader(ment_cand_file, delimiter=' ')
        unary_reader = csv.reader(unary_file, delimiter='\t')
        for unary_row, ment_cand_row in zip(unary_reader, mention_candidate_reader):
            #need to match up with file
            ment_id, *cand_id = ment_cand_row
            if ment_id in ments_files_dict:
                file_name = ments_files_dict[ment_id]
                if not file_name in unary_features:
                    unary_features[file_name] = {}
                index = 0
                for entry in unary_row:
                    wiki_title, *scores = entry.split(' ')
                    assert(len(scores)==3)
                    if wiki_title != '<unk_wid>':
                        fl_score = [float(s) for s in scores]
                        if not ment_id in unary_features[file_name]:
                            unary_features[file_name][ment_id] = [fl_score]
                        else:
                            unary_features[file_name][ment_id].append(fl_score)
                
                # if the mention id isn't in the feature dict then it had no candidates
                if ment_id not in unary_features[file_name]:
                    unary_features[file_name][ment_id] = [[0.0, 0.0, 0.0]]
                if file_name in unary_features:
                    while len(unary_features[file_name][ment_id]) % 30 != 0:
                        unary_features[file_name][ment_id].append([0.0]*3)

def read_and_pad_npy(file_name):
    """
        Creates a an mx(m-1)x30x30 matrix where m is the number of mentions
        in file_name and 30 is the maximum number of candidates per mention.
        An element in the matrix is the number of hyperlinks between a given
        mention's candidate and a different mention's candidate. This is the
        sum of the number of outlinks from each page to the other.
        
        @param: file_name, the name of the file in the corpus for which to 
                           create the Freebase relation matrix.
        @return: a numpy matrix which represents the number of hyperlinks
                 between candidate entities in the document.
    """
    mentions = files_ments_dict[file_name]
    m = len(mentions)
    id2t_map = id_to_title_map()
    hyperlinks_counts = np.zeros((m, m-1, 30, 30))
    for i, ment1 in enumerate(mentions):
        for j, ment2 in enumerate(mentions):
            if j != i:
                for k, wiki_id1 in enumerate(ments_cands_dict[ment1]):
                    for l, wiki_id2 in enumerate(ments_cands_dict[ment2]):
                        try:
                            t1 = id2t_map[wiki_id1]
                            t2 = id2t_map[wiki_id2]
                            hyperlinks_counts[i,j-1,k,l] = \
                                float(count_outlinks(t1,t2,outlinks_map)) + float(count_outlinks(t2,t1,outlinks_map))
                        except:
                            import pdb
                            pdb.set_trace()
    return hyperlinks_counts 

def read_and_pad_freebase(file_name):
    """
        Creates a an mx(m-1)x30x30 matrix where m is the number of mentions
        in file_name and 30 is the maximum number of candidates per mention.
        An element in the matrix is the number of relations between a given
        mention's candidate and a different mention's candidate.     
        
        @param: file_name, the name of the file in the corpus for which to 
                           create the Freebase relation matrix.
        @return: a numpy matrix which represents the number of relations
                 between candidate entities in the document.
    """
    #TODO: move this out of here
    fb_rel_map = load_pkl("resources/freebase_relations.pkl")
    mentions = files_ments_dict[file_name]
    m = len(mentions)

    freebase_counts = np.zeros((m, m-1, 30, 30))
    for i, ment1 in enumerate(mentions):
        for j, ment2 in enumerate(mentions):
            if j != i:
                for k, wiki_id1 in enumerate(ments_cands_dict[ment1]):
                    for l, wiki_id2 in enumerate(ments_cands_dict[ment2]):
                        if wiki_id1 in wiki_mid_dict and wiki_id2 in wiki_mid_dict:
                            mid1 = wiki_mid_dict[wiki_id1]
                            mid2 = wiki_mid_dict[wiki_id2]
                            id_tuple = (mid1,mid2)
                            freebase_counts[i,j-1,k,l] = \
                                float(get_num_freebase_rel(id_tuple,fb_rel_map))
    return freebase_counts

# TODO: replace this with something...
"""
    The next block of logic creates a dictonary which maps the mention id to its
    gold link. The file which is does this for 2016 and 2017 doesn't appear to exist.
"""
#get gold
gold_dict = {}
# Use for 2009, 2010
if year == "2009" or year == "2010":
    with open(data_dir + 'tac' + year + 'trainingdata.tsv', 'r', encoding="utf8") as gold_file: #with tab delimiter
        reader = csv.reader(gold_file, delimiter='\t')
        for row in reader:
            wiki_id = row[1]
            entity = row[-1]
            if "NIL" not in wiki_id:
                gold_dict[entity] = wiki_id

if year== "2016" or year == "2017":
    with open(data_dir + 'test_trunc.txt', 'r', encoding="utf8") as test_file:
        reader = csv.reader(test_file, delimiter="\t")
        for row in reader:
            wiki_id = row[1]
            entity = row[-2]
            if "NIL" not in wiki_id:
                gold_dict[entity] = wiki_id

def pad_and_reshape(arr):
    #The array len should be a multiple of 30
    repeats = len(arr)//30
    arr = np.array(arr)
    return repeats, np.reshape(arr, (repeats, 30 ,3))

# TODO: replace with outlinks common utility
# generate pairwise features and singleton features for each document
for fname in unary_features:
    #get hyperlink counts
    hyperlink_counts = read_and_pad_npy(fname)
    #freebase counts
    freebase_counts = read_and_pad_freebase(fname)
    assert(hyperlink_counts.shape == freebase_counts.shape)
    pairwise_features = np.stack((hyperlink_counts, freebase_counts), axis=-1)
    #get unary for each mention (in correct order)
    single_arr = []
    counts_dict = {} 

    for m in files_ments_dict[fname]:
        count, nump = pad_and_reshape(unary_features[fname][m])
        counts_dict[m] = count
        single_arr.append(nump[0,:,:])

    single = np.stack([single_arr], axis=0)[0,:,:,:]
    #get gold
    accept = True
    gold = np.zeros(single.shape[0])
    for i, m in enumerate(files_ments_dict[fname]):
        if m in gold_dict:
            cands = ments_cands_dict[m]
            if gold_dict[m] in cands:
                gold[i] = cands.index(gold_dict[m])
            else:
                print('Gold ', gold_dict[m], ' not in ', m)
                accept=False
                break
        else:
            accept = False
        if accept:
            #make sure single is an np array
            print("gold: ", gold)
            print("gold.astype(np.int64): ", gold.astype(np.int64))

            np.savez(npz_dir + fname + '.npz', single=single, pair=pairwise_features, truth=gold.astype(np.int64))
        else:
            print("File not accepted.")
