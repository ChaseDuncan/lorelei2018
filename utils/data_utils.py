import logging
import pickle
import os
import operator
import argparse
import sys
import math
import numpy as np
import tensorflow as tf

from .io_utils import id_to_title_map, title_to_id_map, load_pkl, outlinks
from sqlitedict import SqliteDict


# Names of fields in LORELEI kb
fields = ['origin', 'entity_type', 'entityid', 'name', 'asciiname', 'latitude', 'longitude', 'feature_class',
       'feature_class_name', 'feature_code', 'feature_code_name', 'feature_code_description', 'country_code',
       'country_code_name', 'cc2', 'admin1_code', 'admin1_code_name', 'admin2_code', 'admin2_code_name',
       'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'modification_date',
       'per_gpe_loc_of_association', 'per_title_or_position', 'per_org_of_association', 'per_role_in_incident',
       'per_year_of_birth', 'per_year_of_death', 'per_gender', 'per_family_member', 'note', 'aim',
       'org_date_established', 'date_established_note', 'org_website', 'org_gpe_loc_of_association',
       'org_members_employees_per', 'org_parent_org', 'executive_board_members', 'jurisdiction',
       'trusteeship_council', 'national_societies', 'external_link']

class LORELEIKBLoader:
    """
        Class for loading the LORELEI knowledge base (KB).
    """
    def __init__(self, kbfile):
        self.kb = {}  # eid --> item
        self.name2ent = {}
        self.load_kb(kbfile)
    
    """
        Primary function of the class -- loads the LORELEI kb which is to
        be found at the provided path. 
    """
    def load_kb(self, kbfile):
        e2e_path = kbfile + "e2e.pkl"
        n2e_path = kbfile + "n2e.pkl"
        if os.path.exists(e2e_path):
            logging.info("pkl found! loading map %s", e2e_path)
            self.kb = SqliteDict(e2e_path, tablename='lorelei', flag='r')
            self.name2ent = SqliteDict(n2e_path, tablename='name2ent', flag='r')
        else:
            logging.info("pkl not found ...")
            self.kb = SqliteDict(e2e_path, tablename='lorelei', autocommit=False)
            self.name2ent = SqliteDict(n2e_path, tablename='name2ent', autocommit=False)
            try:
                for idx, line in enumerate(open(kbfile)):
                    if idx > 0 and idx % 1000000 == 0:
                        logging.info("read %d lines", idx)

                    parts = line.rstrip('\n').split('\t')

                    if len(parts) != len(fields):
                        logging.info("bad line %d", idx)
                        continue
                        
                    endict = {}
                    
                    for field, v in zip(fields, parts):
                        if len(v) != 0:
                            endict[field] = v
                        self.kb[endict['entityid']] = endict
                        name = endict['name']

                        if name not in self.name2ent:
                            self.name2ent[name] = []
                        lst = self.name2ent[name]
                        lst.append(endict)
                        self.name2ent[name] = lst

                    logging.info("Writing KB dictionary to disk.")
                    self.kb.commit()
                    self.kb.close()
                    self.name2ent.commit()
                    self.name2ent.close()

                    self.kb = SqliteDict(e2e_path, tablename='lorelei', flag='r')
                    self.name2ent = SqliteDict(n2e_path, tablename='name2ent', flag='r')
            except KeyboardInterrupt:
                logging.info("ending prematurely.")
                logging.info("Writing KB dictionary to disk.")
                self.kb.commit()
                self.kb.close()
                self.name2ent.commit()
                self.name2ent.close()
                # reopen the kb now
                self.kb = SqliteDict(e2e_path, tablename='geonames',jflag='r')
                self.name2ent = SqliteDict(n2e_path, tablename='name2ent', flag='r')

        def __getitem__(self, item):
            return self.kb[item]

        def keys(self):
            return self.kb.keys()

def get_num_freebase_rel(wid_tup, freebase_rel_map):
    """
        Function for getting the number of freebase relations between
        two Wikipedia pages.

        @param: wid_tup, tuple of two Wikipedia page ids
        @return: the number of Freebase relations between the two pages
    """
    if wid_tup in freebase_rel_map:
        return freebase_rel_map[wid_tup]
    if wid_tup[::-1] in freebase_rel_map:
        return freebase_rel_map[wid_tup[::-1]]
    return 0
    
def count_outlinks(src, dest, outlinks_map):
    """ 
        Function for getting the number of outlinks a source title
        has to a destination title. This function is not symmetric.

        @param: src, the source title
        @param: dest, the destination title
        @return: the number of outlinks in Wikipedia form src to dest
    """
    print(src)
    print(dest)
    # the title isn't in the outlinks map then it has no outlinks
    try:
        outlinks = outlinks_map[src]    
    except KeyError:
        return 0
    return outlinks.count(dest)

def get_wikititle(kb, eid):
    """
        Function for retrieving Wikipedia link(s) for a given
        entity from the LORELEI KB.

        @param: kb, LORELEI KB please see io_utils.LORELEIKBLoader for details
        @param: eid, the entity id of the kb record
        @return: list of Wikipedia links which correspond to the record or None
                 if there are no Wikipedia pages for the record
    """
    if "external_link" in kb[eid]:
        links = kb[eid]["external_link"].split("|")
        wiki_link = [link for link in links if "en.wikipedia" in link]
        if len(wiki_link) == 0:
            return None
        else:
            return wiki_link
    return None

class Mention:
    """Class for maintaining the meta data associated with a mention in a document."""

    def __init__(self, constituent, max_cands, gold_labels=None):
        """Initializes Mention from Constituent from EL View """

        # tuple for maintaining start and end of span
        self.span = (constituent["start"],constituent["end"])
        # label assigned by the EL system
        self.label = constituent["label"]
        self.score = constituent["score"]
        # gold label if supplied
        self.gold = None
        # candidate titles for mention. list of tuples (title, score)
        self.candidate_titles = []

        label_score_map = constituent["labelScoreMap"]
        for label, score in label_score_map.items():
            self.candidate_titles.append((label,score))
        self.candidate_titles.sort(key=operator.itemgetter(1),reverse=True)
        self.candidate_titles = self.candidate_titles[:max_cands]
        
        # find index of gold in candidate titles list
        if gold_labels != None:
            gold_comp = [i for i, tup in enumerate(self.candidate_titles) \
                            if tup[0] == gold_labels[self.span]]
            if gold_comp:
                self.gold = gold_comp[0]
            else:
                self.gold = None

# end Mention class

class Document:
    """Class for representing a document for coherence."""

    def __init__(self, el_view, max_cands, gold_view=None):
        """Initializes a Document from an EL view."""
        # list of mentions in the document
        self.mentions = []
        # number of mentions to use in scoring -- only includes mentions
        # which weren't linked to NIL by the EL system
        self.m = None
        # dictionary which maps tuples of spans to the gold label annotation 
        self.gold_labels = None
        # if gold annotations are provided, instantiate dictionary of golds
        gold_constituents = gold_view.get_cons()
        if gold_view != None:
            self.gold_labels = {}
            for gold in gold_constituents:
                self.gold_labels[(gold["start"],gold["end"])] = gold["label"]

        # populate mentions list from constituents from an EL view
        constituents = el_view.get_cons()
        for constituent in constituents:
            if constituent["labelScoreMap"]:
                self.mentions.append(Mention(constituent, max_cands, gold_labels=self.gold_labels))
        self.mentions = [mention for mention in self.mentions if mention.gold != None] 
        self.m = len(self.mentions)

# end Document

class CoherenceFeatureExtractor:
    """Class for extracting features of entity linking document."""

    def __init__(self, num_unary_features, num_pairwise_features, max_cands_per_mention):
        # the number of unary features in the model
        self.num_unary_features = num_unary_features
        # the number of pairwise features in the model
        self.num_pairwise_features= num_pairwise_features

        # the maximum number of candidate titles per mention in a document
        self.max_cands_per_mention = max_cands_per_mention
        
        # used for outlinks feature
        outlinks_file = "/shared/preprocessed/cddunca2/wikipedia/outlinks.t2t"
        outlinks_map = outlinks(outlinks_file)
        self.outlinks_map = outlinks_map
        # used for freebase relations feature
        self.fb_relations_map = load_pkl("/home/cddunca2/lorelei2018/resources/freebase_relations.pkl")
        # used for cooccurence feature
        self.cooccurrence_map = None
        # map titles to wid
        self.title_to_id_map = title_to_id_map()

    # end __init__

    def init_unary_feature_matrix(self, document):
        """Initializes unary feature matrix for a Document.
        The matrix is m x max_cands_per_mention x num_unary_features. 
        Where m is number of mentions in the document.

        The features are as follows:
            1. log(p_i(c)) where p_i(c) is the score assigned to the candidate
               by the EL system. By convention log(0) = 0.
            2. log(1-p_i(c))
            3. A binary indicator feature for p_i(c) = 0
            4. A binary indicator feature for p_i(c) = 1
        
        @param: document, doc for which to extract unary features
        @return: an  mxcx4 matrix
        """
        unary_feature_matrix = np.zeros((document.m,self.max_cands_per_mention,\
                                         self.num_unary_features))
        for i, mention in enumerate(document.mentions):
            for j, candidate in enumerate(mention.candidate_titles):
                # break if we've already got the max number of cands
                if j >= self.max_cands_per_mention:
                    break
                candidate_score = candidate[1]

                # calculate value of features
                if math.isclose(candidate_score,0.0,rel_tol=1e-9):
                    f_1 = 0.0
                    f_2 = 0.0
                    f_3 = 1.0
                    f_4 = 0.0
                elif math.isclose(candidate_score, 1,rel_tol=1e-9):
                    f_1 = 0.0
                    f_2 = 0.0
                    f_3 = 0.0
                    f_4 = 1.0
                else:
                    f_1 = np.log(candidate_score)
                    f_2 = np.log(1 - candidate_score)
                    f_3 = 0.0
                    f_4 = 0.0

                # store feature vector for jth candidate of ith mention
                unary_feature_matrix[i,j,:] = np.asarray([f_1,f_2,f_3,f_4])

        return unary_feature_matrix


    def init_pairwise_feature_matrix(self, document):
        """ Initializes pairwise feature matrix for a document. This
        matrix is size 
        
        m x m-1 x max_cands_per_mention x max_cands_per_mention x num_pairwise_features.

        The features are as follows:
        1. Sum of the number of outlinks between the two titles
        2. Number of Freebase relations between the two titles
        3. Binary feature which is high if the titles are the same

        @param: document, document to represented
        @return: feature matrix
        """
        pairwise_feature_matrix = np.zeros((document.m,document.m-1,self.max_cands_per_mention,
                                            self.max_cands_per_mention,self.num_pairwise_features))
        for i, m1 in enumerate(document.mentions):
            for j, m2 in enumerate(document.mentions):
                if i != j:
                    for k, cand_m1 in enumerate(document.mentions[i].candidate_titles):
                        for l, cand_m2 in enumerate(document.mentions[j].candidate_titles):
                            # if k or l are greater than the max number of cands, break
                            if k >= self.max_cands_per_mention or l >= self.max_cands_per_mention:
                                break
                            # adjust second index since the second dimension is 
                            # one smaller than first
                            if j > i:
                                j-=1
                            pairwise_feature_matrix[i,j,k,l,:] = \
                                self._pairwise_feature_vec(cand_m1[0], cand_m2[0])
        return pairwise_feature_matrix
    
    def _pairwise_feature_vec(self, yi, yj):
        """ Helper which populates a pairwise feature vector between two titles.
        See comment in init_pairwise_feature_matrix for description of features.

        @param: yi, first title
        @param: yj, second title
        @return: vector representation of pairwise similarity of two titles
        """
        # TODO: some caching here, probably
        fv = np.zeros(self.num_pairwise_features)
        if "unk_wid" in yi or "unk_wid" in yj:
            return fv
        # outlinks feature
        num_outlinks = \
            count_outlinks(yi,yj,self.outlinks_map) + \
            count_outlinks(yj,yi,self.outlinks_map)
        print("num_outlinks: ", num_outlinks)
        if num_outlinks >= 5:
            fv[4] = 1
        elif num_outlinks == 4:
            fv[3] = 1
        elif num_outlinks == 3:
            fv[2] = 1
        elif num_outlinks == 2:
            fv[1] = 1
        elif num_outlinks == 1:
            fv[0] = 1
        
        # freebase relations feature
        wid_tup = (self.title_to_id_map[yi],self.title_to_id_map[yj])
        num_fb_relations = get_num_freebase_rel(wid_tup,self.fb_relations_map)
        print("num_fb: ", num_fb_relations)
        if num_fb_relations >= 5:
            fv[9] = 1
        elif num_fb_relations == 4:
            fv[8] = 1
        elif num_fb_relations == 3:
            fv[7] = 1
        elif num_fb_relations == 2:
            fv[6] = 1
        elif num_fb_relations == 1:
            fv[5] = 1

        # are the titles the same?
        if yi == yj:
            fv[-1] = 1
        return fv

    def init_gold_vector(self, doc):
        """Initializes a vector is size m where element in the vector
        is the index of the gold label in the candidate list for the corresponding
        entity.

        @return a vector of indices of gold labels
        """
        gv = np.zeros(doc.m,dtype=np.int64)
        for i, mention in enumerate(doc.mentions):
            gv[i] = mention.gold
        return gv
    
  # end CoherenceFeatureExtractor
