import json
import logging
import pickle
import argparse
import sys
import os
from os import listdir
from os.path import isfile, join
from sqlitedict import SqliteDict

from ccg_nlpy import core, local_pipeline, remote_pipeline

# Location of file which maps mids to Wikipedia page ids
MID2WID="/shared/preprocessed/upadhya3/enwiki-datamachine/mid.wikipedia_en_id"
MID2WID_PKL="resources/mid2wid.pkl"
WID2MID_PKL="resources/wid2mid.pkl"

def save_pkl(fname, obj):
    """
        Serializes object to given filename.

        @param: fname, file to write pickle
        @param: obj, object to be serialized (pickled)
    """
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(fname):
    """
        Loads a pickled object.

        @param: fname, filename of pickle to load.
        @return: deserialized object.
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)

def get_ta_dir(directory):
    """
        Returns a list of TextAnnotation objects which are instatiated
        using the serialized json data in the directory parameter.

        @param directory path to directory with serialized TAs
        @return tas a list of TextAnnotations
    """
    #pipeline = local_pipeline.LocalPipeline()
    pipeline = remote_pipeline.RemotePipeline()
    serialized_tas = [join(directory+"/",f) for f in listdir(directory) if isfile(join(directory+"/",f))]
    tas = []

    for ser_ta in serialized_tas:
        print(ser_ta)
        with open(ser_ta, mode='r', encoding='utf-8') as f:
            tas.append(core.text_annotation.TextAnnotation(f.read(),pipeline))
    return tas

def get_ta(path_to_file):
    """
        Returns a TextAnnotation object which has been deserialized from
        a json-serialized TextAnnotation at the given path.

        @param path_to_file json serialized TextAnnotation
        @return ta deserialized TA
    """
    #pipeline = local_pipeline.LocalPipeline()
    pipeline = remote_pipeline.RemotePipeline()
    ta = None
    with open(path_to_file,"r",encoding='utf-8') as f:
        ta = core.text_annotation.TextAnnotation(f.read(),pipeline)
    return ta

def serialize_tas(tas, directory):
    """
        Serialize list of TextAnnotations to a given directory.

        @param directory the path to where the serialized TAs should be written.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    for ta in tas:
        filename = directory + "/" + ta.id
        with open(filename, mode="w", encoding="utf-8") as f:
            print("Writing %s to file."%filename)
            json.dump(ta.as_json, f, indent=4, ensure_ascii=False)
        
def outlinks(outlinks_file):
    """
        Creates a map of titles to hyperlinks on that title's page. This used
        to calculate the number of times two pages link to each other.

        @param: outlinks_file tsv where column 1 is a Wikipedia title and column 2
                              is a space separated string of outlinks from that page
        @return: outlinks_map a map from string to list of strings
    """
    outlinks_map = {}
    with open(outlinks_file, "r") as f:
        # used to fix lines which have titles with a "\n" character
        prev_title = None
        for line in f.readlines():
            spline = line.split("\t")

            # there is at least one case where a hyperlink contains "\n"
            # which creates an erroneous line. 
            if len(spline) == 1:
                outlinks_map[prev_title].extend(spline[0].strip().split(" "))
                continue
            # this is handling the case where there is "\t" in one of the titles
            # this causes the list of titles to be split into two fields
            elif len(spline) > 2:
                title = spline[0]
                outlinks = " ".join(spline[1:]).strip()
            else:
                title = spline[0]
                outlinks = spline[1].strip()
            if outlinks:
                outlinks_map[title] = outlinks.strip().split(" ")
                prev_title = title
    return outlinks_map
    
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

def save(fname, obj):
    """
        Serializes object to given filename.

        @param: fname, file to write pickle
        @param: obj, object to be serialized (pickled)
    """
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load(fname):
    """
        Loads a pickled object.

        @param: fname, filename of pickle to load.
        @return: deserialized object.
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)

class LORELEIKBLoader:
    """
        Class for loading the LORELEI knowledge base (KB).

        The class allows the KB data to be accessed in two ways, either
        using the unique KB id or using the surface of a mention which
        may map to multiple records in the kb.
    """
    def __init__(self, kbfile):
        # map of entity id to kb record
        self.kb = {}
        # map of surface form of mention to list of kb records to which
        # it may refer.
        self.name2ent = {}
        self._load_kb(kbfile)
    
    def _load_kb(self, kbfile):
        """
            Helper function which builds the primary resources of the class.
            Loads the LORELEI kb data which is to
            be found at the provided path. Checks if the source data has already
            been preprocessed and stored as a dictionary. If it has, then those
            files are loaded in, if not then the dictonaries are first built and
            saved, then loaded.

            @param: kbfile, the path to source kb which will be loaded
        """
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

def get_mid_wid_map():
    """
        Retrieves map of Freebase ids to Wikipedia page ids.
        First it looks for a pickle in resources/ if that isn't
        found, it builds the map from MID2WID and pickles it before
        returning the map.

        @return: map of mids to Wiki ids
    """
    if os.path.exists(MID2WID_PKL):
        logging.info("Mid to Wid map found. Loading pickle")
        return load_pkl(MID2WID_PKL)
    else: 
        logging.info("Mid to Wid map not found. Building map.")
        mid2wid = {}
        with open(MID2WID, "r") as f:
            for line in f.readlines():
                mid,wid = line.strip().split("\t")
                mid2wid[mid] = wid
        logging.info("Pickliing map to %s."%MID2WID_PKL)
        save_pkl(MID2WID_PKL,mid2wid)
        return mid2wid

def get_wid_mid_map():
    """
        Retrieves map of Wikipedia page ids to Freebase ids.
        First it looks for a pickle in resources/ if that isn't
        found, it builds the map from MID2WID and pickles it before
        returning the map.

        @return: map of Wiki ids to mids
    """
    if os.path.exists(WID2MID_PKL):
        logging.info("Wid to mid map found. Loading pickle")
        return load_pkl(WID2MID_PKL)
    else: 
        logging.info("Wid to mid map not found. Building map.")
        wid2mid = {}
        with open(MID2WID, "r") as f:
            for line in f.readlines():
                mid,wid = line.strip().split("\t")
                wid2mid[wid] = mid
        logging.info("Pickliing map to %s."%WID2MID_PKL)
        save_pkl(WID2MID_PKL,wid2mid)
        return wid2mid 

def id_to_title_map():
    """
        Returns a map from Wikipedia page ids to Wikipedia titles. First checks
        if a pickle is available, if not it generates the map, serializes it,
        and returns it.
        
        @return: Wiki page id to title map
    """
    # TODO: complete function
    m = load_pkl("/shared/preprocessed/upadhya3/enwiki-datamachine/idmap/enwiki-20170520.id2t.pkl")
    return m[0]
