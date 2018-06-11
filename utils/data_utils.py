import logging
import pickle
import os
import argparse
import sys
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
    src_title = src.split("/")[-1].strip()
    dest_title = src.split("/")[-1].strip()
    # the title isn't in the outlinks map then it has no outlinks
    try:
        outlinks = outlinks_map[src_title]    
    except KeyError:
        return 0
    return outlinks.count(dest_title)

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


