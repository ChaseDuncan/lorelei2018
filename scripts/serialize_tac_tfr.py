# -*- coding: utf-8 -*-

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logger = logging.getLogger(__name__)

from utils.data_utils import *
from utils.io_utils import *

""" Create a TFRecord from 2016/2017 TAC EDL data which has been annotated by Nitish's
EL system. 

1. Convert each TA in the directory into Documents and store them in a list.
2. Instatiate a CoherenceFeatureExtractor with the appropriate parameters.
3. Pass the above to io_utils.write_tfrecord
"""
YEAR = sys.argv[1]
#TA_DIR = "/shared/experiments/cddunca2/tac-nitish-pred-tas-"+YEAR+"-test/"
TA_DIR = "/shared/experiments/cddunca2/tac-nitish-pred-tas-"+YEAR+"/"
TFR = "data/tfr/nitish_"+YEAR+".tfrecord"

logger.info("Loading text annotations.")
tas = get_ta_dir(TA_DIR)
documents = []
logger.info("Converting text annotations to Documents.")
max_cands=20
for ta in tas:
    el_view = ta.get_view("NEUREL")
    gold_view = ta.get_view("ELGOLD")
    doc = Document(el_view,max_cands,gold_view)
    if doc.m > 0:
        documents.append(doc)

logger.info("Calculating accuracy at 1")
total=0.0
correct=0.0
for doc in documents:
    for mention in doc.mentions:
        total+=1.0
        if mention.gold == 0:
            correct+=1.0
print("accuracy: %f"%(correct/total))

#logger.info("Extracting features.")
#
#featureExtractor = CoherenceFeatureExtractor(4,11,max_cands)
#logging.info("Writing TFRecord")
#write_tfrecord(documents, TFR, featureExtractor)

