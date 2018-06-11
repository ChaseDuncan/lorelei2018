import os
from urllib import parse
from data_utils import get_wikititle
from io_utils import LORELEIKBLoader, get_ta_dir
import sys
import logging
logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)

logging.info("logging works")
args = {}
args["indir"] = sys.argv[1]
evaluate_coherence = False
if len(sys.argv) > 2:
    if sys.argv[2] == "coherence":
        evaluate_coherence = True

kb = LORELEIKBLoader(kbfile="data/kb/data/entities.tab")
tas = get_ta_dir(args["indir"])

# total recall
total_hits = 0
# number of times p(t) yields the best candidate
prior_correct = 0
# p(t) accuracy at 2 and 3, respectively
second_correct = 0
third_correct = 0
# number of times only one candidate is generated
single_cand = 0
# number of times only one candidate is generated and it is correct
single_cand_correct = 0
# number of times coherence is correct
coh_correct = 0
# number of correctly linked nil entities
nil_correct = 0
# total number of examples in the gold data
total = 0

for docs_seen, docta in enumerate(tas):
    gold_entities = docta.get_view("WIKIFIER").cons_list
    cands2scores = docta.get_view("CANDGEN").labels_to_scores_array
    # hack so that zipping works below. if we are not evaluating coherence this list is not used
    cohscores = cands2scores
    if evaluate_coherence:
        cohscores = docta.get_view("COHERENCE").cons_list
    for gold_eid, cand2score, coh_con in zip(gold_entities, cands2scores, cohscores):
        total+=1
        gold_eid = gold_eid["label"]

        if gold_eid.startswith("NIL") or gold_eid.startswith("NULL"):
            if len(cand2score)==0:
                nil_correct+=1
            continue
        else:
            gold_eid = gold_eid.split("|")[0]
            wikititle = get_wikititle(kb.kb, gold_eid)

            if wikititle is None:
                continue

            goldtitle = wikititle[0].rsplit('/', 1)[-1]
            goldtitle = parse.unquote(goldtitle)

            if goldtitle in cand2score:
                total_hits+=1
                if evaluate_coherence and coh_con["label"]==goldtitle:
                    coh_correct+=1
            if len(cand2score.keys())>0:
                sorted_cands = sorted(cand2score, 
                                      key=lambda key: cand2score[key],
                                      reverse=True)
                if len(cand2score) == 1:
                    single_cand+=1
                    if sorted_cands[0] == goldtitle:
                        single_cand_correct+=1
                if sorted_cands[0] == goldtitle:
                    prior_correct +=1
                if len(sorted_cands) > 1 and sorted_cands[1] == goldtitle:
                    second_correct+=1
                if len(sorted_cands) > 2 and sorted_cands[2] == goldtitle:
                    third_correct+=1

acc_at_2 = prior_correct+second_correct
acc_at_3 = acc_at_2+third_correct

logging.info("total_hits %d/%d=%.3f", total_hits, total, total_hits / total)
logging.info("prior correct %d/%d=%.3f", prior_correct, total, prior_correct / total)
logging.info("acc_at_2 %d/%d=%.3f", acc_at_2, total, acc_at_2/ total)
logging.info("acc_at_3 %d/%d=%.3f", acc_at_3, total, acc_at_3/ total)
logging.info("only one cand %d/%d=%.3f",single_cand, total, single_cand / total)
logging.info("only one cand is correct %d/%d=%.3f",single_cand_correct, total, single_cand_correct / total)
if evaluate_coherence:
    logging.info("coherence correct %d/%d=%.3f", coh_correct, total, coh_correct / total)
logging.info("nil correct %d/%d=%.3f", nil_correct, total, nil_correct / total)
