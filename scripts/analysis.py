from collections import defaultdict
import pickle
import operator

#GOLD_TAB = "il6_edl.tab"
GOLD_TAB = "il5_edl.tab"

APB_MIN = 20000001
WLL_MIN = 30000001
AUG_MIN = 71000001

SEEN = defaultdict(lambda: False)
COUNTS = {"GEO":0.0,"APB":0.0,"WLL":0.0,"AUG":0.0}
TOTALS = defaultdict(int)
NIL_COUNT = 0.0
NILS = defaultdict(int)

CTRY_CODE = defaultdict(int)
WIKI = {"enwiki":0.0,"rowiki":0.0}
ET_WIKI = 0
NIL_TYPES = defaultdict(int)
WIKI_TYPES = defaultdict(int)

ET_GEO = defaultdict(lambda: defaultdict(int))

KB = pickle.load(open("/home/cddunca2/lorelei2018/kb/kb.p","rb"))
MENTION_IN_WIKI = 0
LINKABLE = 0

with open(GOLD_TAB,"r") as f:
    lines = f.readlines()
    for line in lines:
        spline = line.split("\t")
        kbid = spline[4]
        typ = spline[5]

        if "NIL" in kbid:
            NIL_COUNT+=1
            NILS[kbid]+=1
            NIL_TYPES[typ]+=1
            continue
        if "|" in kbid:
            kbid = kbid.split("|")[0]
        exlink = KB[kbid][-1]
        #links = exlink.split("|")
        #wiki_links = [link for link in links if "en.wikipedia" in link]
        kbid_i = int(kbid)
        LINKABLE+=1
        if "en.wikipedia" in exlink:
            MENTION_IN_WIKI+=1
            WIKI_TYPES[typ]+=1
        if not SEEN[kbid_i]:
            CTRY_CODE[KB[kbid][12]]+=1
            if "en.wikipedia" in exlink:
                if KB[kbid][12] == "ET":
                    ET_WIKI += 1
                    f_class = KB[kbid][7]
                    f_code = KB[kbid][9]
                    ET_GEO[f_class][f_code]+=1
                WIKI["enwiki"]+=1
            elif "wikipedia" in exlink:
                WIKI["rowiki"]+=1

        if kbid_i < APB_MIN:
            TOTALS["GEO"]+=1
            if not SEEN[kbid_i]:
                COUNTS["GEO"]+=1
        elif kbid_i < WLL_MIN:
            TOTALS["APB"]+=1
            if not SEEN[kbid_i]:
                COUNTS["APB"]+=1
        elif kbid_i < AUG_MIN:
            TOTALS["WLL"]+=1
            if not SEEN[kbid_i]:
                COUNTS["WLL"]+=1
        else:
            TOTALS["AUG"]+=1
            if not SEEN[kbid_i]:
                COUNTS["AUG"]+=1

        SEEN[kbid_i] = True

Z = float(len(SEEN.keys()))+len(NILS.keys())
g = COUNTS["GEO"]/Z
ab = COUNTS["APB"]/Z
w = COUNTS["WLL"]/Z
ag = COUNTS["AUG"]/Z
n = len(NILS.keys())/Z

print("Entities")
print("GEO : %f\nAPB : %f\nWLL : %f\nAUG : %f\nNIL: %f\n"%(g,ab,w,ag,n))

Z = sum(TOTALS.values()) + NIL_COUNT

g = TOTALS["GEO"]/Z
ab = TOTALS["APB"]/Z
w = TOTALS["WLL"]/Z
ag = TOTALS["AUG"]/Z
n = sum(NILS.values())/Z

print("Mentions")
#print("GEO : %f\nAPB : %f\nWLL : %f\nAUG : %f"%(g,ab,w,ag))
print("GEO %d/%d : %f\nAPB %d/%d : %f\nWLL %d/%d : %f\nAUG %d/%d : %f\nNIL %d/%d : %f\n"%(TOTALS["GEO"],Z, \
                g,TOTALS["APB"],Z,ab,TOTALS["WLL"],Z,w,TOTALS["AUG"],Z,ag,sum(NILS.values()),Z,n))
print("NIL percentage : %f "%(NIL_COUNT / Z))

Z = float(len(SEEN.keys()))
print("Total entities : %d"%(Z))

total_wiki = sum(WIKI.values())

print("TOTAL entities in wiki : count - %d, per - %f"%(total_wiki, total_wiki / Z ))
print("EN wiki : count - %d, per - %f"%(WIKI["enwiki"], WIKI["enwiki"]/Z))
print("RO wiki : count - %d, per - %f"%(WIKI["rowiki"], WIKI["rowiki"]/Z))

sorted_CTRY_CODE = sorted(CTRY_CODE.items(), key=operator.itemgetter(1))

#for c in sorted_CTRY_CODE:
#    print("%s : %d"%(c[0],c[1]))

print("ET_WIKI : %d"%ET_WIKI)

print("NIL_TYPES")

for t in NIL_TYPES.keys():
    print("%s : %f"%(t,float(NIL_TYPES[t])/NIL_COUNT))

print("Geonames features for ET")

total = 0.0
f_totals = []

for k in ET_GEO.keys():
    #import pdb
    #pdb.set_trace()
    feature_total = sum(ET_GEO[k].values())
    total += feature_total
    f_totals.append((k,feature_total))
    print("Distribution for %s feature code"%k)
    for l in ET_GEO[k].keys():
        print("%s %d/%d = %f"%(l,ET_GEO[k][l],feature_total,ET_GEO[k][l]/float(feature_total)))

print("Distribution for all feature codes")
for f in f_totals:
    print("%s :%d, %f"%(f[0],f[1],f[1]/total))

print("Types in Wikipedia")
for k,v in WIKI_TYPES.items():
    print("%s %d/%d = : %f"%(k,v,MENTION_IN_WIKI,float(v)/MENTION_IN_WIKI))

print("Mentions in Wiki: %d"%MENTION_IN_WIKI)
print("Linkable: %d"%LINKABLE)
