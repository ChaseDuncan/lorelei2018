import pickle

DUMP_DIR = "/home/cddunca2/lorelei2018/kb/"
KB_DICT = {}
def pickle_kb(kb_file):
    with open(kb_file, "r") as kb:
        lines = kb.readlines()
        total = len(lines)
        count = 0.0
        for line in lines:
            count+=1
            if count % 10000 == 0:
                print("%f complete"%(count/total))
            entry = line.split("\t")
            KB_DICT[entry[2]] = entry

    pickle.dump(KB_DICT,open(DUMP_DIR+"kb.p","wb"))

if __name__=="__main__":
    import sys
    pickle_kb(sys.argv[1])
