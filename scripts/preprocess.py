import json
from os import listdir
from os.path import isfile, join


# directory where parsed Wikipedia is stored in json files
WIKI_DUMP_DIR="/shared/preprocessed/wikiparser/wikiExtractorJsonPages/"

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
            print("%f complete"%(proc_ct/float(num_files)))
        with open(wikifile, "r") as f:
            for json_str in f.readlines():
                try:
                    wiki_page = json.loads(json_str.strip())
                    out.write(wiki_page["wikiTitle"] + "\t" + " ".join(wiki_page["hyperlinks"].values()) + "\n")
                except:
                    print("failed to parse " + wikifile)
                    out.flush()
            out.flush()

if __name__=="__main__":
    outlink_counts("/shared/preprocessed/cddunca2/wikipedia/outlinks.t2t")
