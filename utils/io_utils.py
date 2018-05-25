import json
from os import listdir
from os.path import isfile, join

from ccg_nlpy import core, local_pipeline, remote_pipeline

filename = "/shared/experiments/cddunca2/tac-textannotations/en/v1/json/ENG_NW_001278_20131222_F000123EC.json"

"""
    Returns a list of TextAnnotation objects which are instatiated
    using the serialized json data in the directory parameter.

    @param directory path to directory with serialized TAs
    @return tas a list of TextAnnotations
"""
def get_ta_dir(directory):
    #pipeline = local_pipeline.LocalPipeline()
    pipeline = remote_pipeline.RemotePipeline()
    serialized_tas = [join(directory+"/",f) for f in listdir(directory) if isfile(join(directory+"/",f))]
    tas = []

    for ser_ta in serialized_tas:
        print(ser_ta)
        with open(ser_ta, mode='r', encoding='utf-8') as f:
            tas.append(core.text_annotation.TextAnnotation(f.read(),pipeline))
    return tas

"""
    Returns a TextAnnotation object which has been deserialized from
    a json-serialized TextAnnotation at the given path.

    @param path_to_file json serialized TextAnnotation
    @return ta deserialized TA
"""
def get_ta(path_to_file):
    #pipeline = local_pipeline.LocalPipeline()
    pipeline = remote_pipeline.RemotePipeline()
    ta = None
    with open(path_to_file,"r",encoding='utf-8') as f:
        ta = core.text_annotation.TextAnnotation(f.read(),pipeline)
    return ta

"""
    Serialize list of TextAnnotations to a given directory.

    @param directory the path to where the serialized TAs should be written.
"""
def serialize_tas(tas, directory):
    for ta in tas:
        filename = directory + "/" + ta.id
        with open(filename, mode="w", encoding="utf-8") as f:
            print("Writing %s to file."%filename)
            json.dump(ta.as_json, f, indent=4, ensure_ascii=False)
        
"""
    Creates a map of titles to hyperlinks on that title's page. This used
    to calculate the number of times two pages link to each other.

    @param: outlinks_file tsv where column 1 is a Wikipedia title and column 2
                          is a space separated string of outlinks from that page
    @return: outlinks_map a map from string to list of strings
"""
def outlinks(outlinks_file):
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
    
