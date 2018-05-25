"""
    Function for getting the number of outlinks a source title
    has to a destination title. This function is not symmetric.

    @param: src, the source title
    @param: dest, the destination title
    @return: the number of outlinks in Wikipedia form src to dest
"""
def count_outlinks(src, dest, outlinks_map):
    src_title = src.split("/")[-1].strip()
    dest_title = src.split("/")[-1].strip()
    outlinks = outlinks_map[src_title]    
    return outlinks.count(dest_title)

