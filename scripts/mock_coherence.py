
def write_coherence_output_files(cand_file, output_dir):
    with open(cand_file, "r") as cf:
       with open(output_dir+cand_file+".coh","w") as o:
        for line in cand_file.readlines():
            sysid,mid,surf,docid,kbid,etype,mtype,conf,cands,scores = line.split("\t")
            coh_top = cands.split(" ")[0]
            scores = scores.strip()
            o.write("\t".join([sysid,mid,surf,docid,coh_top,etype,mtype,conf,cands,scores,scores,"n"])

            
if __name__=="__main__":
    import sys
    write_coherence_output_files(sys[1],sys[2])

