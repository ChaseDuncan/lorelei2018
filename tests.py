from utils.io_utils import get_ta, get_ta_dir, serialize_tas, outlinks
from utils.data_utils import count_outlinks
from models.star_model import StarModel

#tas = get_ta_dir("/shared/corpora/corporaWeb/lorelei/dryrun-2018/il2/ner/setE-results")
#serialize_tas(tas,"nertas-copy")

#ta = get_ta("/shared/bronte/upadhya3/IL5_DF_020521_20170505_H0040MWIB.json")
tas = [get_ta("/shared/bronte/upadhya3/IL5_DF_020521_20170505_H0040MWIB.json")]


#tas = get_ta_dir("nertas-copy")
outlinks_file = "/shared/preprocessed/cddunca2/wikipedia/outlinks.t2t"
outlinks_map = outlinks(outlinks_file)
model = StarModel(outlinks_map=outlinks_map)
for ta in tas:
    model.add_view(ta)




