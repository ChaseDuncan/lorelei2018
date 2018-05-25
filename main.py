from utils.io_utils import get_ta_dir, serialize_tas, outlinks
from utils.data_utils import count_outlinks

tas = get_ta_dir("/shared/corpora/corporaWeb/lorelei/dryrun-2018/il2/ner/setE-results")

ta = tas[0]
outlinks_file = "/shared/preprocessed/cddunca2/wikipedia/outlinks.t2t"
outlinks_map = outlinks(outlinks_file)

wiki_view = ta.get_view("WIKIFIER")

