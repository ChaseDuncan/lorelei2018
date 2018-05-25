from utils.io_utils import get_ta_dir, serialize_tas, outlinks
from utils.data_utils import count_outlinks
di = "nertas-copy"
tas = get_ta_dir(di)
ta = tas[0]

pdb.set_trace()
wiki_view = ta.get_view("TOKENS").copy()
labelsToScores = {}

wiki_view["viewName"] = "WIKIFIER"
wiki_view["viewData"]["viewName"] = "WIKIFIER"
cons = wiki_view.get_cons()

titles = ['a','b','c','d','e','f','g','h','i','j']
for con in cons:
    scores = np.random.randint(10,size=10)
    scores = scores/np.sum(scores)
    for k,v in zip(titles,scores):
        labelsToScores[k]=v

wiki_view["viewData"]["labelsToScores"] = labelsToScores
ta.view_dictionary["WIKIFIER"] = wiki_view

serialize_tas([ta],di)

