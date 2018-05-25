import numpy as np
import logging
from utils.data_utils import count_outlinks



logger = logging.getLogger(__name__)

class StarModel(object):
    def __init__(self, outlinks_map=None, fb_relations_map=None, cooccurrence_map=None):
    
        self.m = 1
        self.k = 5
        self.w_unary = np.array([0.5,0.5])
        self.w_pairwise = np.array([0.05,0.10,0.15,0.20,0.25,0.25])
        self.outlinks_map = None

        if outlinks_map is not None:
            self.outlinks_map = outlinks_map
            self.m+=5
    
    """
        Adds the star model view to a TA.

        @param: ta, the textannotation to which to add a view
    """
    def add_view(self,ta):
        wiki_view = ta.get_view("CANDGEN")
        wiki_labels_to_scores = wiki_view.get_labels_to_scores()
        wiki_cons = wiki_view.get_cons()
        coh_cons = []

        # calculate attention score for each candidate for each mention
        n = len(wiki_labels_to_scores)
        indices = np.arange(n, -1, -1)
        for i in range(n):
            # mi is the current mention for which to perform coherence
            mi = wiki_labels_to_scores[i]
            scores = np.zeros(len(mi))

            # get list of other mentions which are not mi
            other_indices = [j for j in range(n) if j!=i]
            other_mentions = [wiki_labels_to_scores[idx] for idx in other_indices]

            # calculate coherence score for each candidate title
            idx = 0
            print(mi)
            for k,v in mi.items():
                scores[idx] = self.unary_score(v) + self.attention_score(k,other_mentions)

            # argmax over the attention scores for each candidate
            # to find the most coherence candidate
            title = mi.keys()[np.argmax(scores)]  
            # create new constituent for coherence view
            coh_con = {"label":title,"score":np.max(scores),\
                "start":wiki_cons[i]["start"],"end":wiki_cons[i]["end"]}
            coh_cons.append(coh_con)

        view = {}
        view["viewType"] = "TokenLabelView"
        view["viewName"] = "COHERENCE"
        view["generator"] = "coherence-annotator"
        view["score"] = 1.0
        view["constituents"] = coh_con

    
    """
        Scores a Wikipedia candidate based on attending to the k most similar
        scores from other mentions in the text.

        @param: yi, the Wikipedia candidate to score
        @param: other_mentions, list of dicts of candidate scores for the other
                                mentions in the text
        @return a score for the candidate based on the attention model
    """
    def attention_score(self,yi,other_mentions):
        scores = np.zeros(len(other_mentions))
        idx = 0
        for mention in other_mentions:
            scores[idx] = self.score_yi_mj(yi,mention)
            idx+=1
        return np.sum(scores[::-1].sort()[:self.k])


    """
        Takes a Wikipedia candidate yi for mention mi and list of candidates
        for mention mj and calculates the qij, the maximum support for
        mi by the candidates of mj.

        @param: yi, a candidate for ith mention in the text
        @param: mj_cands, dict of candidates for the jth mention in the text
    """
    def score_yi_mj(self, yi, mj_cands):
        scores = np.zeros(len(mj_cands))
        idx = 0
        for k,v in mj_cands.items():
            scores[idx] = self.unary_score(v) + self.pairwise_score(yi, k)
            idx+=1
        print("scores")
        print(scores)
        return np.max(scores)

    """
        Calculates the unary score for a candidate based on its ranker score.
        This is 

        @param: pi, the prior for the candidate, i.e. the ranker score
        @return: the weighted unary score of the candidate
    """
    def unary_score(self, pi):
        pi -= (.25*pi)
        phi = np.array([np.log(pi),np.log(1-pi)])
        return np.dot(phi,self.w_unary)

    """
        Calculates the similarity score between two Wikipedia titles.

        @param: yi, the first Wikipedia title
        @param: yj, the second Wikipedia title
        @return the similarity score between yi and yj
    """
    def pairwise_score(self, yi, yj):
        return np.dot(self.w_pairwise, self.feature_vec(yi,yj))


    """
        Creates a vector of pairwise features for two candidate Wikipedia titles.

        @param: yi, first title
        @param: yj, second title
        @return: vector representation of pairwise similarity of two titles
    """
    def feature_vec(self, yi, yj):
        # TODO: some caching here, probably
        fv = np.zeros(self.m)
        num_outlinks = \
            count_outlinks(yi,yj,self.outlinks_map) + \
            count_outlinks(yj,yi,self.outlinks_map)
        if num_outlinks >= 5:
            fv[4] = 1
        elif num_outlinks == 4:
            fv[3] = 1
        elif num_outlinks == 3:
            fv[2] = 1
        elif num_outlinks == 2:
            fv[1] = 1
        elif num_outlinks == 1:
            fv[0] = 1
        if yi == yj:
            fv[5] = 1

        return fv

