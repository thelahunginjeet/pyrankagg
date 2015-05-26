from pyrankagg import *
import numpy as np

class TestConversions:

    def setup(self):
        self.scorelist = [{'milk':1.4,'cheese':2.6,'eggs':1.2,'bread':3.0},
                          {'milk':2.0,'cheese':3.2,'eggs':2.7,'bread':2.9},
                          {'milk':2.7,'cheese':3.0,'eggs':2.5,'bread':3.5}]
   

    def test_spearman_footrule(self):
        RA = rankagg.RankAggregator()
        s = RA.convert_to_ranks(self.scorelist[0])
        t = RA.convert_to_ranks(self.scorelist[1])
        sf = metrics.spearman_footrule_distance(s.values(),t.values())
        assert (sf - 0.5) < 1e-08,'Footrule distance is wrong!'


    def test_kendall_tau(self):
        ranks1 = [1,2,3,4,5]
        ranks2 = [3,4,1,2,5]
        kt = metrics.kendall_tau_distance(ranks1,ranks2)
        assert (kt - 0.4) < 1e-08,'Kendall tau distance is wrong!'
