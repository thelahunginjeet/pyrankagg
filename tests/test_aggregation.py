from pyrankagg import *
import numpy as np

class TestConversions:

    def steup(self):
        self.scorelist = [{'milk':1.4,'cheese':2.6,'eggs':1.2,'bread':3.0},
                          {'milk':2.0,'cheese':3.2,'eggs':2.7,'bread':2.9},
                          {'milk':2.7,'cheese':3.0,'eggs':2.5,'bread':3.5}]

    def test_rank_conversion(self):
        RA = rankagg.RankAggregator()
        r1 = RA.convert_to_ranks(self.scorelist[0])
        assert r1['milk'] == 3,'Item \'milk\' has the wrong rank!'
        assert r1['cheese'] == 2,'Item \'cheese\' has the wrong rank!'
        assert r1['eggs'] == 4,'Item \'eggs\' has the wrong rank!'
        assert r1['bread'] == 1,'Item \'bread\' has the wrong rank!'
        