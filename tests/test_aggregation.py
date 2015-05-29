from pyrankagg import *
import numpy as np

class TestFullListAggregation:

    def setup(self):
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
        r2 = RA.convert_to_ranks(self.scorelist[1])
        assert r2['milk'] == 4,'Item \'milk\' has the wrong rank!'
        assert r2['cheese'] == 1,'Item \'cheese\' has the wrong rank!'
        assert r2['eggs'] == 3,'Item \'eggs\' has the wrong rank!'
        assert r2['bread'] == 2,'Item \'bread\' has the wrong rank!'
        r3 = RA.convert_to_ranks(self.scorelist[2])
        assert r3['milk'] == 3,'Item \'milk\' has the wrong rank!'
        assert r3['cheese'] == 2,'Item \'cheese\' has the wrong rank!'
        assert r3['eggs'] == 4,'Item \'eggs\' has the wrong rank!'
        assert r3['bread'] == 1,'Item \'bread\' has the wrong rank!'


    def test_borda_aggregation(self):
        FLRA = rankagg.FullListRankAggregator()
        aggRanks = FLRA.aggregate_ranks(self.scorelist,areScores=True,method='borda')
        assert aggRanks['milk'] == 3,'Item \'milk\' has the wrong aggregate rank!'
        assert aggRanks['cheese'] == 2,'Item \'cheese\' has the wrong aggregate rank!'
        assert aggRanks['eggs'] == 4,'Item \'eggs\' has the wrong aggregate rank!'
        assert aggRanks['bread'] == 1,'Item \'bread\' has the wrong aggregate rank!'


    def test_footrule_aggregation(self):
        FLRA = rankagg.FullListRankAggregator()
        aggRanks = FLRA.aggregate_ranks(self.scorelist,areScores=True,method='spearman')
        assert aggRanks['milk'] == 3,'Item \'milk\' has the wrong aggregate rank!'
        assert aggRanks['cheese'] == 2,'Item \'cheese\' has the wrong aggregate rank!'
        assert aggRanks['eggs'] == 4,'Item \'eggs\' has the wrong aggregate rank!'
        assert aggRanks['bread'] == 1,'Item \'bread\' has the wrong aggregate rank!'

    
    def test_local_kemenization(self):
        FLRA = rankagg.FullListRankAggregator()
        aggRanks = FLRA.aggregate_ranks(self.scorelist,areScores=True,method='borda')
        ranklist = [FLRA.convert_to_ranks(s) for s in self.scorelist]
        lkRanks = FLRA.locally_kemenize(aggRanks,ranklist)
        print lkRanks
        assert lkRanks['milk'] == 3,'Item \'milk\' has the wrong aggregate rank!'
        assert lkRanks['cheese'] == 2,'Item \'cheese\' has the wrong aggregate rank!'
        assert lkRanks['eggs'] == 4,'Item \'eggs\' has the wrong aggregate rank!'
        assert lkRanks['bread'] == 1,'Item \'bread\' has the wrong aggregate rank!'
