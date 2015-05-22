"""
Python code for rank aggregation, for both full and partial lists.  For methods/algorithms
I have followed the paper

"Rank aggregation methods for the web" (2001) C. Dwork, R. Kumar, M. Naor, D. Sivakumar.
Proceedings of the 10th international conference on World Wide Web.

Created May 22, 2015

@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license (see LICENSE file).

Copyright (c) 2015, Kevin S. Brown
All rights reserved.
"""

from kbutil.listutil import sort_by_value

class RankAggregator(object):
    """
    Base class for full and partial list rank aggregation methods.  Should not be called
    directly except in testing situations; houses shared methods to both.
    """
    def __init__(self):
        pass

    def convert_to_ranks(self,scoreDict):
        """
        Accepts an input dictionary in which they keys are items to be ranked (numerical/string/etc.)
        and the values are scores, in which a higher score is better.  Returns a dictionary of 
        items and ranks, ranks in the range 1,...,n.
        """
        # default sort direction is ascending, so reverse (see sort_by_value docs)
        x = sort_by_value(scoreDict,True)
        y = zip(zip(*x)[0],range(1,len(x)+1))
        ranks = {}
        for t in y:
            ranks[t[0]] = t[1]
        return ranks

    def item_mapping(self,items):
        """
        Some methods need to do numerical work on arrays rather than directly using dictionaries.
        This function maps a list of items (they can be strings, ints, whatever) into 0,...,len(items).
        Both forward and reverse dictionaries are created and stored.
        """
        self.itemToIndex = {}
        indexToItem = {}
        next = 0
        for i in items:
            self.itemToIndex[i] = next
            self.indexToItem[next] = i
            next += 1
        return
        

class FullListRankAggregator(RankAggregator):
    """
    Performs rank aggregation, using a variety of methods, for full lists
    (all items are ranked by all experts).
    """
    def __init__(self):
        super(RankAggregator,self).__init__()
