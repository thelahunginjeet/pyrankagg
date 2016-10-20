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
from linear_assignment import linear_assignment
from metrics import kendall_tau_distance
from numpy import zeros,abs
import copy

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
        self.indexToItem = {}
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
        # used for method dispatch
        self.mDispatch = {'borda':self.borda_aggregation,'spearman':self.footrule_aggregation}


    def aggregate_ranks(self,experts,areScores=True,method='borda'):
        """
        Combines the ranks in the list experts to obtain a single
        set of aggregate ranks.  Can operate on either scores
        or ranks; scores are assumed to always mean higher=better.

        INPUT:
            experts : list of dictionaries, required
                each element of experts should be a dictionary of item:score
                or item:rank pairs

            areScores : bool, optional
                set to True if the experts provided scores, False if they
                provide ranks

            method : string, optional
                which method to use to perform the rank aggregation.
                options include:

                    'borda': aggregate by computation of borda scores

                    'spearman' : use spearman footrule distance and
                                 bipartite graph matching
        """
        aggRanks = {}
        # if the input data is scores, we need to convert
        if areScores:
            ranklist = [self.convert_to_ranks(e) for e in experts]
        else:
            ranklist = experts
        # now dispatch on the string method
        if self.mDispatch.has_key(method):
            aggRanks = self.mDispatch[method](ranklist)
        else:
            print 'ERROR: method \'%\' invalid.'
        return aggRanks


    def borda_aggregation(self,ranklist):
        """
        Computes aggregate rank by Borda score.  For each item and list of ranks,
        compute:

                B_i(c) = # of candidates ranks BELOW c in ranks_i

        Then form:

                B(c) = sum(B_i(c))
        and sort in order of decreasing Borda score.

        The aggregate ranks are returned as a dictionary, as are in input ranks.
        """
        # lists are full, so make an empty dictionary with the item keys
        aggRanks = {}.fromkeys(ranklist[0])
        for item in aggRanks:
            aggRanks[item] = 0
        # now increment the Borda scores one list at a time
        maxRank = len(aggRanks)
        for r in ranklist:
            for item in r:
                aggRanks[item] += maxRank - r[item]
        # now convert the Borda scores to ranks
        return self.convert_to_ranks(aggRanks)



    def footrule_aggregation(self,ranklist):
        """
        Computes aggregate rank by Spearman footrule and bipartite graph
        matching, from a list of ranks.  For each candiate (thing to be
        ranked) and each position (rank) we compute a matrix

            W(c,p) = sum(|tau_i(c) - p|)/S

        where the sum runs over all the experts doing the ranking.  S is a 
        normalizer; if the number of ranks in the list is n, S is equal to
        0.5*n^2 for n even and 0.5*(n^2 - 1) for n odd.
        
        After constructing W(c,p), Munkres' algorithm is used for the linear 
        assignment/bipartite graph matching problem.
        """
        # lists are full so make an empty dictionary with the item keys
        items = ranklist[0].keys()
        # map these to matrix entries
        self.item_mapping(items)
        c = len(ranklist[0]) % 2
        scaling = 2.0/(len(ranklist[0])**2 - c)
        # thes are the positions p (each item will get a rank()
        p = range(1,len(items)+1)
        # compute the matrix
        W = zeros((len(items),len(items)))
        for r in ranklist:
            for item in items:
                taui = r[item]
                for j in xrange(0,len(p)):
                    delta = abs(taui - p[j])
                    # matrix indices
                    W[self.itemToIndex[item],j] += delta
        W = scaling*W
        # solve the assignment problem
        path = linear_assignment(W)
        # construct the aggregate ranks
        aggRanks = {}
        for pair in path:
            aggRanks[self.indexToItem[pair[0]]] = p[pair[1]]
        return aggRanks



    def locally_kemenize(self,aggranks,ranklist):
        """
        Performs a local kemenization of the ranks in aggranks and the list
        of expert rankings dictionaries in ranklist.  All rank lists must be full.
        The aggregate ranks can be obtained by any process - Borda, footrule,
        Markov chain, etc.  Returns the locally kemeny optimal aggregate
        ranks.

        A list of ranks is locally Kemeny optimal if you cannot obtain a
        lower Kendall tau distance by performing a single transposition
        of two adjacent ranks.
        """
        # covert ranks to lists, in a consistent item ordering, to use
        # the kendall_tau_distance in metrics.py
        lkranks = {}
        items = aggranks.keys()
        sigma = [aggranks[i] for i in items]
        tau = []
        for r in ranklist:
            tau.append([r[i] for i in items])
        # starting distance and distance of permuted list
        SKorig = 0
        # initial distance
        for t in tau:
            SKorig += kendall_tau_distance(sigma,t)
        # now try all the pair swaps
        for i in xrange(0,len(items)-1):
            SKperm = 0
            j = i + 1
            piprime = copy.copy(sigma)
            piprime[i],piprime[j] = piprime[j],piprime[i]
            for t in tau:
                SKperm += kendall_tau_distance(piprime,t)
            if SKperm < SKorig:
                sigma = piprime
                SKorig = SKperm
        # rebuild the locally kemenized rank dictionary
        for i in xrange(0,len(items)):
            lkranks[items[i]] = sigma[i]
        return lkranks
