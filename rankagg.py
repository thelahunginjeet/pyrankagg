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
from .assignment import linear_assignment
from .metrics import kendall_tau_distance
from numpy import zeros,abs,exp,sort,zeros_like,argmin,delete,mod,mean,median
from numpy.random import permutation
from operator import itemgetter
from scipy.stats import binom,gmean
import copy


def sort_by_value(d,reverse=False):
    """
    One of many ways to sort a dictionary by value.
    """
    return sorted(d.items(), key = itemgetter(1), reverse=reverse)


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
        y = list(zip(list(zip(*x))[0],range(1,len(x)+1)))
        ranks = {}
        for t in y:
            ranks[t[0]] = t[1]
        return ranks


    def item_ranks(self,rank_list):
        """
        Accepts an input list of ranks (each item in the list is a dictionary of item:rank pairs)
        and returns a dictionary keyed on item, with value the list of ranks the item obtained
        across all entire list of ranks.
        """
        item_ranks = {}.fromkeys(rank_list[0])
        for k in item_ranks:
            item_ranks[k] = [x[k] for x in rank_list]
        return item_ranks


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


class PartialListRankAggregator(RankAggregator):
    """
    Performs rank aggregation, using a variety of methods, for partial lists
    (all items were all ranked by all experts).  Borda methods that are prefixed with 't'
    refer to aggregate statistics on truncated Borda counts (so all unranked items are
    given rank = max(rank of ranked items) + 1), effectively meaning they will have scores
    equal to zero.
    """
    def __init__(self):
        super(RankAggregator,self).__init__()
        # method dispatch
        self.mDispatch = {'borda':self.borda_aggregation}


    def aggregate_ranks(self,experts,method='borda',stat='mean'):
        """
        Combines the ranks in the list experts to obtain a single set of aggregate ranks.
        Currently operates only on ranks, not scores.

        INPUT:
        ------
            experts: list of dictionaries, required
                each element of experts should be a dictionary of item:rank
                pairs

            method: string, optional
                which method to use to perform the rank aggregation

            stat: string, optional
                statistic used to combine Borda scores; only relevant for Borda
                aggregation
        """
        agg_ranks = {}
        scores = {}
        # if we are using any of the borda scores, we have to supplement the ranklist
        #   to produce dummy (tied) ranks for all the unranked items.  Scores are
        #   returned because of the lieklihood of ties with rankers that only rank
        #   a few items
        if method in self.mDispatch:
            if ['borda'].count(method) > 0:
                # need to convert to truncated Borda lists
                supp_experts = self.supplement_experts(experts)
                scores,agg_ranks = self.mDispatch['borda'](supp_experts,stat)
            else:
                # does not use stat (not a borda method), does not supplement ranklists
                scores,agg_ranks = self.mDispatch[method](experts)
        else:
            print('ERROR: method',method,'invalid.')
        return scores,agg_ranks


    def supplement_experts(self,experts):
        """
        Converts partial lists to full lists by supplementing each expert's ranklist with all
        unranked items, each item having rank max(rank) + 1 (different for different experts).
        (This has the effect of converting partial Borda lists to full lists via truncated
        count).
        """
        supp_experts = []
        # get the list of all the items
        all_items = list(frozenset().union(*[list(x.keys()) for x in experts]))
        for rank_dict in experts:
            new_ranks = {}
            max_rank = max(rank_dict.values())
            for item in all_items:
                if item in rank_dict:
                    new_ranks[item] = rank_dict[item]
                else:
                    new_ranks[item] = max_rank + 1
            supp_experts.append(new_ranks)
        return supp_experts


    def borda_aggregation(self,supp_experts,stat):
        """
        Rank is equal to mean rank on the truncated lists; the statistic is also returned,
        since in a situation where rankers only rank a few of the total list of items, ties are
        quite likely and some ranks will be arbitrary.

        Choices for the statistic are mean, median, and geo (for geometric mean).
        """
        scores = {}
        stat_dispatch = {'mean':mean,'median':median,'geo':gmean}
        # all lists are full, so any set of dict keys will do
        for item in supp_experts[0].keys():
            vals_list = [x[item] for x in supp_experts]
            scores[item] = stat_dispatch[stat](vals_list)
        # in order to use convert_to_ranks, we need to manipulate the scores
        #   so that higher values = better; right now, lower is better.  So
        #   just change the sign of the score
        flip_scores = copy.copy(scores)
        for k in flip_scores:
            flip_scores[k] = -1.0*flip_scores[k]
        agg_ranks = self.convert_to_ranks(flip_scores)
        return scores,agg_ranks


class FullListRankAggregator(RankAggregator):
    """
    Performs rank aggregation, using a variety of methods, for full lists
    (all items are ranked by all experts).
    """
    def __init__(self):
        super(RankAggregator,self).__init__()
        # used for method dispatch
        self.mDispatch = {'borda':self.borda_aggregation,'spearman':self.footrule_aggregation,
                'median':self.median_aggregation,'highest':self.highest_rank,'lowest':self.lowest_rank,
                'stability':self.stability_selection,'exponential':self.exponential_weighting,
                'sborda':self.stability_enhanced_borda,'eborda':self.exponential_enhanced_borda,
                'robust':self.robust_aggregation,'rrobin':self.round_robin}


    def aggregate_ranks(self,experts,areScores=True,method='borda',*args):
        """
        Combines the ranks in the list experts to obtain a single
        set of aggregate ranks.  Can operate on either scores
        or ranks; scores are assumed to always mean higher=better.

        INPUT:
        ------
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
        if method in self.mDispatch:
            aggRanks = self.mDispatch[method](ranklist)
        else:
            print('ERROR: method \'',method,'\'  invalid.')
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


    def median_aggregation(self,rank_list):
        """
        Computes median aggregate rank.  Start's each items score M_i at zero,
        and then for each rank 1,...,M, the item's score is incremented by the
        number of lists in which it has that rank.  The first item over L/2
        gets rank 1, the next rank 2, etc.  Ties are broken randomly.

        Aggregate ranks are returned as a dictionary.
        """
        theta = 1.0*len(rank_list)/2
        # this hold the scores
        M = {}.fromkeys(rank_list[0])
        for k in M:
            M[k] = 0
        # lists of item ranks (across all lists) for each item
        item_ranks = self.item_ranks(rank_list)
        # this holds the eventual voted ranks
        med_ranks = {}.fromkeys(rank_list[0])
        # the next rank that needs to be assigned
        next_rank = 1
        # once the next-to-last item has a rank, assign the one remaining item
        #   the last rank
        for r in range(1,len(med_ranks)):
            # increment scores
            for k in M:
                M[k] += item_ranks[k].count(r)
            # check if any of the items are over threshold; randomly permute
            #   all over-threshold items for rank assignment (tie breaking)
            items_over = list(permutation([k for k in M if M[k] >= theta]))
            for i in range(len(items_over)):
                med_ranks[items_over[i]] = next_rank + i
                M.pop(items_over[i])
            next_rank = next_rank + len(items_over)
            if next_rank == len(med_ranks):
                break
        # if we are out of the loop, there should only be one item left to
        #   rank
        med_ranks[list(M.keys())[0]] = len(med_ranks)
        return med_ranks


    def highest_rank(self,rank_list):
        """
        Each item is assigned the highest rank it obtains in all of the
        rank lists.  Ties are broken randomly.
        """
        min_ranks = {}.fromkeys(rank_list[0])
        item_ranks = self.item_ranks(rank_list)
        for k in min_ranks:
            min_ranks[k] = min(item_ranks[k])
        # sort the highest ranks dictionary by value (ascending order)
        pairs = sort_by_value(min_ranks)
        # assign ranks in order
        pairs = list(zip(list(zip(*pairs))[0],range(1,len(item_ranks)+1)))
        # over-write the min_ranks dict with the aggregate ranks
        for (item,rank) in pairs:
            min_ranks[item] = rank
        return min_ranks


    def lowest_rank(self,rank_list):
        """
        Each item is assigned the lowest rank it obtains in all of the rank
        lists.  Ties are broken randomly.
        """
        max_ranks = {}.fromkeys(rank_list[0])
        item_ranks = self.item_ranks(rank_list)
        for k in max_ranks:
            max_ranks[k] = max(item_ranks[k])
        # sort the worst ranks dictionary by value (ascending order)
        pairs = sort_by_value(max_ranks)
        # assign ranks in order
        pairs = list(zip(list(zip(*pairs))[0],range(1,len(item_ranks)+1)))
        # over-write the max_ranks dict with the aggregate ranks
        for (item,rank) in pairs:
            max_ranks[item] = rank
        return max_ranks


    def stability_selection(self,rank_list,theta=None):
        """
        For every list in which an item is ranked equal to or higher than theta
        (so <= theta), it recieves one point.  Items are then ranked from most to
        least points and assigned ranks.  If theta = None, then it is set equal to
        half the number of items to rank.
        """
        if theta is None:
            theta = 1.0*len(rank_list[0])/2
        scores = {}.fromkeys(rank_list[0])
        item_ranks = self.item_ranks(rank_list)
        for k in scores:
            scores[k] = sum([i <= theta for i in item_ranks[k]])
        return self.convert_to_ranks(scores)


    def exponential_weighting(self,rank_list,theta=None):
        """
        Like stability selection, except items are awarded points according to
        Exp(-r/theta), where r = rank and theta is a threshold.  If theta = None,
        then it is set equal to half the number of items to rank.
        """
        if theta is None:
            theta = 1.0*len(rank_list[0])/2
        scores = {}.fromkeys(rank_list[0])
        item_ranks = self.item_ranks(rank_list)
        for k in scores:
            scores[k] = exp([-1.0*x/theta for x in item_ranks[k]]).sum()
        return self.convert_to_ranks(scores)


    def stability_enhanced_borda(self,rank_list,theta=None):
        """
        For stability enhanced Borda, each item's Borda score is multiplied
        by its stability score and larger scores are assigned higher ranks.
        """
        if theta is None:
            theta = 1.0*len(rank_list[0])/2
        scores = {}.fromkeys(rank_list[0])
        N = len(scores)
        item_ranks = self.item_ranks(rank_list)
        for k in scores:
            borda = sum([N - x for x in item_ranks[k]])
            ss = sum([i <= theta for i in item_ranks[k]])
            scores[k] = borda*ss
        return self.convert_to_ranks(scores)


    def exponential_enhanced_borda(self,rank_list,theta=None):
        """
        For exponential enhanced Borda, each item's Borda score is multiplied
        by its exponential weighting score and larger scores are assigned higher
        ranks.
        """
        if theta is None:
            theta = 1.0*len(rank_list[0])/2
        scores = {}.fromkeys(rank_list[0])
        N = len(scores)
        item_ranks = self.item_ranks(rank_list)
        for k in scores:
            borda = sum([N - x for x in item_ranks[k]])
            expw = exp([-1.0*x/theta for x in item_ranks[k]]).sum()
            scores[k] = borda*expw
        return self.convert_to_ranks(scores)


    def robust_aggregation(self,rank_list):
        """
        Implements the robust rank aggregation scheme of Kolde, Laur, Adler,
        and Vilo in "Robust rank aggregation for gene list integration and
        meta-analysis", Bioinformatics 28(4) 2012.  Essentially compares
        order statistics of normalized ranks to a uniform distribution.
        """
        def beta_calc(x):
            bp = zeros_like(x)
            n = len(x)
            for k in range(n):
                b = binom(n,x[k])
                for l in range(k,n):
                    bp[k] += b.pmf(l+1)
            return bp
        scores = {}.fromkeys(rank_list[0])
        item_ranks = self.item_ranks(rank_list)
        N = len(scores)
        # sort and normalize the ranks, and then compute the item score
        for item in item_ranks:
            item_ranks[item] = sort([1.0*x/N for x in item_ranks[item]])
            # the 1.0 here is to make *large* scores correspond to better ranks
            scores[item] = 1.0 - min(beta_calc(item_ranks[item]))
        return self.convert_to_ranks(scores)


    def round_robin(self,rank_list):
        """
        Round Robin aggregation.  Lists are given a random order.  The highest
        ranked item in List 1 is given rank 1 and then removed from consideration.
        The highest ranked item in List 2 is given rank 2, etc.  Continue until
        all ranks have been assigned.
        """
        rr_ranks = {}.fromkeys(rank_list[0])
        N = len(rr_ranks)
        next_rank = 1
        next_list = 0
        # matrix of ranks
        rr_matrix = zeros((len(rr_ranks),len(rank_list)))
        items = list(rr_ranks.keys())
        # fill in the matrix
        for i in range(len(items)):
            for j in range(len(rank_list)):
                rr_matrix[i,j] = rank_list[j][items[i]]
        # shuffle the columns to randomize the list order
        rr_matrix = rr_matrix[:,permutation(rr_matrix.shape[1])]
        # start ranking
        while next_rank < N:
            # find the highest rank = lowest number in the row
            item_indx = argmin(rr_matrix[:,next_list])
            item_to_rank = items[item_indx]
            # rank the item and remove it from the itemlist and matrix
            rr_ranks[item_to_rank] = next_rank
            rr_matrix = delete(rr_matrix,item_indx,axis=0)
            items.remove(item_to_rank)
            next_rank += 1
            next_list = mod(next_list + 1,len(rank_list))
        # should only be one item left
        rr_ranks[items[0]] = N
        return rr_ranks


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
                for j in range(0,len(p)):
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
        items = list(aggranks.keys())
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
        for i in range(0,len(items)-1):
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
        for i in range(0,len(items)):
            lkranks[items[i]] = sigma[i]
        return lkranks
