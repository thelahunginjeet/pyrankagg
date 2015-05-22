from numpy import sum,asarray,abs

def spearman_footrule_distance(s,t):
    """
    Computes the Spearman footrule distance between two full lists of ranks:

        F(s,t) = (2/|S|^2)*sum( |s(i) - t(i)| )

    the normalized sum over all elements in a set of the absolute difference
    between the rank according to two different lists s and t.  As defined,
    0 <= F(s,t) <= 1.

    s,t should be array-like (lists are OK).

    If s,t are *not* full, this function should not be used.
    """
    # check that the lists are both full
    assert len(s) == len(t)
    return (2.0/len(s)**2)*sum(abs(asarray(s) - asarray(t)))
