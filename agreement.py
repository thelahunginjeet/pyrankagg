"""
Methods for measuring inter-ranker agreement.
"""

from numpy import zeros

def kendallW(ranks):
    '''
    Accepts a list of lists/arrays of ranks and computes the
    degree of inter-ranker agreement via Kendall's W.

    Values of W close to 1.0 indicate unanimity, and those clost to zero
    indicate complete lack of consensus.

    INPUT:
        ranks : list of lists, required
            order of the ranklists must be the same for all rankers

    OUTPUT:
        W : float
            Kendall's W
    '''
    # build up the rij matrix
    m = len(ranks)
    n = len(ranks[0])
    rij = zeros((n,m))
    for i in range(n):
        for j in range(m):
            rij[i,j] = ranks[j][i]
    # total rank
    Ri = rij.sum(axis=1)
    # mean total rank
    Rbar = Ri.sum()/n
    # sum of squared deviations
    S = ((Ri - Rbar)**2).sum()
    # Kendall's W
    W = 12.0*S/((m**2)*n*(n**2 - 1))
    return W
