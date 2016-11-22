"""
Methods for measuring inter-ranker agreement.
"""

from numpy import zeros

def icc(ranks):
    '''
    Accepts a list of lists/arrays of ranks and compute the degree of
    inter-ranker agreement via the Interclass Correlation Coefficient.

    Values of the ICC close to 1.0 indicate unanimity, and those close to
    zero indicate complete lack of consensus.

    INPUT:
        ranks: list of lists, required
            ordering of the ranklists must be the same for all rankers

    OUTPUT:
        icc : float
            Interclass Correlation Coefficient
    '''
    K = len(ranks)
    N = len(ranks[0])
    rij = zeros((N,K))
    for i in range(N):
        for j in range(K):
            rij[i,j] = ranks[j][i]
    # average item ranks
    xnbar = rij.sum(axis=1)/K
    # overall mean
    xbar = xnbar.sum()/N
    # compute s^2
    s2 = (1.0/(K*N))*((rij - xbar)**2).sum()
    # compute the icc
    d1 = 1.0/(K-1)
    t1 = (((xnbar-xbar)**2).sum())/N
    icc = K*d1*t1/s2 - d1
    return icc


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
