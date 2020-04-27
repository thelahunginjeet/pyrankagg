from numpy import sum,asarray,abs,sign

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


def kendall_tau_distance(s,t):
    """
    Computes the Kendall tau distance between two full lists of ranks,
    which counts all discordant pairs (where s(i) < s(j) but t(i) > t(j),
    or vice versa) and divides by:

            k*(k-1)/2

    This is a slow version of the distance; a faster version can be
    implemented using a version of merge sort (TODO).

    s,t should be array-like (lists are OK).

    If s,t are *not* full, this function should not be used.
    """
    numDiscordant = 0
    for i in range(0,len(s)):
        for j in range(i+1,len(t)):
            if (s[i] < s[j] and t[i] > t[j]) or (s[i] > s[j] and t[i] < t[j]):
                numDiscordant += 1
    return 2.0*numDiscordant/(len(s)*(len(s)-1))


def partial_kendall(s1,s2,p=0.5):
    """
    Kendall distance with penalty p between two partial rank lists (lists with ties).
    This could be sped up with more efficient computation of the size of the sets R1,
    R2, and D. I normalize the distance by dividing by k*(k-1)/2.
    """
    cardD = 0
    cardR1 = 0
    cardR2 = 0
    k = len(s1)
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] < s1[j] and s2[i] > s2[j]:
                cardD += 1
            elif s1[i] == s1[j] and s2[i] != s2[j]:
                cardR1 += 1
            elif s2[i] == s2[j] and s1[i] != s1[j]:
                cardR2 += 1
    return (2/(k*(k-1)))*(cardD + p*(cardR1 + cardR2))


def partial_hausdorff(s1,s2):
    """
    Hausdorff distance for partial rank lists; as in partial_kendall, could be
    sped up by more efficient set size calculation.  I normalize the distance by
    dividing by k*(k-1)/2.
    """
    cardD = 0
    cardR1 = 0
    cardR2 = 0
    k = len(s1)
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] < s1[j] and s2[i] > s2[j]:
                cardD += 1
            elif s1[i] == s1[j] and s2[i] != s2[j]:
                cardR1 += 1
            elif s2[i] == s2[j] and s1[i] != s1[j]:
                cardR2 += 1
    return (2./(k*(k-1)))*(cardD + max(cardR1,cardR2))
