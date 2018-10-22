#!/usr/bin/python
import math, string, sys, fileinput
import numpy as np

# this code computes the approximate entropy used to quantify the amount of
# regularity and the unpredictability of fluctuations over time-series data. In
# our context it will be used to check the patterns of start sesssions of each
# stream request of user, the more organized it seems the more the chances of
# be a bot: reference https://en.wikipedia.org/wiki/Approximate_entropy

def ApEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))
    N = len(U)
    return abs(_phi(m+1) - _phi(m))


# since I expect a bootnet to do tasks in a certain order if musisc tracks,
# artistics or albums seems to selected in a sorted manner this measure will
# increase the chance of our user being a bot
# reference: http://pythonfiddle.com/shannon-entropy-calculation/
# .
def range_bytes (): return range(256)

def range_printable(): return (ord(c) for c in string.printable)

def H(data, iterator=range_bytes):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy

def get_dist(seq_str):
    char_distance = []
    for i in range(1, len(seq_str)):
        char_distance.append(ord(seq_str[i]) - ord(seq_str[i-1]))
    return("".join(list(map(str,char_distance))))

# # testing fucntions
# if __name__=='__main__':
#     # testing entropy of alphabetical order
#     for seq_str in ['abcdefgh', 'acegikmo', 'magnus', 'lkjas;danbfddlk', 'aaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbb', 'sadfasdfasdf', '7&wS/p(']:
#         print ("%s: %f" % (seq_str, H(get_dist(seq_str), range_printable)))
#
#     # testing entropy for timestamp
#     U = np.array([2, 2.1, 2.2])
#     print ApEn(U, 2, 3)
#     #1.0996541105257052e-05
#     randU = np.array([3.2, 20.4, 145.6])
#     print (ApEn(randU, 2, 3))
#     #0.8626664154888908
#     import pdb; pdb.set_trace()  # XXX BREAKPOINT
