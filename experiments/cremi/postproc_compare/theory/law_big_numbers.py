import numpy as np
import scipy.special as sci_spec

def flip_probability(nb_edges_in_boundary, eta):
    half = int(nb_edges_in_boundary / 2)
    N = nb_edges_in_boundary

    a = sci_spec.comb(N, half+1, exact=True)
    b = sci_spec.hyp2f1(1, 1-half,int((N+4)/2), eta/(eta-1))
    c = np.power(1-eta, half-1)
    d = np.power(eta, half + 1)

    return a*b*c*d

def flip_probability_2(nb_edges_in_boundary, eta):
    half = int(nb_edges_in_boundary / 2)
    N = nb_edges_in_boundary

    a = sci_spec.comb(N, half, exact=True)
    b = sci_spec.hyp2f1(1, -half,int((N+2)/2), eta/(eta-1))
    c = np.power(-(eta-1)*eta, half)

    return a*b*c


pin = 0.2

for N in range(2,100,4):
    print(N, "Neg:\t", flip_probability(N, pin), "\tEqual: ", flip_probability_2(N, pin))



