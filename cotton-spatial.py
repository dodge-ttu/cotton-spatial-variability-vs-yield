import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean_poly_eq(coefficients, dec_dig):
    n = len(coefficients)
    degs = [i for i in range(n)]
    coefficients = [round(i, dec_dig) for i in coefficients]
    coefficients.reverse()
    pieces = []
    for (cof, deg) in zip(coefficients, degs):
        if deg == 0:
            a = ' + {0}'.format(cof)
            pieces.append(a)
        else:
            a = '{0} x^{1} '.format(cof, deg)
            pieces.append(a)

    equation = 'y = ' + ''.join(pieces[::-1])

    return equation


def get_poly_hat(x_values, y_values, poly_degree):
    coeffs = np.polyfit(x_values, y_values, poly_degree)
    poly_eqn = np.poly1d(coeffs)

    y_bar = np.sum(y_values) / len(y_values)
    ssreg = np.sum((poly_eqn(x_values) - y_bar) ** 2)
    sstot = np.sum((y_values - y_bar) ** 2)
    r_square = ssreg / sstot

    return (coeffs, poly_eqn, r_square)

