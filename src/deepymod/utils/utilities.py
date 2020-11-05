""" Useful tools, such as combining string matrix product and computing the terms in the library """
from itertools import product, combinations, chain
import torch
import torch.nn as nn


def string_matmul(list_1, list_2):
    ''' Matrix multiplication with strings.'''
    prod = [element[0] + element[1] for element in product(list_1, list_2)]
    return prod


def terms_definition(poly_list, deriv_list):
    ''' Calculates which terms are in the library.'''
    if len(poly_list) == 1:
        theta = string_matmul(poly_list[0], deriv_list[0]) # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:
        theta_uv = list(chain.from_iterable([string_matmul(u, v) for u, v in combinations(poly_list, 2)]))  # calculate all unique combinations between polynomials
        theta_dudv = list(chain.from_iterable([string_matmul(du, dv)[1:] for du, dv in combinations(deriv_list, 2)])) # calculate all unique combinations of derivatives
        theta_udu = list(chain.from_iterable([string_matmul(u[1:], du[1:]) for u, du in product(poly_list, deriv_list)])) # calculate all unique combinations of derivatives
        theta = theta_uv + theta_dudv + theta_udu
    return theta


