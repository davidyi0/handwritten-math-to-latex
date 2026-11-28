'''
generated look-alike candidates with k-nearest neighbours over rendered symbols, then screened them with an LLM and checked each merge by hand.
'''
from ours.config import HME100K_DICT
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

K = 5


def read_symbols():
    """The 188 symbols in hme100k_dictionary.txt, in file order."""
    text = Path("ours/hme1")

def render(symbol):
    raise NotImplementedError
    pass

def to_vecotr(image, size=32):
    raise NotImplementedError
    pass

def nearest(vectors, k):
    raise NotImplementedError
    pass

def main():
    raise NotImplementedError
    pass

if __name__ == "__main__":
    main()