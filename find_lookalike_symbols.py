'''
generated look-alike candidates with k-nearest neighbours over rendered symbols
'''
from ours.config import HME100K_DICT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

K = 5

def read_symbols():
    return HME100K_DICT.read_text(encoding="utf-8").split("\n")[:-1]

def render(symbol):

    fig = plt.figure(figsize=(1,1), dpi=100)
    fig.text(0.5,0.5,f"${symbol}$", ha="center", va="center", fontsize =40, math_fontfamily="cm")
    try:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:,:,0].copy()
    except ValueError:
        return None
    finally:
        plt.close(fig)


def to_vector(image, size=32):
    ink = image < 255
    if not ink.any():
        return None
    rows = ink.any(axis = 1)
    cols = ink.any(axis = 0)
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    crop = image[r[0]:r[-1]+1, c[0] : c[-1]+1]
    side = max(crop.shape)
    square = np.full((side, side),255, np.uint8)
    top = (side - crop.shape[0])// 2
    left = (side - crop.shape[1]) // 2
    square[top:top + crop.shape[0] , left: left + crop.shape[1]] = crop
    small = Image.fromarray(square).resize((size,size) , Image.BOX)
    return (1-np.asarray(small)/255).flatten()

def nearest(vectors, k):
    searcher = NearestNeighbors(n_neighbors=5).fit(vectors)
    distances , indices = searcher.kneighbors(n_neighbors=k)
    return indices, distances

COLUMNS = ["symbol", "neighbour", "action"]
OUT = Path("lookalike_candidates.tsv") 
SKIP = {"、", "。"}  

def main():
    symbols = read_symbols()
    drawn, vectors, skipped = [] , [] , []
    for i in range(len(symbols)):
        symbol = symbols[i]
        if symbol in SKIP:
            continue
        image = render(symbol)
        vector = None if image is None else to_vector(image)
        if vector is None:
            continue
        drawn.append(symbol)
        vectors.append(vector)

    vectors = np.array(vectors)
    indices, distance = nearest(vectors, K)

    rows = []
    seen = set()

    for i in range(len(indices)):

        for rank in range(K):
            j =int(indices[i][rank])
            if (i,j) in seen or (j,i) in seen:
                continue
            seen.add((i,j))
            same = np.array_equal(vectors[i], vectors[j])
            rows.append([drawn[i], drawn[j], "merge" if same else "unknown"])

    assert not any("\t" in v or "\n" in v for row in rows for v in row)
    lines = ["\t".join(COLUMNS)] + ["\t".join(row) for row in rows]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()