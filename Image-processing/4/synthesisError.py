from math import floor
from matplotlib.pyplot import axis
import numpy as np

def synthesis_error(img, out_slice, block_size, overlap_size, loc):
    H, W = img.shape[:2]
    E = np.zeros((H - block_size, W - block_size), dtype=float)

    for i in range(H-block_size):
        for j in range(W-block_size):

            row, col = i, j
            
            if(loc == "left"):
                e_row = row + block_size
                e_col = col + overlap_size
            elif (loc == "up"):
                e_row = row + overlap_size
                e_col = col + block_size
            elif (loc == "corner"):
                e_row = row + overlap_size
                e_col = col + overlap_size
            img_block = img[row:e_row, col:e_col, :]

            """
            Error is given by 
            error = (B1_ov - B2_ov)^2
            E[i,j] = e[i,j] + min(E[i-1,j-1], E[i-1,j], E[i,j-1])
            """
            diff = out_slice - img_block
            err = np.reshape(diff, (diff.shape[0]*diff.shape[1]*diff.shape[2], 1))
            err = np.multiply(err, err)
            ssum = np.sum(err, axis=0)
            E[i, j] += ssum

    return E