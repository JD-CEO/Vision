import numpy as np
from sys import maxsize


def minCut(img_overlap, out_overlap, loc):

    diff = img_overlap - out_overlap

    E = np.sum(np.multiply(diff, diff), axis=2)

    if(loc == "horizontal"):
        E = np.transpose(E)

    R, C = E.shape[:2]
    cut = np.ones((R, C))
    DP = np.zeros((R, C))

    for i in range(C):
        DP[0, i] = E[0, i]

    for i in range(1, R):
        for j in range(C):
            paths = []
            paths.append(DP[i-1, j])
            if j != 0:
                paths.append(DP[i-1, j])
            if j != C-1:
                paths.append(DP[i-1, j+1])
            DP[i, j] = E[i, j] + min(paths)

    min_val = min_idx = maxsize
    for i in range(C):
        if min(min_idx, E[R-1, i]) < min_val:
            min_idx = i

    """
    E[i,j] = e[i,j] + min(E[i-1,j-1], E[i-1,j], E[i,j-1])
    """

    cut[R-1, min_idx] = 0
    cut[R-1, min_idx+1:C] = 1
    cut[R-1, 0:min_idx] = -1

    for i in range(R-2, -1, -1):
        for j in range(C):
            if min_idx < C-1:
                if E[i, min_idx+1] == min(E[i, max(0, min_idx-1):min_idx+2]):
                    min_idx = min_idx + 1
            if min_idx > 0:
                if E[i, min_idx-1] == min(E[i, min_idx-1:min(C-1, min_idx+2)]):
                    min_idx = min_idx - 1
            cut[i, min_idx] = 0
            cut[i, min_idx+1:C] = 1
            cut[i, 0:min_idx] = -1

    if loc=="horizontal":
        cut = np.transpose(cut)

    return cut