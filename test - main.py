import numpy as np
from utils import *

def main(ListFit, n):          #On lui fournit une matrice n*n*4 avec le fitting de chaque sous-imageimage par rapport à chaque autre, dans chaque direction

    # ListFit = OrderFit(FitMatrix)   #OrderFit renvoie une liste de 4-uplet (image1, image2, direction(N/E/S/W), fit) des couples d'images avec le meilleur fitting

    L = [np.array([(i,j)]) for i in range(n) for j in range(n)]       # la liste des clusters de blocs rassemblés, chaque cluster étant représenté par une liste des coordonnées de chaque bloque qui le forme. On l'initialise naturellement comme étant une liste de n**2 clusters de taille 1 chacun

    D_cluster = {(i,j) : i*n + j for i in range(n) for j in range(n)}               #On numérote les clusters de manière arbitraire, le tout étant de savoir dans quel cluster se trouve chaque morceau d'image


    L_blank = []   #List of the image pieces that should be colored in white because of image overlap
    for match in ListFit:

        m1, m2 = D_cluster[match[0]], D_cluster[match[1]]
        (i1, j1), (i2, j2) = LocateInCluster(match[0], L, m1), LocateInCluster(match[1], L, m2)

        L[m2] = MergeClusters(L[m1], L[m2], (i1, j1), (i2, j2), match[2])
        RearangeD_cluster(D_cluster, L[m1], m2)    # the puzzle pieces moved from one cluster to the other induce a refreshing of D_cluster
        del L[m1]


        if len(L) == 1:
            break

    return L[0]




def LocateInCluster(k, L, m):      # returns position of k in the m-th element of the cluster-list L
    res = np.where(k == L[m])

    if len(res[0]) != 1:
        print('WARNING : none or multiple instances of ', k, 'found in cluster')

    return res[0][0], res[1][0]


def  MergeClusters(C1, C2, pos1, pos2, dir):  # the clusters, the position (in the clusters) of the matching pieces, and the direction of the match, this adds C1 on C2 and expands it if necessary
    n1, m1 = np.shape(C1)
    n2, m2 = np.shape(C2)

    D_dir = dict(N=[1, 0], E=[0, 1], S=[-1, 0], W=[0, -1])

    locmatch = pos2 + D_dir[dir]        # position of the matched piece from C1 in C2 once it will be added
    space_expanse = [[n1 - pos1[0] - 1, m1 - pos1[1] - 1], [-pos1[0], -pos1[1]]]   # fitting C1 in C2 is equivalent to checking that two opposites corner of C1 fit in C2, we therefore compute the positions of these opposite corners in C2 and extend C2 if the positions of the corners are out of C2's bounds

    cor1, cor2 = locmatch + space_expanse[0], locmatch + space_expanse[1]
    cor1 -= (n2, m2)
    exp1, exp2 = sum(np.greater(cor1, (0,0))), sum(~np.greater(cor2, (0, 0)))

    if exp1:
        cor1 = [max(x, 0) for x in cor1]
        A = np.full((n2, m2) + cor1, 'x')       #A is not filled with zeros since 0 refers to a particular pieceof the puzzle
        A[:n2, :m2] = C2
        C2 = A
        n2, m2 = np.shape(C2)

    if exp2:
        cor2 = [-min(x, 0) for x in cor2]
        A = np.full((n2, m2) + cor2, 'x')
        A[cor2[0]:, cor2[1]:] = C2
        C2 = A

    for i in range(n1):
        for j in range(m1):             #adding C1 to C2 with the offset due to the location of the map, there is probably a better way to do this using slicing

            if C2[i + cor2[0], j + cor2[1]] == 'x':
                print('WARNING - puzzle pieces overlappings')
            C2[i + cor2[0], j + cor2[1]] = C1[i, j]

    return C2

def RearangeD_cluster(D_cluster, C1, m2):
    n, m = np.shape(C1)
    for i in range(n):
        for j in range(m):
            if C1[i, j] != 'x':
                D_cluster[(i, j)] = m2








