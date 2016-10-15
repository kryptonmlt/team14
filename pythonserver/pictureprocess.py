import numpy as np

import scipy
from scipy import misc

import cv2


def create_world():
    ## load the image
    img0 = misc.imread('./example2.jpg')

    ## make sure it is landscape
    x,y,z = img0.shape
    if (x>y):
        img1 = np.rollaxis(img0, 1, 0)
    else:
        pass

    ## resize to (768, 1024)
    img1 = misc.imresize(img1, (768, 1024))

    ## load the player
    player = np.zeros((64,64,3))
    cv2.circle(player, (32,32), 30, (1,1,1), 3)
    cv2.line(player, (0,0), (64,64), (1,1,1), 3)
    cv2.line(player, (0,64), (64,0), (1,1,1), 3)

    ## match
    player_255_uint8 = player.astype(np.uint8)*255

    FLANN_INDEX_KDTREE = 0
    MIN_MATCH_COUNT = 5

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(player_255_uint8,None)
    kp2, des2 = sift.detectAndCompute(img1,None)

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    bfMatcher = cv2.BFMatcher()
    matches = bfMatcher.knnMatch(des1, des2, k=2)


    ## get the most matches
    sorted_matches = sorted([(m,n) for m,n in matches], key=lambda x: np.abs(x[0].distance - x[1].distance))

    flag_xy = []
    for match in [kp2[ni.trainIdx].pt for (mi, ni) in sorted_matches[:10]]:
        flag_xy.append(match)

    from collections import Counter
    k = Counter(flag_xy).keys() # counts the elements' frequency
    v = Counter(flag_xy).values() # counts the elements' frequency

    topposition = sorted(zip(v,k), reverse=True)[:2]

    # make sure this is greater than 2

    player1, player2 = sorted(zip(v,k), reverse=True)[:2]


    ## make



    return player1, player2