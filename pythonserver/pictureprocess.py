import numpy as np

import scipy
from scipy import misc

import cv2


def create_world(path):
    ## load the image
    img0 = misc.imread(path)

    ## make sure it is landscape
    x, y, z = img0.shape
    if (x > y):
        img1 = np.rollaxis(img0, 1, 0)
    else:
        img1 = img0

    ## resize to (768, 1024)
    img1 = misc.imresize(img1, (768, 1024))

    ## load the player
    player = np.zeros((64,64,3), dtype=np.uint8)
    cv2.circle(player, (32,32), 30, (1,1,1), 3)
    cv2.line(player, (0,0), (64,64), (1,1,1), 3)
    cv2.line(player, (0,64), (64,0), (1,1,1), 3)
    player *= 255

    ## match the player locations
    player_255_uint8 = player

    FLANN_INDEX_KDTREE = 0
    MIN_MATCH_COUNT = 5

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(player_255_uint8, None)
    kp2, des2 = sift.detectAndCompute(img1, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    bfMatcher = cv2.BFMatcher()
    matches = bfMatcher.knnMatch(des1, des2, k=2)

    ## get the most matches
    sorted_matches = sorted([(m, n) for m, n in matches], key=lambda x: np.abs(x[0].distance - x[1].distance))

    flag_xy = []
    for match in [kp2[ni.trainIdx].pt for (mi, ni) in sorted_matches[:10]]:
        flag_xy.append(match)

    from collections import Counter
    k = Counter(flag_xy).keys()  # counts the elements' frequency
    v = Counter(flag_xy).values()  # counts the elements' frequency


    # make sure this is greater than 2 or pop missing location arbitrarily
    # topposition = sorted(zip(v, k), reverse=True)[:2]
    # player1, player2 = sorted(zip(v, k), reverse=True)[:2]
    # import scipy
    from scipy import cluster
    players_pos = cluster.vq.kmeans(np.array(k, dtype=np.float), 2)

    x, y = players_pos[0][0]
    player1 = x/16 + 64*y/16
    x, y = players_pos[0][1]
    player2 = x/16 + 64*y/16

    ## pixelate by 16x16
    pxlimg = np.zeros((4*48,4*64,3), dtype=np.uint8)
    img = img1
    pR = 4

    for i in range(pxlimg.shape[0]):
        for j in range(pxlimg.shape[1]):
            pxlimg[i,j] = np.mean(img[i*pR:(i+1)*pR, j*pR:(j+1)*pR].reshape(-1,3), axis=0, dtype=int)

    pxlimg_grs = cv2.cvtColor(pxlimg, cv2.COLOR_RGB2GRAY)
    ret, frame = cv2.threshold(pxlimg_grs, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    background_idx = np.argwhere(frame.ravel())

    # ## get the background
    # median_pxlimg = np.median(pxlimg.reshape(-1,3), axis=0)
    # background_idx = np.argwhere((np.linalg.norm((pxlimg.reshape(-1,3) - median_pxlimg), axis=1)) < 50)

    ## tile id for non background tiles
    wall_tileid = set(range(48*64)) - set(np.unique(background_idx))

    return player1, player2, background_idx, pxlimg, img1
