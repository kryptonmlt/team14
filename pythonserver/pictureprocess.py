import numpy as np

import scipy
from scipy import misc

import cv2


def create_world(path):
    ## load the image from path saved in POST request
    img0 = misc.imread(path)

    ## make sure it is landscape
    x, y, z = img0.shape
    if (x > y):
        img1 = np.rollaxis(img0, 1, 0)
    else:
        img1 = img0

    ## resize to (768, 1024)
    img1 = misc.imresize(img1, (768, 1024))

    ## create the player marker
    player = np.zeros((64,64,3), dtype=np.uint8)
    cv2.circle(player, (32,32), 20, (1,1,1), 3)
    cv2.line(player, (10,10), (54,54), (1,1,1), 3)
    cv2.line(player, (10,54), (54,10), (1,1,1), 3)
    player *= 255

    ## match the player locations
    player_255_uint8 = cv2.cvtColor(player, cv2.COLOR_RGB2GRAY)
    img_grs = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    FLANN_INDEX_KDTREE = 0
    MIN_MATCH_COUNT = 5
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(~player_255_uint8, None)
    kp2, des2 = sift.detectAndCompute(img_grs, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    bfMatcher = cv2.BFMatcher()
    matches = bfMatcher.knnMatch(des1, des2, k=2)

    sorted_matches = sorted([(m, n) for m, n in matches], key=lambda x: np.abs(x[0].distance - x[1].distance))
    flag_xy = []
    for match in [kp2[ni.trainIdx].pt for (mi, ni) in sorted_matches]:
        flag_xy.append(match)

    from collections import Counter
    k = Counter(flag_xy).keys()  # counts the elements' frequency
    v = Counter(flag_xy).values()  # counts the elements' frequency

    from scipy import cluster
    players_pos = cluster.vq.kmeans(np.array(k, dtype=np.float), 2)

    x, y = players_pos[0][0]
    print 'p0 ', x,y
    player1 = (int(x),int(y))
    x, y = players_pos[0][1]
    print 'p1 ', x,y
    player2 = (int(x),int(y))

    ## pixelate by 4x4
    pxlimg = np.zeros((4*48,4*64,3), dtype=np.uint8)
    img = img1
    pR = 4
    for i in range(pxlimg.shape[0]):
        for j in range(pxlimg.shape[1]):
            pxlimg[i,j] = np.mean(img[i*pR:(i+1)*pR, j*pR:(j+1)*pR].reshape(-1,3), axis=0, dtype=int)

    ## get the background from thresholding
    pxlimg_grs = cv2.cvtColor(pxlimg, cv2.COLOR_RGB2GRAY)
    ret, frame = cv2.threshold(pxlimg_grs, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    background_idx = np.argwhere(~frame.ravel())

    return player1, player2, background_idx, pxlimg, img1
