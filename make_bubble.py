# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:52:02 2020

@author: user
"""

import os
import cv2

def outt(outputpath, fpnumber):
    from PIL import Image
    import queue
    
    def check_bound(curpos):
        if (curpos[0]>= n) or (curpos[1]>= m) or (curpos[0]<= 0) or (curpos[1]<= 0):
            return False
        else:
            return True
    
    def search_direc(node, img):
        toreturn = []
        curx = node[0]
        cury = node[1]
        for k in range(1,9):
            if k == 1 and check_bound((node[0], node[1] + 1)):#R
                if img[curx, cury + 1] == 0:
                    toreturn.append(k)
            elif k == 2 and check_bound((node[0] + 1, node[1] + 1)):#RD
                if img[curx + 1, cury + 1] == 0:
                    toreturn.append(k)
            elif k == 3 and check_bound((node[0] + 1, node[1])):#D
                if img[curx + 1, cury] == 0:
                    toreturn.append(k)
            elif k == 4 and check_bound((node[0] + 1, node[1] - 1)):#LD
                if img[curx + 1, cury - 1] == 0:
                    toreturn.append(k)
            elif k == 5 and check_bound((node[0], node[1] - 1)):#L
                if img[curx, cury - 1] == 0:
                    toreturn.append(k)
            elif k == 6 and check_bound((node[0] - 1, node[1] - 1)):#LU
                if img[curx - 1, cury - 1] == 0:
                    toreturn.append(k)
            elif k == 7 and check_bound((node[0] - 1, node[1])):#U
                if img[curx - 1, cury] == 0:
                    toreturn.append(k)
            elif k == 8 and check_bound((node[0] - 1, node[1] + 1)):#RU
                if img[curx - 1, cury + 1] == 0:
                    toreturn.append(k)
        return toreturn
    def find_inverse(curdir):
        if curdir == 1:
            return 5
        elif curdir == 2:
            return 6
        elif curdir == 3:
            return 7
        elif curdir == 4:
            return 8
        elif curdir == 5:
            return 1
        elif curdir == 6:
            return 2
        elif curdir == 7:
            return 3
        elif curdir == 8:
            return 4
    def move_to_curdir(node, curdir, test):
        curpos = node
        if curdir == direction["R"]:
            if test[node[0], node[1] + 1] == 0:
                curpos = (node[0], node[1] + 1)
        elif curdir == direction["RD"]:
            if test[node[0] + 1, node[1] + 1] == 0:
                curpos = (node[0] + 1, node[1] + 1)
        elif curdir == direction["D"]:
            if test[node[0] + 1, node[1]] == 0:
                curpos = (node[0] + 1, node[1])
        elif curdir == direction["LD"]:
            if test[node[0] + 1, node[1] - 1] == 0:
                curpos = (node[0] + 1, node[1] - 1)
        elif curdir == direction["L"]:
            if test[node[0], node[1] - 1] == 0:
                curpos = (node[0], node[1] - 1)
        elif curdir == direction["LU"]:
            if test[node[0] - 1, node[1] - 1] == 0:
                curpos = (node[0] - 1, node[1] - 1)
        elif curdir == direction["U"]:
            if test[node[0] - 1, node[1]] == 0:
                curpos = (node[0] - 1, node[1])
        elif curdir == direction["RU"]:
            if test[node[0] - 1, node[1] + 1] == 0:
                curpos = (node[0] - 1, node[1] + 1)
        return curpos
    
    
    def main(sp, idxcnt):
        start = None#시작점 찾기
        try:
            sx, sy = sp[0], sp[1]
        except(TypeError):
            return (0,0)
        for i in range(sx, n):
            for j in range(sy, m):
                if cur_img[i,j] == 0:
                    start = (i, j)
                    break;
            if start != None:
                break;
        if start == None:
            return False
        nodes[idxcnt] = start
        nodes_[start] = idxcnt
        cur_img[nodes[idxcnt][0]][nodes[idxcnt][1]] = 255
        idxcnt += 1    
        
        towhere = search_direc(start, cur_img)
        qu = queue.Queue()
        
        for i in towhere: #갈림길 모두 큐에 넣기
            qu.put((nodes_[start], i))
            
        while(qu.qsize() != 0): #큐가 다 빌 때까지
            cur = qu.get()#현재 대상 노드와 방향
            curpoint = curcoord = nodes[cur[0]]#대상 노드 좌표
            curdir = cur[1]#대상 방향
            curdir_ = find_inverse(curdir)
            nextpoint = move_to_curdir(curcoord, curdir, cur_img)#대상 방향으로 이동, 이동 못하면 False
            while(nextpoint != False and curpoint != nextpoint):#이동 성공했으면
                nodes[idxcnt] = nextpoint # 다음 점을 노드로 추가하고
                nodes_[nextpoint] = idxcnt
                cur_img[nodes[idxcnt][0]][nodes[idxcnt][1]] = 255 #이미지에 마킹
                idxcnt += 1
                edges.append((nodes_[curpoint], nodes_[nextpoint]))#현재 위치와 다음 위치를 엣지로 추가
                curpoint = nextpoint  #현재 위치 변경
                towhere = search_direc(curpoint, cur_img)#갈림길이 있는지 보자
                for i in towhere:
                    if i != curdir_:
                        qu.put((nodes_[curpoint], i))
        return start,idxcnt
    
    ##########################################parameters setup####################################################
        
    os.chdir(outputpath)
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]    
        
    test_img = cv2.imread('./' + fpnumberr + '.png/' + fpnumberr + '_thin.bmp', 0) #read the image in grayscale
    direction = {"R":1, "RD":2, "D":3, "LD":4, "L":5, "LU":6, "U":7, "RU":8}
    im = Image.fromarray(~test_img)
    l, u, r, d = im.getbbox()# 영상의 바운딩 박스를 구해 알고리즘의 효울성 증진
    m,n = test_img.shape
    cur_img = test_img.copy()
    nodes = dict()
    nodes_ = dict()
    edges = []
    idxcnt = 0
    sp = (u,l)
    cnt = 0
    
    #############################################execute main############################################
    while(True):
        cnt += 1
        sp, idxcnt = main(sp, idxcnt)
        pixels = sum(sum(~cur_img))
        print("iter {} remain pixels {}".format(cnt, round(pixels)))
        if pixels <= 0:
            break
        im = Image.fromarray(~cur_img)
        l, u, r, d = im.getbbox()
        sp = (u,l)
        
    
    cenx = int(m/2)
    ceny = int(n/2)
    
    ################################rotating(270')#############################################################
    
    for i in range(3):
        for k, v in nodes.items():
            newx = int((v[0]-cenx)*0 - (v[1]-ceny)*1 + cenx)
            newy = int((v[0]-cenx)*1 + (v[1]-ceny)*0 + ceny)
            nodes[k] = (newx, newy)
    nodes_ = dict()
    for k, v in nodes.items():
        nodes_[v] = k
    import networkx as nx
    from matplotlib import pyplot as plt
    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(nodes))])    
    G.add_edges_from(edges)
#    posi = dict()
#    for i in range(0, len(nodes)):
#        posi[i] = nodes[i]
#    nx.draw_networkx(G, posi, with_labels = False, node_size = 3)
#    plt.plot()
    
    return nodes, nodes_, edges, G
nodes, nodes_, edges, G_ = outt('C:/Users/user/Desktop/module3_vec/module3/test_output', '1-1.png')
###################################GeoJSON export################################################
#
#from geojson import Feature, FeatureCollection, dump, LineString
#
#
#features = []
#for i in edges:
#    line = []
#    line.append(nodes[i[0]])
#    line.append(nodes[i[1]])
#    
#    poly = LineString(line)
#    features.append(Feature(geometry=poly))
#
#feature_collection = FeatureCollection(features)
#
#with open('test_img' + '__.geojson', 'w') as f:
#   dump(feature_collection, f)


