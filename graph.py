# nodes, nodes_, edges, G_ = make_bubble(outputpath, fpnumber)
# wall_corner, wall_corner_ = wall_corner_nearest(outputpath, fpnumber)
# window_coord= window_nodes_extraction(filepath, separated_path, fpnumber, outputpath)
# window_coord_id = find_window_coord_from_nodes(nodes_, window_coord)
# wall_coord, wall_coord_  = wall_nodes_extraction(nodes, window_coord_id)
# wall_coord_id = find_wall_coord_from_nodes(nodes_, wall_coord)
# wall_corner_id = change2wall_corner_id(wall_corner)

# attrs = dict() #key:value = node_id:property
# for i in nodes.keys(): 
#     if i in window_coord_id:
#         attrs[i]=1 #window
#     else:
#         attrs[i]=0 #wall

# w_attrs = dict()
# for i in nodes.keys():
#     if i in wall_corner_id:
#         w_attrs[i]=1 #corner
#     else: 
#         w_attrs[i]=0 #wall 


# nx.set_node_attributes(G_, attrs, 'object')

# diver_point = [] #창문>>벽 or 벽>>창문 으로 바뀌는 지점
# for i in window_coord_id:
#     neis = [k for k in nx.neighbors(G_, i)]
#     neis_attr_set = set([G_.nodes[j]['object'] for j in neis])
#     if len(neis_attr_set) > 1 : 
#         diver_point.append(i)


# = nx.set_node_attributes(G_, w_attrs, 'object')
#
#w_corner_point = [] #벽>>벽코너 or 벽코너>>벽 으로 바뀌는 지점
#for i in wall_coord_id:
#    w_neis = [k for k in nx.neighbors(G_, i)]
#    w_neis_attr_set = set([G_.nodes[j]['object'] for j in neis])
#    if len(w_neis_attr_set) > 1 : 
#        w_corner_point.append(i)
        

# G_win = nx.Graph()  
# G_win.add_nodes_from(window_coord_id)
# G_wall = nx.Graph()

# ''' 
# wall coord & wall corner coord
# '''
# wall_coord_arr = []
# for key, value in wall_coord.items():
#     x, y = value
#     wall_coord_arr.append([x, y])

# wall_corner_arr = []
# for key, value in wall_corner.items():
#     x, y = value 
#     wall_corner_arr.append([x, y])

# un_corner = [] 

# wall_corner_new = dict() 
# wall_corner_deleted = dict()
# for i in range(len(wall_corner_arr)):
#     if wall_corner_arr[i] not in wall_coord_arr: 
#         x, y = wall_corner_arr[i]
#         d = wall_corner_[x, y]
# #        wall_corner_deleted.update(d)
#     elif wall_corner_arr[i] in wall_coord_arr: 
#         x, y = wall_corner_arr[i]
#         temp = wall_corner_[x, y]
#         temp2 = wall_coord_[x, y]
#         if temp != temp2:
#             new_temp = temp2 
#         wall_corner_new[temp2] = (x,y)

    
# wall_corner_new_temp = wall_corner_new.copy()
# wall_corner_new_id = [] 
    
# for key in wall_corner_new.keys():
#     wall_corner_new_id.append(key) #wall dict에서 key (node id)만 추출하여 리스트 형태로 저장; window_coord_id 와 동일
        

    
# G_wall.add_nodes_from(wall_corner_new_id)

# win_edges = []
# for i in edges:
#     if (i[0] in window_coord_id) and (i[1] in window_coord_id):
#         win_edges.append(i)
        
# wall_edges = []
# for i in edges:
#     if (i[0] in wall_coord_id) and (i[1] in wall_coord_id):
#         wall_edges.append(i)


# G_win.add_edges_from(win_edges)
# G_wall.add_edges_from(wall_edges)


# #final window edges
# wins = []
# for i in range(len(diver_point)):
#     for j in range(i, len(diver_point)):
#         if len([k for k in nx.all_simple_paths(G_win, diver_point[i], diver_point[j])]) > 0:
#             wins.append((diver_point[i], diver_point[j]))


# walls = [] 
# start= time.process_time()
# for i in range(len(wall_corner_new_id)):
#     for j in range(i, len(wall_corner_new_id)):
#         if len([k for k in nx.all_simple_paths(G_wall, wall_corner_new_id[i], wall_corner_new_id[j])]) > 0:
#             walls.append((wall_corner_new_id[i], wall_corner_new_id[j]))
#             print(str(time.process_time() - start) + " passed" )
#             print(str(i) + " out of " + str(len(wall_corner_new_id)) + " done")
        

# window graph            
# G_win = nx.Graph()
# G_win.add_nodes_from(diver_point)
# G_win.add_edges_from(wins)

# #wall graph
# G_wall = nx.Graph()
# G_wall.add_nodes_from(wall_corner_new_id)
# G_wall.add_edges_from(walls)


# m = n = 3309
# cenx = int(m/2)
# ceny = int(n/2)
    

# nodes_pos = nodes.copy()
# for i in range(3):
#     for k, v in nodes_pos.items():
#         newx = int((v[0]-cenx)*0 - (v[1]-ceny)*1 + cenx)
#         newy = int((v[0]-cenx)*1 + (v[1]-ceny)*0 + ceny)
#         nodes_pos[k] = (newx, newy)
# nx.draw(G_win, pos = nodes_pos, node_size = 3)
        
        
# # read whole vector file as linevec and convert wins to win_vector

        
# data = gpd.read_file(outputpath + fpnumberr + "/" + fpnumberr + "_wall.geojson")
# win_vec = []
# for i in wins:
#     win_vec.append(point(nodes[i[0]][1], nodes[i[0]][0]))
#     win_vec.append(point(nodes[i[1]][1], nodes[i[1]][0]))

# win_points = gpd.GeoDataFrame(win_vec, columns = ['geometry'])
# clip = gpd.clip(data, win_points)