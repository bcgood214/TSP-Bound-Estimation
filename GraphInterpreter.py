import networkx as nx
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# select file to open
with open('./trainingGraphs/5-5000.json', 'r') as f:
  data = json.load(f)

# load the data from json into pd datafram
df = pd.DataFrame(data)

# function to convert a row from the datafram into a nx graph and get tensors from it
def graph_from_row(row):
  G = nx.Graph()
  v = getattr(row, 'V')
  sorted_nodes = sorted(v)
  for i, node_id in enumerate(sorted_nodes):
    original_index = v.index(node_id)
    coords = getattr(row,'Vcoords')[original_index]
    G.add_node(node_id, pos=coords)
    
  for edge_str, weight in getattr(row,'Eweights').items():
    u_str, v_str = edge_str.split(',')
    u, v = int(u_str), int(v_str)
    G.add_edge(u, v, weight=weight)
    
  adj_matrix = nx.to_numpy_array(G, nodelist=sorted_nodes, weight='weight')
  adj_tensor = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
  
  coords_list = [G.nodes[n]['pos'] for n in sorted_nodes]
  coords_tensor = tf.convert_to_tensor(coords_list, dtype=tf.float32)
  
  tour_1based = getattr(row, 'tourOrder')
  tour_0based = [n-1 for n in tour_1based]
  tour_tensor = tf.convert_to_tensor(tour_0based, dtype=tf.int32)
  
  return G, adj_tensor, coords_tensor, tour_tensor

# process all the data
graphs = []
adjTensors = []
coordTensors = []
tourTensors = []
for row in df.itertuples(index=False):
  graph, adj, coord, tour = graph_from_row(row)
  graphs.append(graph)
  adjTensors.append(adj)
  coordTensors.append(coord)
  tourTensors.append(tour)

# get information from the graph
print(nx.display(graphs[0]))
print(nx.is_weighted(graphs[0]))
print(graphs[0].edges())