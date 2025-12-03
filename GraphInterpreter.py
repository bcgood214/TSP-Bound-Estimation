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
  # subtract 1 from vertex labels to account for the 1-index
  v = [i-1 for i in v]
  sorted_nodes = sorted(v)
  coord_list = getattr(row, 'Vcoords')
  for i, node_id in enumerate(sorted_nodes):
    original_index = v.index(node_id)
    coords = coord_list[original_index]
    G.add_node(node_id, pos=coords)
    
  for edge_str, weight in getattr(row,'Eweights').items():
    u_str, v_str = edge_str.split(',')
    # again, subtract 1 from edge labels to account for the 1-index
    u, v = int(u_str) - 1, int(v_str) - 1
    G.add_edge(u, v, weight=weight)
    
  # adjacency matrix
  adj_matrix = nx.to_numpy_array(G, nodelist=sorted_nodes, weight='weight')
  adj_tensor = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
  
  # coordinate tensor
  coords_list = [G.nodes[n]['pos'] for n in sorted_nodes]
  coords_tensor = tf.convert_to_tensor(coords_list, dtype=tf.float32)
  
  # the original indices were 1 indexed (thank you Mathematica)
  # so this adjusts them to align with the 0 indexing of python that is used for
  # the other tensors and issues
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
  
testi = 2
# get information about a graph's raw data before conversion and tensor
print(df.iloc[testi])

# get information from the graph
print(graphs[testi].nodes())
print(nx.display(graphs[testi]))
print(nx.is_weighted(graphs[testi]))
print(graphs[testi].edges())
# print(nx.get_edge_attributes(graphs[testi],'weight'))
# print(adjTensors[testi])
# print(coordTensors[testi])
# print(tourTensors[testi])

# Example of way to interact with the graph
tsp = nx.approximation.traveling_salesman_problem
print("nx solver: ")
tour = tsp(graphs[testi], weight='weight')
print(tour)
print(nx.path_weight(graphs[testi],tour,weight='weight'))

# showing more ways to get information from the graph, such as an iterator
# This gives an iterator to the neighbors of vertex 0
print(nx.neighbors(graphs[testi], 0))