import tensorflow as tf
import networkx as nx
import json
import numpy as np
import pandas as pd
import os


def load_model(file):
    return tf.keras.models.load_model(file)

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
  
  return G

def load_graphs(folder):
    # go through all files in the folder
    for (root, dirs, file) in os.walk(folder):
        for f in file:
            # get json files
            if '.json' in f:
                with open(os.path.join(folder, f)) as json_file:
                    # load that set of graphs
                    graphset = json.load(json_file)
                # turn into dataframe
                df = pd.DataFrame(graphset)
                for row in df.itertuples(index=False):
                    # each row in dataframe represents a graph, turn into graph object
                    graphs.append(graph_from_row(row))
                    # also record known tour value
                    known_tour_vals.append(getattr(row, 'tourCost'))

#Edges [origin = u, destination = v, weight = c]
# E = [u , v, c]
#List of vertices
# V = []
#Number of vertices
# n = len(V)
#Number of edges
# m = len(E)
#Starting vertex
# s = V[0]
#number of iterations to approximate on
# max = 100

#output in visiting order
solution = []

#Algorithmic variables
q = []  #search stack (represents the search tree)

# cur_ver = s #current vertex for searching

#Best known cost and edges
solution = {"cost": float("inf"), "edges": []}

#Best current cost and edges
new_sol = {"cost": 0, "edges": []}

#Visited vertices in new_sol
new_verts = set()

#upper bound to prune
bound = float("inf")

def solve_tsp():
    #Find initial solution using DFS
    bound = 9999    #initialize bound to high value
    solution = dfs(bound)

    #outer loop - repeat a set number of iterations (time limit)
    for i in range(max):
        new_sol = dfs()

        #error happens when no more edges to explore, premature termination 
        if new_sol is not None:
            if new_sol["cost"] < solution["cost"]:
                bound = new_sol["cost"]
                solution = new_sol
        else: #new_sol is error, no more edges to explore
            return solution
        
    return solution

def dfs(bound):
    #Check if solution needs to return
    # if len(new_sol) == n:
        # return new_sol
    
    #add edges to search stack
    add_edges()

    while True:
        if len(q) == 0:
            return None #error, no more edges to explore
        
        can_edge = q.pop(0)

        if can_edge["prediction"] > bound:
            continue

        if can_edge.u == cur_ver:
            new_sol["edges"].append(can_edge)
            new_sol["cost"] += can_edge["c"]
            cur_ver = can_edge.v
            new_verts.add(cur_ver)
            break
        else:
            q.prepend(can_edge)
            backtrack()
            
        return dfs()

def add_edges():
    valid_exp = []

    #find new valid edges and get their predicted value
    for (u, v, c) in E:
        # if u != cur_ver:
            # continue

        #skip edges that would loop back
        if len(new_sol["edges"]) > 0:
            last_u = new_sol["edges"][-1]["u"]
            if v == last_u:
                continue

        #skip edges that go to explored nodes unless it is start
        if v in new_verts and v != s:
            continue

        new_edge = {"u": u, "v": v, "c": c}

        #prediction for a potential new_edge = predict
        predict = model.predictValue(new_sol, new_edge)

        if predict <= bound:
            valid_exp.append({
                "u": u,
                "v": v,
                "c": c,
                "prediction": predict
            })

    #sort new valid edges by prediction
    valid_exp.sort(key=lambda e: e["prediction"])

    #add the new edges to the queue in sorted order
    q[0:0] = valid_exp

def backtrack():
    if not new_sol["edges"]:
        #nothing to backtrack
        # cur_ver = s
        return

    edge_rem = new_sol["edges"][-1]  #get_last_item
    new_verts.discard(edge_rem["v"])  #remove that vertex from visited
    new_sol["cost"] -= edge_rem["c"]
    new_sol["edges"].pop()           #remove_last_item

    #move current vertex back to the previous vertex
    if new_sol["edges"]:
        cur_ver = new_sol["edges"][-1]["v"]
    # else:
        # cur_ver = s
        
        
if __name__ == "__main__":
    global model
    global graphs
    global known_tour_vals
    global solved_tour_vals
    model_file = './new_model.keras'
    graph_folder = './smallsetgraphs'
    
    print('Attempting to load model')
    model = load_model(model_file)
    print(model.summary())
    print('Loaded model')
    
    print('Attempting to import graphs')
    known_tour_vals = []
    graphs = []
    load_graphs(graph_folder)
    print('Imported graphs')
    # print(len(known_tour_vals))
    # print(known_tour_vals)
    
    # solved_tour_vals = []
    # for g in graphs:
    #     solve_tsp(model)