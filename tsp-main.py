import tensorflow as tf
import networkx as nx
import json
import pandas as pd
import os
import numpy as np
import argparse

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

def solve_tsp(g, max_iterations):
    #Find initial solution using DFS
    bound = float("inf")    #initialize bound to high value
    solution = dfs(g, bound)
    bound = solution["cost"]

    #outer loop - repeat a set number of iterations (time limit)
    for i in range(max_iterations):
        new_sol = dfs(g, bound)

        #error happens when no more edges to explore, premature termination 
        if new_sol is not None:
            if new_sol["cost"] < solution["cost"]:
                bound = new_sol["cost"]
                solution = new_sol
        else: #new_sol is error, no more edges to explore
            return solution
        
    return solution

def dfs(g, bound):
    new_sol = {"cur_ver": 0, "n": len(g.nodes()), 
               "cost": 0, "edges": [], 
               "vertices": set(), "search_stack": []}  #vertices does not include start vertex 0 until last edge
    
    # loop to drive the search
    while True:
        #Check if solution needs to return
        if len(new_sol["edges"]) == new_sol['n'] and 0 in new_sol["vertices"]:
            return new_sol
    
        if debug: print("cur_ver:", new_sol['cur_ver'])
        #add edges to search stack based on a new current vertex
        add_edges(g, new_sol, bound)

        # loop to handle searching newly discovered edges and backtracking
        while True:
            if len(new_sol['search_stack']) == 0:
                return None #error, no more edges to explore

            # get the next candidate edge from the search stack
            can_edge = new_sol["search_stack"].pop(0)

            # check if edge even still in bounds (should be)
            if can_edge["prediction"] > bound:
                continue

            # check if this edge matches the depth-first approach and still valid
            if can_edge['u'] == new_sol['cur_ver']:
                # valid new depth search, update information
                new_sol["edges"].append(can_edge)
                new_sol["cost"] += can_edge["c"]
                new_sol["cur_ver"] = can_edge['v']
                new_sol["vertices"].add(new_sol["cur_ver"])
                if debug: print("vertices: ", new_sol['vertices'])
                # a new current vertex order was discovered, exit the searching edges to check if solution complete or add new edges
                break
            else:   
                #edge must be from vertex higher in search tree, save edge and backtrack to other vertex that has already 
                #had its edges added above, updating current vertex
                new_sol['search_stack'].insert(0, can_edge)
                backtrack(new_sol)

def add_edges(g, new_sol, bound):
    valid_exp = []
    u = new_sol['cur_ver']
    if debug: print("u: ", u)
    unsolved_nodes = g.nodes() - new_sol['vertices']
    # unsolved_nodes.add(u)
    if debug: print("unsolved nodes: ", unsolved_nodes)
    unsolved_graph = g.subgraph(unsolved_nodes)
    if debug: print(unsolved_graph.edges())
    for v in nx.neighbors(g, u):
        if debug: print("v: ", v)
        if v == 0:  #skip over going back to 0 unless it is the last edge taken
            if len(new_sol['edges']) == new_sol['n']-1:
                # this straight line distance from 0 to 0 is probably 0... 
                # but to keep consistency with the inadmissability of the model, we make a prediction anyways
                straight_dist = np.linalg.norm(np.array(g.nodes[v]['pos']) - np.array(g.nodes[0]['pos']))
                # add last possible return edge to the start to complete the cycle
                prediction = model.predict(np.array([[len(unsolved_nodes), straight_dist]]), verbose = model_verbose)[0,0]
                valid_exp.append({'u': u, 'v': v, 'c': g[u][v]['weight'], "prediction": prediction})
                break   # only add the zero edge
        elif not v in new_sol["vertices"]:
            if debug: print("checking path for v: ", v)
            if nx.has_path(unsolved_graph, v, 0):   #check if a path exists from what would be the new vertex to the start
                # if path exists, get the straight line distance to the start
                straight_dist = np.linalg.norm(np.array(g.nodes[v]['pos']) - np.array(g.nodes[0]['pos']))
                # make prediction based on 
                prediction = model.predict(np.array([[len(unsolved_nodes), straight_dist]]), verbose = model_verbose)[0,0]
                if not prediction > bound:
                    c = g[u][v]['weight']
                    valid_exp.append({'u': u, 'v': v, 'c': g[u][v]['weight'], "prediction": prediction})
    
    # sort any potential edges added by their prediction value
    valid_exp.sort(key=lambda e: e["prediction"])
    
    # prepend these new edges to the search stack to enforce depth first searching
    new_sol["search_stack"] = valid_exp + new_sol['search_stack']

def backtrack(new_sol):
    if debug: print("backtracking; edges: ", new_sol['edges'])
    if not new_sol["edges"]:
        #nothing to backtrack, shouldn't ever really run
        return

    # get the last added edge to backtrack on
    edge_rem = new_sol["edges"][-1]
    new_sol['vertices'].remove(edge_rem["v"])  #remove the destination vertex from visited
    new_sol["cost"] -= edge_rem["c"] # remove the cost
    new_sol['cur_ver'] = edge_rem['u'] # set the current vertex to the source
    new_sol["edges"].pop(-1)  # pop from the edges list and discard

if __name__ == "__main__":
    global model
    global graphs
    global known_tour_vals
    global solved_tour_vals
    global debug
    global model_verbose
    
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('solution_file')
    parser.add_argument('model')
    parser.add_argument('-v', '--model_verbose', type=int, default=0)
    parser.add_argument('-d', '--debug', type=bool, default=False)
    parser.add_argument('-s', '--solver_verbose', type=bool, default=False)
    parser.add_argument('-i', '--iterations', type=int, default=10)
    args = parser.parse_args()
    
    model_file = args.model #'./new_model.keras'
    graph_folder = args.folder #'./validationGraphs/5'
    solution_file = args.solution_file #'./validationGraphs/5/solution_comp.csv'
    max_iterations = args.iterations
    debug = args.debug
    verbose_solving = args.solver_verbose
    model_verbose = args.model_verbose
    
    print('Attempting to load model')
    model = load_model(model_file)
    if debug: print(model.summary())
    print('Loaded model')
    
    print('Attempting to import graphs')
    known_tour_vals = []
    graphs = []
    load_graphs(graph_folder)
    print('Imported ', len(graphs), ' graphs')
    if debug: print(len(known_tour_vals))
    if debug: print(known_tour_vals)
    
    # soltest1 = solve_tsp(graphs[0], max_iterations)
    # print(soltest1)
    # print(known_tour_vals[0])
    solved_tour_vals = [0] * len(known_tour_vals)
    for i in range(len(graphs)):
        print("Graph: ", i)
        g = graphs[i]
        if verbose_solving: print('n = ', g.number_of_nodes())
        g_sol = solve_tsp(g, max_iterations)
        if verbose_solving: print('solve: ', g_sol['cost'], ' exact: ', known_tour_vals[i])
        solved_tour_vals[i] = g_sol['cost']
    
    np.savetxt(solution_file,np.array([known_tour_vals,solved_tour_vals]),delimiter=',')