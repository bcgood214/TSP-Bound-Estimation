#Tried to match the algorithm with the pseudocode. You can make any changes to make it work properly. 

#inputs
model = model

#Edges [origin = u, destination = v, weight = c]
E = [u , v, c]
#List of vertices
V = []
#Number of vertices
n = len(V)
#Number of edges
m = len(E)
#Starting vertex
s = V[0]
#number of iterations to approximate on
max = 100

#output in visiting order
solution = []

#Algorithmic variables
q = []  #search stack (represents the search tree)

cur_ver = s #current vertex for searching

#Best known cost and edges
solution = {"cost": float("inf"), "edges": []}

#Best current cost and edges
new_sol = {"cost": 0, "edges": []}

#Visited vertices in new_sol
new_verts = set()

#upper bound to prune
bound = float("inf")

def main():
    #Find initial solution using DFS
    cur_ver = s
    solution = dfs()

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

def dfs():
    #Check if solution needs to return
    if len(new_sol) == n:
        return new_sol
    
    #add edges to search stack
    addEdges()

    while True:
        if len(q) == 0:
            return None #error, no more edges to explore
        
        can_edge = q.pop(0)

        if can_edge["prediction"] > bound:
            continue

        if can_edge.u == cur_vert:
            new_sol["edges"].append(can_edge)
            new_sol["cost"] += can_edge["c"]
            cur_ver = can_edge.v
            new_verts.add(cur_ver)
            break
        else:
            q.prepend(can_edge)
            backtrack()
            
        return dfs()

def addEdges():
    valid_exp = []

    #find new valid edges and get their predicted value
    for (u, v, c) in E:
        if u != cur_ver:
            continue

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
        cur_ver = s
        return

    edge_rem = new_sol["edges"][-1]  #get_last_item
    new_verts.discard(edge_rem["v"])  #remove that vertex from visited
    new_sol["cost"] -= edge_rem["c"]
    new_sol["edges"].pop()           #remove_last_item

    #move current vertex back to the previous vertex
    if new_sol["edges"]:
        cur_ver = new_sol["edges"][-1]["v"]
    else:
        cur_ver = s