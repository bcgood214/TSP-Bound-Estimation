# TSP-Bound-Estimation

Ian Rybak, Benjamin Good, Raul Baez, Christian Garcia  
Advanced Algorithms - Fall 2025  
The University of Texas at El Paso

Requirements:  
- Make your own venv, since part of git ignore
  - `python -m venv ./venv`
- Activate environment
  - `source ./venv/bin/activate`
- Install requirements 
  - `python -m pip install -r ./requirements.txt`
  - May need to install prerequisite on system (~20 mb): 
    - Linux-based command: `sudo apt install libgraphviz-dev graphviz`
    - May need to install graphviz separately if running on Windows

Running the algorithm:  
`python tsp-main.py {path-to-folder-of-graphs} {path-to-solution.csv} {path-to-model.keras}`

Optional arguments:
- `-v` or `--model_verbose` = *0* for no forward pass output, 1 otherwise
- `-d` or `--debug` = True or *False* for more output text used during development
- `-s` or `--solver_verbose` = True or *False* for information on the solver progress as it moves on to each graph
- `-i` or `--iterations` = set the max iterations, default *10*, for the searching. Smaller will stop getting deeper solutions faster, larger should produce better results

Example run:  
`python tsp-main.py validationGraphs/5 validationGraphs/5/test.csv models/3_larger.keras -v 1 -d True`  
Runs the algorithm on the 2000 graphs of 5 nodes, saving the result to a file called test.csv in their folder, using the 3rd larger model, turning on the verbose output of the forward pass and the debugging.
