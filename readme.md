# Improvements to Scalable Graph Neural Networks for Friend Recommendations

This is an adaptation of *Friend Recommendations with Self-Rescaling Graph Neural Networks* by Kiran Song.

My contribution is the extension of PPRGo's "Push-Flow" Personalized Page-Rank algorithm to determine the most important 
neighbors in large homogenous graphs. This pre-processing allows us to fix the size of neighborhoods for every node, 
stabilizing memory and reducing convergence time. 

## Dependencies

*Python* 3.8

*Modules*: The dependencies are listed in `requirements.txt`. However, there are some caveats.
    
- To utilize DGL w/GPU install dgl-cu111
- To utilze torch w/GPU install 1.8.0+cu11
- Visual Studious C++ Tools version 13 or greater. There are sub-dependencies in here I didn't document. However, a simple google search on produced exceptions quickly reveal the solution. 
- Bash is required. If you are window user I recommend using [Git](https://git-scm.com/) Bash. 
 
 
## Data format

All Data should be stored in a separate folder than these one. The convention used in these scripts is '../data'
- The original unprocessed Pokec data can be found [here](https://snap.stanford.edu/data/soc-pokec.html). 
- The original unprocessed LiveJournal data can be found [here](https://snap.stanford.edu/data/soc-LiveJournal1.html). 

However, Song didn't share how to preprocess the data. Instead, Song shares the processed data in the proper format for 
these scripts [here](https://drive.google.com/file/d/1MGIQyZwZQgIMn53ih6wulcrTaraxHFin/view)

## Model running

There are three models used in my project. PPRGo, LightGCN, and PPRLightGCN.
- PPRGO and PPRLightGCN require run_ppr.py as a prerequisite. 
- To change datasets alter global variable in './Script/run_model_name.sh'
- To alter model hyperparameters change: './config/common_gnn-config.yaml'
- Top-K: To change top-k used in PPRGO/LightGCN change the TOP_K global variable in either script. 

## Miscellaneous
- './DataModifications/split.py': Used to subset Song's data. 
- './HashEmbedding/Encoding.py': Contains the reproduced LHS Projections algorithm for [hashing node features](https://arxiv.org/abs/2208.05648). 
- './ReverseGraph/Reverse.py': Reproduced code for producing [Reverse Graphs](https://ieeexplore.ieee.org/document/10069377)
