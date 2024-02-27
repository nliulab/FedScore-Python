# FedScore-Python
### Introduction
Federated learning in healthcare research has primarily focused on black-box
models, leaving a notable gap in interpretability crucial for clinical decision-making. 
While scoring systems, acknowledged for their transparency, are widely employed in 
clinical science, there are notably limited privacy-preserving solutions for scoring 
system generators. FedScore, an example of such a solution, has been demonstrated 
using artificially partitioned data. In this study, we further improve FedScore and 
conduct empirical experiments utilizing real-world heterogeneous clinical data.

![Figure 1: Overview of the FedScore algorithm](Figures/workflow.jpg)

### System requirements
- Python 3.9
- To install required Python packages, run
```
pip install -r requirements.txt
```

### Test locally
- In `FedScore` directory, open three terminals. First start the server in the first terminal.
```
python server.py
```
- Then start two clients, one for each terminal.
```
python client1.py
python client2.py
```
