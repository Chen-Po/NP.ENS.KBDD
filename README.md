# NEKBDD: a nonparametric ensemble knowledge-based and data-driven method for genetic network construction


## Description
NEKBDD is a Python library designed to construct genetic networks. The NEKBDD algorithm can integrate knowledge-based (pathway analysis) and data-driven (statistical modeling) methods. It can summarize the network structure using available biological information, explore network properties, and generate a final ensemble network. It is a flexible approach that can assign various degree distributions for different scientific questions and construct networks without assuming specific data distribution.

## Install from PyPi
```
pip install NEKBDD
```
### The procedures includes the following steps (as illustrated in tutorial file):
1. Import packages
2. Import KEGG signaling pathways as reference networks (XML files)
3. Estimate the parameter of power-law distribution from reference networks
4. Input data (gene expression data for calculating confidence matrix)
5. Potential network structures construction
6. Deviation calculated by 3 network properties
7. Labeling by permutation and calculating network score
8. Plot the 1st potential network
9. Construct the ensemble network
10. Plot the ensemble network and proportion network


<img width="1478" alt="Graphical abstract" src="https://github.com/user-attachments/assets/b577b12f-812f-44e6-ba65-f98a395f0a7b">

![image](https://github.com/Chen-Po/KBDD/assets/109202495/5c619eda-8f82-488b-b9cc-9bc76d1e6a7f)


## Reference
Chen-Po Liao, Hung-Ching Chang, Chuhsing Kate Hsiao. "A nonparametric ensemble knowledge-based and data-driven method for genetic network construction
" (2024)
