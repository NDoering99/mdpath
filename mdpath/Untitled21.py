#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MDAnalysis as mda
from MDAnalysisTests.datafiles import GRO, XTC
import pandas as pd
from MDAnalysis.analysis.dihedrals import Dihedral
from sklearn.metrics import mutual_info_score
import numpy as np
from Bio import PDB
import networkx as nx
from tqdm import tqdm
from itertools import combinations
   
# Distance between atoms 
def calculate_distance(atom1, atom2):
    distance_vector = atom1 - atom2
    distance = np.linalg.norm(distance_vector)
    return distance

# Define paths to PSF and XTC files
psf_file = "D:/OT/Mu sim/run1/11882_dyn_199.psf"
xtc_file = "D:/OT/Mu sim/run1/11888_trj_199.xtc"

# Create Universe with only the last 1000 frames
u = mda.Universe(psf_file, xtc_file, start=-1000)
x = 351
y = 64

# Dihedral angle movements
df_all_residues = pd.DataFrame()
for i in tqdm(range(y, x + 1), desc="Processing residues"):
    try:
        res = u.residues[i]
        if res is None:
            continue  
        ags = [res.phi_selection()]
        R = Dihedral(ags).run()
        dihedrals = R.results.angles
        dihedral_movements = np.diff(dihedrals, axis=0)
        df_residue = pd.DataFrame(dihedral_movements, columns=[f'Res {i}'])
        df_all_residues = pd.concat([df_all_residues, df_residue], axis=1)
    except Exception as e:
        continue


# In[2]:


import numpy as np
from scipy.stats import entropy


normalized_mutual_info = {}

# Number of bins for histogram
num_bins = 35

total_iterations = len(df_all_residues.columns) ** 2
progress_bar = tqdm(total=total_iterations, desc="Calculating Normalized Mutual Information")

for col1 in df_all_residues.columns:
    for col2 in df_all_residues.columns:
        if col1 != col2:
            hist_col1, _ = np.histogram(df_all_residues[col1], bins=num_bins)
            hist_col2, _ = np.histogram(df_all_residues[col2], bins=num_bins)
            hist_joint, _, _ = np.histogram2d(df_all_residues[col1], df_all_residues[col2], bins=num_bins)
            # Compute MI 
            mi = mutual_info_score(hist_col1, hist_col2, contingency=hist_joint)
            # Compute Entropies 
            entropy_col1 = entropy(hist_col1)
            entropy_col2 = entropy(hist_col2)
            # Compute NMI based on entropy
            nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
            normalized_mutual_info[(col1, col2)] = nmi
            progress_bar.update(1)

progress_bar.close()
print(normalized_mutual_info)



mi_diff_df = pd.DataFrame(normalized_mutual_info.items(), columns=['Residue Pair', 'MI Difference'])
max_mi_diff = mi_diff_df['MI Difference'].max()
mi_diff_df['MI Difference'] = max_mi_diff - mi_diff_df['MI Difference']

# Firstframe for pdb parsing 
first_frame = u.trajectory[-1]
with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
    pdb.write(u.atoms)
    
pdb_file = "first_frame.pdb"
residue_graph = nx.Graph()

parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure('pdb_structure', pdb_file)
heavy_atoms = ['C', 'N', 'O', 'S']

# Build graph on close residues
for model in structure:
    for chain in model:
        residues = [res for res in chain if res.get_id()[0] == ' ']
        for res1, res2 in tqdm(combinations(residues, 2), desc="Processing residues", total=len(residues)*(len(residues)-1)//2):
            res1_id = res1.get_id()[1]
            res2_id = res2.get_id()[1]
            if res1_id <= x and res2_id <= x: 
                for atom1 in res1:
                    if atom1.element in heavy_atoms:
                        for atom2 in res2:
                            if atom2.element in heavy_atoms:
                                distance = calculate_distance(atom1.coord, atom2.coord)
                                if distance <= 5.0:
                                    residue_graph.add_edge(res1.get_id()[1], res2.get_id()[1], weight=0)

for edge in residue_graph.edges():
    u, v = edge  
    pair = ('Res ' + str(u), 'Res ' + str(v))  # Creating key in the same format as in normalized_mutual_info
    if pair in normalized_mutual_info:
        residue_graph.edges[edge]['weight'] = normalized_mutual_info[pair]


# Remove this later
for edge in residue_graph.edges():
    print(edge, residue_graph.edges[edge]['weight'])
print(residue_graph)
print(mi_diff_df)


# In[3]:


import pandas as pd
from itertools import combinations
from tqdm import tqdm

# get far away residues for paths 
for chain in model:
    residues = [res for res in chain if res.get_id()[0] == ' ']
    for res1, res2 in tqdm(combinations(residues, 2), desc="Processing residues", total=len(residues)*(len(residues)-1)//2):
        res1_id = res1.get_id()[1]
        res2_id = res2.get_id()[1]
        if res1_id <= x and res2_id <= x:  
            are_distant = True
            for atom1 in res1:
                if atom1.element in heavy_atoms:
                    for atom2 in res2:
                        if atom2.element in heavy_atoms:
                            distance = calculate_distance(atom1.coord, atom2.coord)
                            if distance <= 12.0:
                                are_distant = False
                                break 

                    if not are_distant:
                        break 
            if are_distant:
                distant_residues.append((res1.get_id()[1], res2.get_id()[1]))

df_distant_residues = pd.DataFrame(distant_residues, columns=['Residue1', 'Residue2'])

# remove later
print(df_distant_residues)
df_distant_residues.to_excel("distant_residues3.xlsx", index=False)


# In[4]:


import networkx as nx

# Find the shortest path based on maximum mutual information
def max_weight_shortest_path(graph, source, target):
    shortest_path = nx.dijkstra_path(graph, source, target, weight='weight')
    total_weight = sum(graph[shortest_path[i]][shortest_path[i + 1]]['weight'] for i in range(len(shortest_path) - 1))
    return shortest_path, total_weight
 
# Collect paths and their total weights
path_total_weights = []
for index, row in df_distant_residues.iterrows():
    try:
        shortest_path, total_weight = max_weight_shortest_path(residue_graph, row['Residue1'], row['Residue2'])
        path_total_weights.append((shortest_path, total_weight))
    except nx.NetworkXNoPath:
        continue

# Sort paths based on the sum of their weights
sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)

#remove this later
for path, total_weight in sorted_paths[:500]:
    print("Path:", path, "Total Weight:", total_weight)


# In[6]:


import pandas as pd
from itertools import combinations
from tqdm import tqdm

# List to store pairs of residues closer than or equal to 10 Å apart
close_residues = []
for chain in model:
    residues = [res for res in chain if res.get_id()[0] == ' ']
    for res1, res2 in tqdm(combinations(residues, 2), desc="Processing residues", total=len(residues)*(len(residues)-1)//2):
        res1_id = res1.get_id()[1]
        res2_id = res2.get_id()[1]
        if res1_id <= x and res2_id <= x:
            are_close = False
            for atom1 in res1:
                if atom1.element in heavy_atoms:
                    for atom2 in res2:
                        if atom2.element in heavy_atoms:
                            distance = calculate_distance(atom1.coord, atom2.coord)
                            if distance <= 10.0:
                                are_close = True
                                break  
                    if are_close:
                        break 
            if are_close:
                close_residues.append((res1_id, res2_id))

df_close_residues = pd.DataFrame(close_residues, columns=['Residue1', 'Residue2'])


# In[8]:


import tqdm
import pandas as pd

#Computation of overlap by comparing every residue of every path with each other 
pathways = [path for path, _ in sorted_paths[:500]]
overlap_df = pd.DataFrame(columns=['Pathway1', 'Pathway2', 'Overlap'])

def overlap(df, res1, res2):
    return ((df["Residue1"] == res1) & (df["Residue2"] == res2)) | ((df["Residue1"] == res2) & (df["Residue2"] == res1))

for i, path1 in enumerate(tqdm.tqdm(pathways)):
    for j, path2 in enumerate(pathways):
        count_true = 0
        for res1 in path1:
            for res2 in path2:
                if overlap(df_close_residues, res1, res2).any():
                    count_true += 1
        overlap_df = overlap_df.append({'Pathway1': i, 'Pathway2': j, 'Overlap': count_true}, ignore_index=True)

print(overlap_df)


# In[9]:


overlap_df.to_excel("overlap_data.xlsx", index=False)


# In[2]:


import pandas as pd
from scipy.cluster import hierarchy
import plotly.figure_factory as ff
import plotly.graph_objs as go

overlap_df = pd.read_excel("overlap_data.xlsx")

#Distance matrix based on overlap
overlap_matrix = overlap_df.pivot(index='Pathway1', columns='Pathway2', values='Overlap').fillna(0)
distance_matrix = 1 - overlap_matrix
linkage_matrix = hierarchy.linkage(distance_matrix.values, method='complete')
fig = ff.create_dendrogram(distance_matrix.values, orientation='bottom', labels=overlap_matrix.index)
fig.update_layout(title='Hierarchical Clustering Dendrogram',
                  xaxis=dict(title='Pathways'),
                  yaxis=dict(title='Distance'),
                  xaxis_tickangle=-90)

fig.show()



# In[ ]:




