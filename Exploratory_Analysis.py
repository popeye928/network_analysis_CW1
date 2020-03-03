#!/usr/bin/env python
# coding: utf-8

# Conduct exploratory analysis on the graph

# References:

# https://towardsdatascience.com/graph-algorithms-part-2-dce0b2734a1d
# GitHub/Social-network-Graph-Link-Prediction---Facebook-Challenge-master/FB_EDA.ipynb#

# In[1]:

import numpy as np
import matplotlib.pylab as plt
import networkx as nx
import pandas as pd 
from networkx.algorithms import tree

#%%

# 1. Exploratory Data Analysis

# Reading the file. "DiGraph" is telling to reading the data with node-node. "nodetype" will identify whether the node is number or string or any other type.

g = nx.read_edgelist("Amazon0302.txt",create_using=nx.DiGraph(), nodetype = int)

# check if the data has been read properly or not.

print(nx.info(g))

#%%
print(nx.is_weakly_connected(g))
print(nx.number_weakly_connected_components(g))
print(nx.is_strongly_connected(g))
print(nx.number_strongly_connected_components(g))

#%%

# Create a subgraph to show the basic structure

pd.read_csv('Amazon0302.txt', nrows=500).to_csv('Amazon0302_sample.txt',header=False,index=False)

subgraph = nx.read_edgelist("Amazon0302_sample.txt",create_using=nx.DiGraph(), nodetype = int)
pos=nx.spring_layout(subgraph)
print(nx.info(subgraph))
nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
plt.savefig("graph_sample.pdf")


#%%

mst = tree.minimum_spanning_edges(g, algorithm='prim', data=False)
edgelist = list(mst)
sorted(edgelist)

#%%

# No of Unique nodes 

degree_sequence = list(g.degree())
nb_nodes = len(g.nodes())
nb_arr = len(g.edges())
avg_degree = np.mean(np.array(degree_sequence)[:,1])
med_degree = np.median(np.array(degree_sequence)[:,1])
max_degree = max(np.array(degree_sequence)[:,1])
min_degree = np.min(np.array(degree_sequence)[:,1])

print("Number of nodes : " + str(nb_nodes))
print("Number of edges : " + str(nb_arr))
print("Maximum degree : " + str(max_degree))
print("Minimum degree : " + str(min_degree))
print("Average degree : " + str(avg_degree))
print("Median degree : " + str(med_degree))

#%% Node Degree Distribution 

degree_freq = np.array(nx.degree_histogram(g)).astype('float')
plt.figure(figsize=(12, 8))
plt.stem(degree_freq)
plt.ylabel("Frequence")
plt.xlabel("Degree")
plt.show()


indegree_dist = list(dict(g.in_degree()).values())
#print(indegree_dist)
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of In-Degree')
plt.show()


### 90-100 percentile
### 99-100 percentile
for i in range(10,110,10):
    print(i,'percentile value is',np.percentile(indegree_dist,i))
for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(indegree_dist,90+i))
    
degree_freq = np.array(nx.degree_histogram(g)).astype('float')
plt.figure(figsize=(12, 8))
plt.stem(degree_freq)
plt.ylabel("Frequence")
plt.xlabel("Degree")
plt.show()


outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of Out-Degree')
plt.show()


for i in range(10,100,10):
    print(i,'percentile value is',np.percentile(outdegree_dist,i))

### 90-100 percentile
for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(outdegree_dist,90+i))


### 99-100 percentile
for i in range(10,110,10):
    print(99+(i/100),'percentile value is',np.percentile(outdegree_dist,99+(i/100)))


