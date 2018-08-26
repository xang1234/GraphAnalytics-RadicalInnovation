# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:25:17 2018

@author: david
"""
import os
os.chdir(r'C:\Users\david\github\GraphAnalytics-RadicalInnovation\code')


import radical_innovation_functions as f
import networkx as nx
import numpy as np


# Read csv and create graph
ml_articles="D:/Social Project/ML articles expanded keywords.csv"
G=f.create_graph(ml_articles)



# Write Graph
nx.write_gexf(G,"C:/Users/david/weighted_graph.gexf",encoding='utf-8')


# Read Graph
G=nx.read_gexf("C:/Users/david/weighted_graph.gexf")


# Clustering
G=G.to_undirected()
G_connected=max(nx.connected_component_subgraphs(G), key=len)

# Select Attribute for weight
jac=nx.get_edge_attributes(G_connected,'Jacquard')
jac_exp={k:np.exp(5*v) for (k,v) in jac.items()}
nx.set_edge_attributes(G_connected,jac_exp,'weight' )

# Louvain Clustering
Cluster_Louvain=f.cluster_subgraph_by_year_louvain(G_connected,option='accumulate',
                                     connected='yes',retain_clus='no',
                                     res_louv=5,wei='weight') 

Cluster_CNM=f.cluster_subgraph_by_year_cnm(G_connected,option='accumulate',
                                     connected='yes')


# Analyze results
Louvain=['Louvain cluster'+str(i) for i in range(1991,2017,1)] 
f.cluster_hist(Cluster_Louvain,Louvain,log=True)