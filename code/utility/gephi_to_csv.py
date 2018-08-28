
# coding: utf-8

# In[1]:


# location of input files
graph_dir = 'C:/Users/U/Documents/BigData/'
#graph_dir = './'


# In[2]:


import numpy as np
import pandas as pd
import os
import networkx as nx


# In[3]:


pd.options.display.max_rows = 8


# In[4]:


gephi_input_file = 'ML Extended.gexf'
Graph = nx.read_gexf(graph_dir + gephi_input_file) 


# In[5]:


ML_df = pd.DataFrame([i[1] for i in Graph.nodes(data=True)], index=[i[0] for i in Graph.nodes(data=True)])
ML_df.index.name = 'id'
ML_df.reset_index(inplace=True)
display(ML_df)
ML_df.to_csv(graph_dir + gephi_input_file + "_nodes.csv")


# In[6]:


edges = Graph.edges(data=True)
edge_df = pd.DataFrame([i[2] for i in edges], index=[[i[0] for i in edges], [i[1] for i in edges]])
edge_df.index.names = ['id_0', 'id_1']
edge_df.reset_index(inplace=True)
display(edge_df)
edge_df.to_csv(graph_dir + gephi_input_file + "_edges.csv")

