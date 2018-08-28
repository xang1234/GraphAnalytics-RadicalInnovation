
# coding: utf-8

# ## ver10.0

# In[1]:


# location of input files
graph_dir = 'C:/Users/U/Documents/BigData/'
#graph_dir = './'


# In[2]:


### Set colaboratory True to run in Google Colaboratory. 
colaboratory = False


# In[3]:


if colaboratory:
  get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
  get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
  get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
  get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')
  from google.colab import auth
  auth.authenticate_user()
  from oauth2client.client import GoogleCredentials
  creds = GoogleCredentials.get_application_default()
  import getpass
  get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
  vcode = getpass.getpass()
  get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')


# In[4]:


if colaboratory:
  # drive mean root directory of  google drive
  get_ipython().system('mkdir -p drive')
  get_ipython().system('google-drive-ocamlfuse drive')
  get_ipython().system('ls drive/"Colab Notebooks"/Social_Proj/data')


# In[5]:


if colaboratory:
  # location of input files
  graph_dir = 'drive/Colab Notebooks/Social_Proj/data/'
  #graph_dir = './'


# In[6]:


if colaboratory:
  get_ipython().system('pip install python-louvain')


# In[7]:


import networkx as nx
import timeit
import community
#from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import pandas as pd
 
import operator
from timeit import default_timer as timer
from collections import OrderedDict
import os


# In[8]:


pd.options.display.max_rows = 8


# In[9]:


min_year = 1991
max_year = 2016
n_year=int(max_year - min_year + 1)


# In[10]:


global_start = timer() ### Just for total calculation time measurement


# In[11]:


#gephi_input_file = 'ml_graph expanded.gexf'
#gephi_input_file = 'ml_graph.gexf'
#gephi_input_file = 'ML Extended.gexf'
gephi_input_file = 'weighted_graph.gexf'
gephi_input = True


# In[12]:


node_input = gephi_input_file
#node_input = 'ML articles expanded'
#node_input = 'ML articles'


# In[13]:


if gephi_input:
    Graph = nx.read_gexf(graph_dir + gephi_input_file) 
    
    print('Gephi file read. Networkx class : ', type(Graph).__name__)
    if type(Graph).__name__ == 'DiGraph':
        Graph = Graph.to_undirected()
        print('Graph converted to: ', type(Graph).__name__)
    
    ML_df = pd.DataFrame([i[1] for i in Graph.nodes(data=True)], index=[i[0] for i in Graph.nodes(data=True)])
    ML_df.index.name = 'id'
    ML_df.reset_index(inplace=True)
    display(ML_df)

if not gephi_input:
    #ML_df = pd.read_csv(graph_dir + 'ML articles.csv')
    #ML_df = pd.read_csv(graph_dir + 'ML articles expanded.csv')
    ML_df = pd.read_csv(graph_dir + node_input + '.csv')
    #ML_df.loc[:,'year'] = ML_df.loc[:,'year'].apply(int)
    display(ML_df)


# In[14]:



cite_df = pd.read_csv(graph_dir + 'ML_cite_adjlist.csv')
cite_df.loc[:,'year'] = cite_df.loc[:,'year'].apply(int)
cite_df


# In[15]:



### Limit the citaiton from ML papers only
cite_df = pd.merge(cite_df, ML_df[['id']], left_on ='from', right_on = 'id', how = 'inner')
cite_df.loc[:,'year'] = cite_df.loc[:,'year'].apply(int)
cite_df


# In[16]:


ML_df.set_index('id', inplace = True)
dataset_df = ML_df.rename(index=str, columns={"year": "Year"})
dataset_df


# In[17]:


if gephi_input:
    merged_df = dataset_df.copy()
    

if not gephi_input:
    

    
    oldest_in_cite_df = cite_df.groupby(['to'])[['year']].min()
    oldest_in_cite_df.index.name = 'id'
    print('# of all in-cited papers: ', len(oldest_in_cite_df))
    oldest_in_cite_df
    
    merged_df = pd.merge(dataset_df, oldest_in_cite_df, left_index = True, right_index = True, how = 'outer')
    #display(merged_df)
    #merged_df['Year'].combine_first(merged_df['year'])
    merged_df.loc[:, 'ML_flag'] = True
    merged_df.loc[pd.isnull(merged_df['Year']), 'ML_flag'] = False
    merged_df.loc[pd.isnull(merged_df['Year']), 'Year'] = merged_df.loc[pd.isnull(merged_df['Year']), 'year']
    merged_df.drop(['year'], axis=1, inplace = True)

    print(merged_df.shape)
    display(merged_df)


# In[18]:


cite_df.loc[:,'Num_Out_Citations'] = 1

out_cite_df = cite_df.groupby(['from'])[['Num_Out_Citations']].sum()
out_cite_df.index.name = 'id'
out_cite_df


# In[19]:



merged1_df = pd.merge(merged_df, out_cite_df, left_index = True, right_index = True,  how = 'left')
merged1_df.loc[:,'Num_Out_Citations'].fillna(0, inplace = True)

if gephi_input:
    cols = ['Year', 'authors', 'journalName', 'title', 'paperAbstract'] + ['Num_Out_Citations']
if not gephi_input:
    cols = ['Year', 'authors', 'journalName', 'title', 'paperAbstract'] + ['keyPhrases', 'venue', 'pdfUrls', 's2Url', 'ML_flag'] + ['Num_Out_Citations']
merged1_df = merged1_df.loc[:, cols]
print(merged_df.shape)
merged1_df


# In[20]:


merged2_df = merged1_df[merged1_df['Year'].notnull()].copy()
merged2_df.loc[:,:].fillna('_', inplace = True)

print(merged2_df.shape)
merged2_df


# In[21]:


### merged2_df.to_csv(graph_dir + 'All_Papers' '.csv')


# In[22]:


if not gephi_input:
    ### Create graph

    Graph = nx.Graph()

    ### Add edges and edge attributes (year)
    cite_a_list_df = cite_df

    adj=cite_a_list_df[['from','to']]
    adj_tuple=list(adj.itertuples(index=False,name=None))   
    Graph.add_edges_from(adj_tuple)
    #nx.write_gexf(Graph, graph_dir + 'ML+_graph.gexf') ## To check numpy float/int error


# In[23]:


if not gephi_input:
    cite_a_list_df['from_to'] = cite_a_list_df[['from', 'to']].apply(tuple, axis=1)
    edge_year_dict=dict(zip(cite_a_list_df['from_to'], [int(year) for year in cite_a_list_df['year']]))
    nx.set_edge_attributes(Graph, edge_year_dict,name='Year')
    #nx.write_gexf(Graph, graph_dir + 'ML+_graph.gexf') ## To check numpy float/int error


# In[24]:



for col in cols:    
    node_dict = merged2_df[col].to_dict()
    if col in ['Year', 'Num_Out_Citations']:
        node_dict  = {k:int(v) for (k, v) in node_dict.items()}
    if col in ['ML_flag']:
        node_dict  = {k:bool(v) for (k, v) in node_dict.items()}
    nx.set_node_attributes(Graph, node_dict, col)

### Write to Gephi files
#nx.write_gexf(Graph, graph_dir + 'ML+_graph.gexf') ## To check numpy float/int error


# In[25]:


num_nodes = len(Graph.node)
num_edges = Graph.number_of_edges()
print('# of nodes : ', num_nodes, '| # of edges : ', num_edges)


# ### Extract the largest connected component

# In[26]:


Graph = max(nx.connected_component_subgraphs(Graph), key=len)


# In[27]:


# add degree and Num_In_Citations later
degree_dict = dict(Graph.degree())
nx.set_node_attributes(Graph, degree_dict, 'Degree')
num_out_citations_dict = nx.get_node_attributes(Graph, 'Num_Out_Citations' )

num_in_citations_dict = {i: (int(degree_dict[i] - num_out_citations_dict[i])) for i in Graph.node}
#num_in_citations_dict = dict([[i, (int(degree_dict[i] - num_out_citations_dict[i]))] for i in Graph.node])
#num_in_citations_dict  = {}
#for i in Graph.node:
#    num_in_citations_dict[i] = int(degree_dict[i] - num_out_citations_dict[i])

nx.set_node_attributes(Graph,num_in_citations_dict,'Num_In_Citations')


# In[28]:


# Here we create subgraphs, nodes and edges are accumulated every year. Calculation done for CNM and Louvain method

def subgraph_by_year(Graph,option='accumulate',connected='yes'):
    """
    option='accumulate' will accumulate the nodes and edges of the graph year on year
    option='separate' will only keep the nodes year on year, edges from previous years will not be retained
    connected='yes' will only use the largest connected component for each year
    connected='no' will use all available nodes for each year
    """
    
    # get node and edge year
    node_yr=nx.get_node_attributes(Graph,'Year')
    edge_yr=nx.get_edge_attributes(Graph,'Year')
    
    # dictionarys to filter nodes and edges by year
    n_year=int(max(node_yr.values())-min(node_yr.values()))+1
    min_year=min(node_yr.values())
    list_dict_node_year=[{} for i in range(n_year)]
    list_dict_edge_year=[{} for i in range(n_year)]
    for i in range(n_year):
        if option=='accumulate':
            list_dict_edge_year[i]={k:v for (k,v) in edge_yr.items() if  v<=min_year+i}
            list_dict_node_year[i]={k:v for (k,v) in node_yr.items() if  v<=min_year+i}
        elif option=='separate':
            list_dict_edge_year[i]={k:v for (k,v) in edge_yr.items() if  v==min_year+i}
            list_dict_node_year[i]={k:v for (k,v) in node_yr.items() if  v<=min_year+i}       
        
        else:
            raise Exception("wrong keyword for option. use accumulate or separate only")
     
    print('Input Graph has',Graph.number_of_nodes(),'nodes and',Graph.number_of_edges(),'edges')  
    H=[nx.Graph() for i in range(n_year)]
    if option=='accumulate':
        for i in range(n_year):
            start = timer()
            if connected=='no':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys())).copy()
            elif connected=='yes':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys())).copy()
                 H[i]=max(nx.connected_component_subgraphs(H[i]), key=len)
            #H[i].add_nodes_from(list(list_dict_node_year[i].keys()))
            #H[i].add_edges_from(list(list_dict_edge_year[i].keys()))
            else:
                raise Exception("wrong keyword for connected. use yes or no only")
            end = timer()    
            print('Calculation Time (sec): ', "{0: 6.0f}".format(end - start), '|' + 'Year:',str(i+min_year),'--',H[i].number_of_nodes(),'nodes --',H[i].number_of_edges(),'edges')

    elif option=='separate':
        for i in range(n_year):
            start = timer()
            if connected=='no':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys())).copy()
            elif connected=='yes':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys())).copy()
                 H[i]=max(nx.connected_component_subgraphs(H[i]), key=len)
            #H[i].add_nodes_from(list(list_dict_node_year[i].keys()))
            #H[i].add_edges_from(list(list_dict_edge_year[i].keys()))
            else:
                raise Exception("wrong keyword for connected. use yes or no only")            
            
            end = timer()    
            print('Calculation Time (sec): ', "{0:.0f}".format(end - start), '|' + 'Year:',str(i+min_year),'--',H[i].number_of_nodes(),'nodes --',H[i].number_of_edges(),'edges')
            

    return H


# In[29]:


connected = 'no' ### connected='yes' option used to take too long.


# In[30]:


Graphs = subgraph_by_year(Graph,option='accumulate',connected=connected) 


# In[31]:


### Add Degree and Num_In_Citations which may increase every year 

for j in range(n_year):
    Graph = Graphs[j]
    
    # add degree and Num_In_Citations later
    degree_dict = dict(Graph.degree())
    nx.set_node_attributes(Graph, degree_dict, 'Degree')
    num_out_citations_dict = nx.get_node_attributes(Graph, 'Num_Out_Citations' )

    num_in_citations_dict = {i: (int(degree_dict[i] - num_out_citations_dict[i])) for i in Graph.node}
    #num_in_citations_dict = dict([[i, (int(degree_dict[i] - num_out_citations_dict[i]))] for i in Graph.node])
    #num_in_citations_dict  = {}
    #for i in Graph.node:
    #    num_in_citations_dict[i] = int(degree_dict[i] - num_out_citations_dict[i])

    nx.set_node_attributes(Graph,num_in_citations_dict,'Num_In_Citations')


# In[38]:


### Write to Gephi files
#for i in range(len(Graphs)):
for i in range(1):
    nx.write_gexf(Graphs[i], graph_dir + 'ML_sub_' + node_input + '_' + str(i+min_year) + ".gexf")


# ### Cluster graph

# In[32]:


try:
    Graphs
except NameError:
    ### Read Gephi files
    Graphs = [[] for i in range(n_year)]  
    for i in range(n_year):
        Graphs[i]=nx.read_gexf(graph_dir + 'ML_sub_' + node_input + '_' + str(i+min_year) + ".gexf")
else:
    print('Graphs are already in memory.')



# In[33]:


def cluster_graph(H, resolution, weight = 'weight'):
    print("------------Louvain------------------")
    results_df = pd.DataFrame()
    results_df.index.name = 'Timepoint'
    for i in range(n_year):
        start = timeit.default_timer()      
        if i == 0:
            cluster_dict = {} 
        num_clusters_last = len(set(cluster_dict.values()))
        Graph = H[i]
        num_nodes = len(Graph.nodes)
        partition_dict = {}
        num_increment = 0
        for node in Graph.nodes:
            if node in cluster_dict:
                partition_dict[node] = cluster_dict[node]
            else: 
                partition_dict[node] = num_clusters_last + num_increment
                num_increment += 1
        cluster_dict = community.best_partition(Graph, resolution=resolution, partition = partition_dict, weight = weight)
        num_clusters = len(set(cluster_dict.values()))
        nx.set_node_attributes(Graph, cluster_dict,'Louvain cluster')

        stop = timeit.default_timer()
        cal_time = stop-start
        
        num_edges = len(Graph.edges)
        modularity = community.modularity(cluster_dict,Graph)
        
        results_df.loc[(i+min_year), 'Resolution'] = resolution
        results_df.loc[(i+min_year), 'Num_Clusters'] = num_clusters
        results_df.loc[(i+min_year), 'Modularity'] = modularity
        results_df.loc[(i+min_year), 'Num_Nodes'] = num_nodes
        results_df.loc[(i+min_year), 'Num_Edges'] = num_edges
        results_df.loc[(i+min_year), 'Calculation_Time'] = cal_time
        #display(results_df.loc[(i+min_year):(i+min_year+1),:])
        print('Year: {:4d}'.format(i+min_year), "| {:6d} nodes ".format(num_nodes),"| {: 5d} clusters".format(num_clusters),"| Modularity: {:.6f}".format(modularity), " | Calculation time: {: 6.2f} sec".format(cal_time))  
    
    #w = pd.ExcelWriter(graph_dir + 'Clustering_Results' + desc + '.xlsx')
    #sheetname = 'Clustering_Results'
    #results_df.to_excel(w, sheetname)
    #w.sheets[sheetname].set_column(0, 7, 10)
    #w.save()
    results_df.to_csv(graph_dir + 'Clustering_Results_ver10' + desc + '.csv')
    display(results_df)
    
    return H


# In[39]:


resolution = 2.2
#weight = 'weight'
#weight = 'Jacquard'
weight = 'Year_Diff'

desc = '_{0:s}_connected{1:s}_res{2:.2f}_wtBy{3:s}_initpart'.format(node_input, connected, resolution, weight) 
print(desc)

Clustered=cluster_graph(Graphs, resolution=resolution, weight=weight) 


# ### Write to Gephi files
# for i in range(len(Graphs)):
#     nx.write_gexf(Graphs[i], graph_dir + 'ML_clustered_' + node_input + '_' + str(i+min_year) + ".gexf")

# ### Calculate z and P

# In[40]:


#Copy_Clus2 = Clustered.copy()
Copy_Clus2 = [g.copy() for g in Clustered.copy()]


# In[41]:


try:
    Graphs
except NameError:
    ### Read Gephi files
    Graphs = [[] for i in range(n_year)]  
    for i in range(n_year):
        Graphs[i]=nx.read_gexf(graph_dir + 'ML_clustered_' + node_input + '_' + str(i+min_year) + ".gexf")
else:
    print('Graphs are already in memory.')



# In[42]:


### Calculate P, z, and cluster size by Pandas DataFrame manipulation



#for Graph in Graphs[0:1]:
def calculate_z_p(Graph,clus_attr = 'Louvain cluster'):
    start = timer()
    
    ### Calculate P value by Pandas 
    edges = [[u,v] for (u,v) in Graph.edges()]
    n0_df = pd.DataFrame.from_records(edges, columns=['Node', 'Node Connected'])
    n1_df = pd.DataFrame.from_records(edges, columns=['Node Connected', 'Node'])
    n2_df = pd.concat([n0_df, n1_df], ignore_index = True)
    n2_df.set_index('Node Connected', inplace=True)
    #display(n2_df)
    num_edges = n2_df.shape[0]
                                   
    c_df = pd.DataFrame(pd.Series(nx.get_node_attributes(Graph,clus_attr), name='Cluster of Node Connected'))
    c_df.index.name = 'Node Connected'
    #display(c_df)
    num_nodes = c_df.shape[0]
    
    
    n3_df = pd.merge(n2_df, c_df, left_index=True, right_index=True)
    #display(n3_df)
    n3_df.loc[:,'Degree per Cluster'] = 1
    n4_df = n3_df.groupby(['Node','Cluster of Node Connected'])[['Degree per Cluster']].sum()
    n4_df.reset_index(inplace=True)
    n4_df.set_index('Node', inplace=True)
    #display(n4_df)
    
    d_df = pd.DataFrame(pd.Series(nx.get_node_attributes(Graph,'Degree'), name='Degree'))
    d_df.index.name = 'Node'
    #display(d_df)
    
    n5_df = pd.merge(n4_df, d_df, left_index=True, right_index=True, how='right')
    n5_df['Degree per Cluster'].fillna(0, inplace=True)
    n5_df.loc[:,'Degree positive'] = n5_df['Degree']
    n5_df.loc[n5_df['Degree'] == 0,'Degree_positive'] = 1 ### Avoid zero that will be enominator
    n5_df.loc[:,'Sq Ratio of Degree per Cluster'] = n5_df[['Degree per Cluster', 'Degree positive']].apply(lambda x: (x[0] / x[1]) ** 2, axis=1)
    #display(n5_df)
    
    n6_df = n5_df.reset_index().groupby('Node')[['Sq Ratio of Degree per Cluster']].sum()
    n6_df.loc[:,'P'] = 1 - n6_df['Sq Ratio of Degree per Cluster']
    #display(n6_df)
    
    P_dict_np = n6_df['P'].to_dict()
    #display(P_dict_np.items())
    P_dict = dict([k,float(v)] for (k,v) in P_dict_np.items())
    #display(P_dict)
    
    nx.set_node_attributes(Graph,P_dict,'P')
    
    ### Calculate z
    
    #display(n3_df)
    z4_df = n3_df.reset_index()
    z4_df.set_index('Node', inplace=True)
    #display(z4_df)
    
    c2_df = c_df.rename(columns={'Cluster of Node Connected':'Cluster of Node'})
    c2_df.index.name = 'Node'
    #display(c2_df)
    z5_df = pd.merge(z4_df, c2_df, left_index=True, right_index=True)
    z5_df.loc[:,'Within Cluster Flag'] = z5_df[['Cluster of Node', 'Cluster of Node Connected']].apply(lambda x: x[0] == x[1], axis=1)
    #display(z5_df)
    #display(z5_df.groupby(['Within Cluster Flag']).count())
    z6_df = z5_df.loc[z5_df['Within Cluster Flag'] == True,:]
    z6_df.reset_index(inplace=True)
    #display(z6_df)
    z7_df = z6_df.groupby(['Node','Cluster of Node'])[['Degree per Cluster']].sum()
    #display(z7_df)
    #z7_2_df = z7_df.copy()
    #z7_2_df.loc[:,'Degree per Cluster Positive'] = z7_2_df['Degree per Cluster']
    #z7_2_df = z7_2_df.groupby(['Cluster of Node'])[['Degree per Cluster Positive']].transform(lambda x: x.std())
    #z7_2_df.loc[z7_2_df['Degree per Cluster Positive'] == 0,'Degree per Cluster Positive'] = 1 ### Avoid zero that will be enominator 
    #z7_df.loc[:,'Degree per Cluster Positive'] = z7_2_df['Degree per Cluster Positive']
    #z8_df = z7_df.groupby(['Cluster of Node'])[['Degree per Cluster', 'Degree per Cluster Positive']].transform(lambda x: ((x[0] - x[0].mean()) / x[1]), axis=1)
    z8_df = z7_df.groupby(['Cluster of Node'])[['Degree per Cluster']].transform(lambda x: ((x - x.mean()) / x.std()))
    z8_df.reset_index(inplace=True)
    z8_df.set_index('Node', inplace=True)
    #display(z8_df)
    z_dict_np = z8_df['Degree per Cluster'].to_dict()
    z_dict = dict([k,float(v)] for (k,v) in z_dict_np.items())
    #display(z_dict)
    nx.set_node_attributes(Graph,z_dict,'z')
    
    ##### Calculate correct P value (20180412)
    P8_df = z7_df.reset_index()
    P8_df.set_index('Node', inplace=True)
    display(P8_df)
    P9_df = pd.merge(P8_df, d_df, left_index=True, right_index=True, how='right')
    display(P9_df)
    #P9
    
    #####
    
    ### Calculate cluster size
    #display(c_df)
    #c2_df = c_df.reset_index()
    c2_df = c_df.copy()
    c2_df.loc[:,'Cluster Size'] = 1
    #display(c2_df)
    s_df = c2_df.groupby(['Cluster of Node Connected'])[['Cluster Size']].transform(sum)
    #display(s_df)
    csize_dict_np = s_df['Cluster Size'].to_dict()
    csize_dict = dict([k,int(v)] for (k,v) in csize_dict_np.items())
    #display(csize_dict)
    nx.set_node_attributes(Graph,csize_dict,'Cluster_Size')
    
    end = timer()
    print("#{0: 7d} nodes".format(num_nodes), " | {0: 7d} edges".format(num_edges), ' | Time for calculation of z and P: {0:.0f} sec'.format(end - start))


# In[43]:


def get_oldest_paper(Graph,clus_attr):
    clusters=nx.get_node_attributes(Graph,clus_attr)
    n_clus=len(set(clusters.values()))
    
    seed_df=pd.DataFrame()
    for i in range(n_clus):
        c= {k for (k,v) in clusters.items() if  v==i}
        H=nx.subgraph(Graph,c)
        old=min(nx.get_node_attributes(H,'Year').values())
        paper_id=[k for (k,v) in nx.get_node_attributes(H,'Year').items() if v==old]       
        
        J=nx.subgraph(H,paper_id)
        
        df =  pd.DataFrame([i[1] for i in J.nodes(data=True)], index=[i[0] for i in J.nodes(data=True)])
        df.loc[:,'cluster'] = i
        df.index.name = 'id'
        df.reset_index(inplace=True)
        df.index.name = 'index'
        df = df.loc[:,['Year','cluster','cluster_size','id','authors','journalName','title','paperAbstract','Num_Out_Citations','degree','Num_In_Citations', 'z','P']]
        #display(df)

        seed_df=seed_df.append(other=df)
    return seed_df
 


# def write_oldest_papers(seed_df, year):
#     
#     seed_df.sort_values(by = ['Year', 'cluster'], ascending = False, inplace = True)
#     seed_df.reset_index(drop=True, inplace=True)
#     seed_df.index.name = 'index'
#     #seed_df = seed_df.drop(['id'],axis=1)
# 
#     #seed_df.to_csv(graph_dir + 'oldest_paper_' + str(i+min_year) '.csv') 
#     w = pd.ExcelWriter(graph_dir + 'Oldest_Papers_' + str(year) + '.xlsx')
#     sheetname = 'Oldest_Papers'
#     seed_df.to_excel(w, sheetname)
#     w.sheets[sheetname].set_column(7, 8, 50)
#     w.sheets[sheetname].autofilter(0,0,seed_df.shape[0] + 1, seed_df.shape[1] + 1)
#     w.save()

# def write_all_papers(Graph, year):
#     ML_df = pd.DataFrame([i[1] for i in Graph.nodes(data=True)], index=[i[0] for i in Graph.nodes(data=True)])
#     ML_df.index.name = 'id'
#     ML_df.reset_index(inplace=True)
#     ML_df.index.name = 'index'
#     ML_df.to_csv(graph_dir + 'All_Papers_' + str(year) + '.csv')

# In[44]:


start = timer()
df_list = []
for year in range(min_year, max_year+1):
    print('[Processing year:' + str(year) + ']')
    Graph = Copy_Clus2[year-min_year]
    calculate_z_p(Graph,'Louvain cluster')
    nodes = Graph.nodes(data=True)
    df = pd.DataFrame([i[1] for i in nodes], index=[i[0] for i in nodes])
    df.index.name = 'id'
    df.reset_index(inplace=True)
    df.loc[:, 'Timepoint'] = year
    #print(df.columns)
    df_list.append(df[['id', 'Timepoint', 'Louvain cluster', 'Cluster_Size', 'z', 'P', 'Degree', 'Num_In_Citations']])

end = timer()
print('\n[Finished to calculate z and P for all years] \n Total calculation time: {0:.0f} sec'.format(end - start))


# In[45]:


stack_df = pd.concat(df_list) 
stack_df.sort_values(['id', 'Timepoint'], inplace=True)
stack_df.reset_index(drop=True, inplace=True)
stack_df.index.name = 'index'
display(stack_df)


# In[46]:


#stack_df.to_csv(graph_dir + 'All_Papers_Timepoint_Stack' + desc + '.csv', encoding='utf-8')


# In[47]:


stack_df.set_index(['id', 'Timepoint'])


# In[48]:


output_unstack = True


# In[49]:


if output_unstack:
    unstack_df = stack_df.set_index(['id', 'Timepoint']).unstack()
    unstack_df


# In[50]:


if output_unstack:
    unstack_df.columns = ['{0[0]:s} {0[1]:4d}'.format(col) for col in unstack_df.columns]
    unstack_df.index.name = 'id'
    display(unstack_df)


# In[51]:


if output_unstack:
    unstack_w_df = unstack_df.reset_index()
    display(unstack_w_df)
    unstack_w_df.index.name = 'index'
    #unstack_w_df.to_csv(graph_dir + 'All_Papers_Timepoint_Untack' + desc + '.csv', encoding='utf-8')


# In[52]:


Graph = Copy_Clus2[-1]
nodes = Graph.nodes(data=True)
df = pd.DataFrame([i[1] for i in nodes], index=[i[0] for i in nodes])
df.index.name = 'id'



# In[53]:


### Add Oldest in cluster flag

df.loc[:,'Oldest_in_Cluster Year'] = df.groupby(['Louvain cluster'])['Year'].transform(min)
df.loc[:,'Oldest_in_Cluster Flag'] = df[['Year', 'Oldest_in_Cluster Year']].apply(lambda x: x[0] == x[1], axis=1)
df


# In[54]:


cols = [col for col in ['ML_flag', 'Oldest_in_Cluster Flag', 'Oldest_in_Cluster Year', 'Year', 'authors', 'journalName', 'venue', 'pdfUrls', 's2Url', 'keyPhrases', 'title', 'paperAbstract', 'Num_Out_Citations'] if col in df.columns.tolist()]
cols


# In[55]:


static_df = df[cols].copy()


# In[56]:


if output_unstack:
    complete_df = pd.merge(static_df, unstack_df, left_index = True, right_index = True)
    display(complete_df)
    complete_w_df = complete_df.reset_index()
    complete_w_df.index.name = 'index'
    complete_w_df.to_csv(graph_dir + 'All_Papers_Complete_ver10' + desc + '.csv', encoding='utf-8')


# In[57]:


measures = ['Cluster_Size', 'z', 'P', 'Degree', 'Num_In_Citations']


# In[58]:


transform_dict = {col: {col+'_diff':'diff', col+'_shift':'shift'} for col in measures}
display(transform_dict)
stack_diff_df = stack_df.groupby('id').agg(transform_dict)
stack_diff_df


# In[59]:


stack_diff_df.columns = ['{0[1]:s}'.format(col) for col in stack_diff_df.columns]
stack_diff_df.index.name = 'index'
display(stack_diff_df)


# In[60]:


#stack2_df = pd.merge(stack_df, stack_diff_df, left_index = True, right_index = True)
#stack2_df


# In[61]:


stack2_df = stack_df.copy()
for measure in measures:
    stack2_df.loc[:,measure+'(d)'] = stack_diff_df.loc[:,measure+'_diff'] / stack_diff_df.loc[:,measure+'_shift']
stack2_df
    


# In[62]:


stack2_df = stack2_df.reset_index(drop=True)
stack2_df.set_index('id', inplace=True)
stack2_df


# In[63]:


static2_df = df[['Oldest_in_Cluster Flag', 'Oldest_in_Cluster Year', 'Year', 'title', 'Num_Out_Citations']].copy()
static2_df


# In[64]:


long_df = pd.merge(static2_df, stack2_df, left_index = True, right_index = True)
display(long_df)
long_w_df = long_df.reset_index()
long_w_df.index.name = 'index'
long_w_df.to_csv(graph_dir + 'All_Papers_Long_ver10' + desc + '.csv', encoding='utf-8')



# In[65]:


long2_df = long_df.copy()
long2_df.groupby('Cluster_Size')[['Louvain cluster']].nunique()


# In[66]:


start = timer()
nx.write_gexf(Graph, graph_dir + 'ML_zP_ver10_' + str(max_year) + desc + '.gexf')
end = timer()
print('Time to write Gephi file: {0:.0f} sec'.format(end - start)) 


# In[67]:


global_end = timer()
print('Time for all the processings: {0:.0f} sec'.format(global_end - global_start)) 

