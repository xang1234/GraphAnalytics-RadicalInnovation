# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:13:42 2018

@author: david
"""


import pandas as pd
import ast
import time
import progressbar
import gc
import networkx as nx
import community
import numpy as np
from itertools import islice
import operator 
import json
import matplotlib.pyplot as plt
import seaborn as sns

def create_graph(ml_articles):
    """
    Creates a networkx graph from Semantic Scholar csv file
    """
    start=time.time()
    ### Read Data and drop Articles without 
    print('reading file')
    df_ml=pd.read_csv(ml_articles)
    df_ml=df_ml[[(len(ast.literal_eval(x)) > 0) for x in df_ml['outCitations']]]
    df_ml=df_ml.dropna(subset = ['year'])
    df_ml=df_ml.reset_index(drop=True)
    
    #get incoming and outgoing citations from the dataframe as set
    in_cite_set=set()
    out_cite_set=set()
    my_set=set(df_ml['id'])
    print('calculating in and out citations')
    p = progressbar.ProgressBar()
    p.start()
    
    for i in range(len(df_ml['id'])): #for each article in this database
        p.update(i/len(df_ml['id']) * 100)
        outCite = ast.literal_eval(df_ml['outCitations'][i])  
        inCite = ast.literal_eval(df_ml['inCitations'][i])
        for k in range(len(outCite)):
            out_cite_set.add(outCite[k])

        for j in range(len(inCite)):
            in_cite_set.add(inCite[j])
         
    p.finish()
        
    # create dataframe to record edge list with year attribute : 
    # we only have to scan out going citations and the year is the year of the article
    c = dict()
    cite_adj_list = []
    print('Collecting Graph Data')
    p = progressbar.ProgressBar()
    p.start()
    c_original = dict()
    art_id=df_ml['id']
    for i in range(len(df_ml['id'])): #for each article in this database
        p.update(i/len(df_ml['id']) * 100)
        outCite=ast.literal_eval(df_ml['outCitations'][i])
    
        for k in range(len(outCite)):
            if outCite[k] in my_set:
                c = c_original.copy()
                c['from'] = df_ml['id'][i]
                c['to'] = outCite[k]
                c['year'] = df_ml['year'][i]
                #l=df_ml.index[df_ml['id']==outCite[k]].values[0]
                l = np.flatnonzero(art_id == outCite[k]).item(0)
                
                c['year diff']=(df_ml['year'][i]-df_ml['year'][l]+5)
                c['year sum']=(df_ml['year'][i]+df_ml['year'][l])
                a=set(ast.literal_eval(df_ml['outCitations'][i]))
                b=set(ast.literal_eval(df_ml['outCitations'][l]))
                c['Jacquard']=len(a&b)/len(a|b)
                
                #c['common author']=False
                #c['common PI']=False 
                #author_in=ast.literal_eval(df_ml['authors'][i])
                #author_out=ast.literal_eval(df_ml['authors'][l])
                #set_author_in=set()
                #set_author_out=set()
                #for kk in range(len(author_in)):
                #    if len(author_in[kk]['ids'])>0:
                #        set_author_in.add(author_in[kk]['ids'][0])
                #for ll in range(len(author_out)):
                #    if len(author_out[ll]['ids'])>0:
                #        set_author_out.add(author_out[ll]['ids'][0])
                #if len(set_author_out & set_author_in) > 0 :
                #    c['common author']=True
                #if len(author_in)>0 and len(author_out)>0:
                #    if len(author_in[0]['ids'])>0 and len(author_out[0]['ids'])>0:
                #        if author_in[0]['ids'][0]==author_out[0]['ids'][0]: 
                #            c['common PI']=True
                cite_adj_list.append(c)
    p.finish()            
    cite_a_list_df = pd.DataFrame(cite_adj_list)
    cite_a_list_df=cite_a_list_df.dropna(subset = ['year'])
    cite_a_list_df['from_to'] = cite_a_list_df[['from', 'to']].apply(tuple, axis=1)
    edge_year_dict=dict(zip(cite_a_list_df['from_to'],cite_a_list_df['year'].astype('int')))
    
    # create dictionary of edge tuples (from,to) so that networkx can be updated


    edge_year_diff_dict=dict(zip(cite_a_list_df['from_to'],cite_a_list_df['year diff'].astype('int')))
    edge_year_sum_dict=dict(zip(cite_a_list_df['from_to'],cite_a_list_df['year sum'].astype('int')))
    edge_jacquard_dict=dict(zip(cite_a_list_df['from_to'],cite_a_list_df['Jacquard']))
    #edge_c_author_dict=dict(zip(cite_a_list_df['from_to'],cite_a_list_df['common author']))
    #edge_c_PI_dict=dict(zip(cite_a_list_df['from_to'],cite_a_list_df['common PI']))
    # create dictionaries of the attributes so that networkx can be updated

    df_ml=df_ml.dropna(subset = ['year'])
     
  
    node_year_dict=dict(zip(df_ml['id'],df_ml['year'].astype('int')))
    node_keyPhrases_dict=dict(zip(df_ml['id'],df_ml['keyPhrases']))
    #node_journalName_dict=dict(zip(df_ml['id'],df_ml['journalName']))
    node_title_dict=dict(zip(df_ml['id'],df_ml['title']))
    node_authors_dict=dict(zip(df_ml['id'],df_ml['authors']))
    #node_paperAbstract_dict=dict(zip(df_ml['id'],df_ml['paperAbstract']))
    

    
    # create graph in networkx
    
    print('Building Graph')
    adj=cite_a_list_df[['from','to']]
    adj_tuple=list(adj.itertuples(index=False,name=None))
    
    #node_paperAbstract_dict_strip={k:unicodetoascii(v) for (k,v) in node_paperAbstract_dict.items()}
    del df_ml, cite_a_list_df, cite_adj_list, c, c_original
    gc.collect()                                
    
    G=nx.DiGraph()
    G.add_edges_from(adj_tuple)

    nx.set_node_attributes(G,node_year_dict,name='Year')
    nx.set_node_attributes(G,node_keyPhrases_dict,name='keyPhrases')
    #nx.set_node_attributes(G,node_journalName_dict,name='journalName')
    nx.set_node_attributes(G,node_title_dict,name='title')
    nx.set_node_attributes(G,node_authors_dict,name='authors')
    #nx.set_node_attributes(G,node_paperAbstract_dict_strip,name='paperAbstract')

    nx.set_edge_attributes(G,edge_year_dict,name='Year')
    nx.set_edge_attributes(G,edge_year_diff_dict,name='Year_Diff')
    nx.set_edge_attributes(G,edge_year_sum_dict,name='Year_Sum')
    nx.set_edge_attributes(G,edge_jacquard_dict,name='Jacquard')
    #nx.set_edge_attributes(G,edge_c_author_dict,name='common author')
    #nx.set_edge_attributes(G,edge_c_PI_dict,name='common PI')



    print("number of nodes in dataset -",G.number_of_nodes())
    print("number of edges in dataset -",G.number_of_edges())
    print("number of unique in citations -",len(in_cite_set))
    print("number of unique out citations -",len(out_cite_set))
    print("number of unique in citations not in dataset -",len(in_cite_set-my_set))
    print("number of unique out citations not in dataset -",len(out_cite_set-my_set))
    print("number of total in and out citations not in dataset -",len((in_cite_set|out_cite_set)-my_set))
    print("time taken",round(time.time()-start,1))
    
    del in_cite_set, out_cite_set, my_set, adj, adj_tuple
    del node_year_dict, node_title_dict, node_keyPhrases_dict, node_authors_dict, edge_year_dict, edge_year_diff_dict,edge_year_sum_dict,edge_jacquard_dict
    gc.collect()

    return(G)

def delete_edge_attr(G,att_list):
    """
    Deletes Edge Attributes from Graph
    """
    for n1, n2, d in G.edges(data=True):
        for att in att_list:
            d.pop(att, None)
def delete_node_attr(G,att_list):
    """
    Deletes Node Attributes from Graph
    """    
    for n, d in G.nodes(data=True):
        for att in att_list:
            d.pop(att, None)
            
            
def cluster_subgraph_by_year_louvain(Graph,option='accumulate',
                             connected='yes',retain_clus='yes',
                             res_louv=1,wei='weight'):
    """
    option='accumulate' will accumulate the nodes and edges of the graph year on year
    option='separate' will only keep the nodes year on year, edges from previous years will not be retained
    connected='yes' will only use the largest connected component for each year
    connected='no' will use all available nodes for each year
    retain_clus='yes' will initialize the louvain calculation such that the previous year's cluster is used to initialize this year's cluster
    retain_clus='no' will use a random initialization for the louvain calculation
    res_louv is used to set the resolution parameter for the louvain clustering calculation
    wei is used for the edge weight in the louvain clustering calculation
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
            if connected=='no':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
            elif connected=='yes':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
                 H[i]=max(nx.connected_component_subgraphs(H[i]), key=len)
            #H[i].add_nodes_from(list(list_dict_node_year[i].keys()))
            #H[i].add_edges_from(list(list_dict_edge_year[i].keys()))
            else:
                raise Exception("wrong keyword for connected. use yes or no only")
            print('Year:',str(i+min_year),'--',H[i].number_of_nodes(),'nodes --',H[i].number_of_edges(),'edges')
    elif option=='separate':
        for i in range(n_year):
            if connected=='no':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
            elif connected=='yes':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
                 H[i]=max(nx.connected_component_subgraphs(H[i]), key=len)
            #H[i].add_nodes_from(list(list_dict_node_year[i].keys()))
            #H[i].add_edges_from(list(list_dict_edge_year[i].keys()))
            else:
                raise Exception("wrong keyword for connected. use yes or no only")            
                      
            print('Year:',str(i+min_year),'--',H[i].number_of_nodes(),'nodes --',H[i].number_of_edges(),'edges')
    #  implement clustering
    
    c_louv=[{} for i in range(n_year)]
    J=Graph
    print("------------Louvain------------------")
              
    for i in range(n_year):
        start = time.time()
        if retain_clus=="yes" and i>0 :
            last_year_clus=c_louv[i-1]
            others_no=max(c_louv[i-1].values())+1
            others=list(set(nx.nodes(H[i]))-c_louv[i-1].keys())
            l=0
            for k in others:
                last_year_clus[k]=others_no+l
                l+=1
            if H[i].number_of_edges()>0:
                c_louv[i]=community.best_partition(H[i],resolution=res_louv,partition=c_louv[i-1],weight=wei)
        else:
            if H[i].number_of_edges()>0:
                c_louv[i]=community.best_partition(H[i],resolution=res_louv,weight=wei)
            
          
        nx.set_node_attributes(J,c_louv[i],'Louvain cluster'+str(i+min_year))
        stop = time.time()
        if H[i].number_of_edges()>0:
            print('Year:',str(i+min_year),'--',round(stop-start,2),'seconds --',str(len(set(c_louv[i].values()))),'clusters --',round(community.modularity(c_louv[i],H[i]),2),'modularity')
      
    del H, c_louv, node_yr, edge_yr, n_year, min_year, list_dict_edge_year, list_dict_node_year
    gc.collect()
    return J
    #return H

# define function to assign cluster to a networkx graph. The output from the greedy modularity function is a list, a dictionary has to be created for networkx
def set_cluster(cluster_list_set,G,attribute_name):
    dict_G=dict() 
    for i in range(len(cluster_list_set)):
        for j in cluster_list_set[i]:
            dict_G[j]=i
            
    nx.set_node_attributes(G,dict_G,name=attribute_name)

    
def cluster_subgraph_by_year_cnm(Graph,option='accumulate',
                             connected='yes',wei='weight'):
    """
    option='accumulate' will accumulate the nodes and edges of the graph year on year
    option='separate' will only keep the nodes year on year, edges from previous years will not be retained
    connected='yes' will only use the largest connected component for each year
    connected='no' will use all available nodes for each year
    retain_clus='yes' will initialize the louvain calculation such that the previous year's cluster is used to initialize this year's cluster
    retain_clus='no' will use a random initialization for the louvain calculation
    res_louv is used to set the resolution parameter for the louvain clustering calculation
    wei is used for the edge weight in the louvain clustering calculation
    """
    from networkx.algorithms.community import greedy_modularity_communities # This is to run CNM , remove if not neededs

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
            if connected=='no':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
            elif connected=='yes':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
                 H[i]=max(nx.connected_component_subgraphs(H[i]), key=len)
            #H[i].add_nodes_from(list(list_dict_node_year[i].keys()))
            #H[i].add_edges_from(list(list_dict_edge_year[i].keys()))
            else:
                raise Exception("wrong keyword for connected. use yes or no only")
            print('Year:',str(i+min_year),'--',H[i].number_of_nodes(),'nodes --',H[i].number_of_edges(),'edges')
    elif option=='separate':
        for i in range(n_year):
            if connected=='no':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
            elif connected=='yes':
                 H[i]=nx.subgraph(Graph,list(list_dict_node_year[i].keys()))
                 H[i]=max(nx.connected_component_subgraphs(H[i]), key=len)
            #H[i].add_nodes_from(list(list_dict_node_year[i].keys()))
            #H[i].add_edges_from(list(list_dict_edge_year[i].keys()))
            else:
                raise Exception("wrong keyword for connected. use yes or no only")            
                      
            print('Year:',str(i+min_year),'--',H[i].number_of_nodes(),'nodes --',H[i].number_of_edges(),'edges')
    
    #  implement clustering
    J=Graph

    print("------------Clauset-Newman-Moore------------------")
    c_cnm=[[]for i in range(n_year)]
    for i in range(n_year):
        start = time.time()
        c_cnm[i]=list(greedy_modularity_communities(H[i]))
        set_cluster(c_cnm[i],H[i],'CNM cluster') 
        set_cluster(c_cnm[i],J,'CNM cluster'+str(i+min_year))
        stop = time.time()
        print('Year:',str(i+min_year),'--',round(stop-start,2),'seconds --',str(len(set(c_cnm[i]))),'clusters')        
    del H, c_cnm, node_yr, edge_yr, n_year, min_year, list_dict_edge_year, list_dict_node_year
    gc.collect()
    return J
    #return H
    

def cluster_hist(G,attr_list,log=False):
    fig, axs = plt.subplots(figsize=(12,30), ncols=4, nrows=7)

    for i in range(len(attr_list)):
        d=list(nx.get_node_attributes(G,attr_list[i]).values())
        sns.distplot(d, kde=False, hist=True, ax=axs[i//4, i%4],
                     bins=list(range(min(d),max(d)+2)),
                     hist_kws={'log':log}).set_title(attr_list[i])
        #sns.distplot(d, kde=False, hist=True, ax=axs[i//4, i%4],
        #             bins=list(range(min(d),min(100,max(d)+2))),
        #             hist_kws={'log':log}).set_title(attr_list[i])
        
def get_node_attr(G,value,attr):
    """
    Returns nodes from a dictionary with attribute matching value
    """
    out=[x for x,y in G.nodes(data=True) if y.get(attr,"No")==value]
    return out     
    
def get_edge_attr(G,value,attr):
    """
    Returns edges from a dictionary with attribute matching value
    """
    out=[(x,y) for x,y,z in G.edges(data=True) if z.get(attr,"No")==value]
    return out   
    
def get_cluster_size(G,cluster_no,attr):
    """
    Return the size of the cluster with cluster_no
    """
    return len(get_node_attr(G,cluster_no,attr))
    
def get_papercluster(G,paper_id,attr_list):
    """
    quick diagnostic tool to check which cluster the paper is and size of cluster
    """
    cluster=[nx.get_node_attributes(G,i).get(paper_id,None) for i in attr_list]
    cluster_size=[get_cluster_size(G,cluster[i],attr_list[i]) for i in range(len(cluster))]
    return cluster,cluster_size

         

            
            
##############################################################
# CALCULATION OF FUTURERANK: alpha*Pagerank + gamma*R_Time
##############################################################
def futurerank_CT_scipy(G, alpha=0.2, gamma=0.8, personalization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
    """Return the FutureRank CT of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Parameter for FutureRank CT, default=0.2
      
    gamma : float, optional
      Parameter for FutureRank CT, default=0.8

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specfiied, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.
    Returns
    -------
    FutureRank CT : dictionary
       Dictionary of nodes with FutureRank as value

    """
    import scipy.sparse

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            + gamma * p + \
            (1 - alpha-gamma) * scipy.repeat(1.0 / N, N) 
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


##############################################################
# CALCULATION OF FUTURERANK: alpha*Pagerank + gamma*R_Time + delta*clusterization
##############################################################
def futureclusterank_CT_scipy(G, alpha=0.2, gamma=0.7, delta=0.1, personalization=None, clusterization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
    """Return the FutureRank CT of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Parameter for FutureClusterRank CT, default=0.2
      
    gamma : float, optional
      Parameter for FutureClusterRank CT, default=0.7
      
    delta : float, optional
      Parameter for FutureClusterRank CT, default=0.1      

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specfiied, a nodes personalization value will be zero.
      By default, a uniform distribution is used.
     
    clusterization: dict, optional
      The "clusterization vector" consisting of a dictionary with a
      key some subset of graph nodes and clusterization value each of those.
      At least one clusterization value must be non-zero.
      If not specfiied, a nodes clusterization value will be zero.
      By default, a uniform distribution is used.    

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.
    Returns
    -------
    FutureRank CT : dictionary
       Dictionary of nodes with FutureRank as value

    """
    import scipy.sparse

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0.5) for n in nodelist], dtype=float)
        p = p / p.sum()
        
    # Clusterization vector
    if clusterization is None:
        q = scipy.repeat(1.0 / N, N)
    else:
        q = scipy.array([clusterization.get(n, 0.33) for n in nodelist], dtype=float)
        q = q / q.sum()        


    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            + gamma * p + delta * q + \
            (1 - alpha-gamma-delta) * scipy.repeat(1.0 / N, N) 
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


###################################################################
# EXECUTION OF FUTURERANK, AND COLLECTION OF EVALUATION RESULTS
###################################################################
def getgraphforpred(G, uptill_year):
    rel_nodes = []
    #train_node_sample = list(G.nodes(data=True))[0] # This lets you see a sample of the data.
    for n in G.nodes(data=True):
        if n[1]['Year'] <= uptill_year:
            rel_nodes.append(n[0])
    G_forpred = G.subgraph(rel_nodes)
    return G_forpred

def getactualgraph_next3years(G, uptill_year):
    startfrom = uptill_year + 1
    enduptill = uptill_year + 3
    relevantnodes = []
    for e in G.edges(data=True):
        if (e[2]['Year'] >= startfrom) & (e[2]['Year'] <= enduptill):
            relevantnodes.append(e[0])
            relevantnodes.append(e[1])
    relevantnodes = list(set(relevantnodes))
    G_actual_next3years = G.subgraph(relevantnodes)
    return G_actual_next3years
            

def rankdict(resultsdict):
    ranked = {}
    for i,paper in enumerate(resultsdict):
        ranked['rank_'+"%06d" % (i+1)] = {}
        ranked['rank_'+"%06d" % (i+1)]['node'] = paper
        ranked['rank_'+"%06d" % (i+1)]['rank'] = i+1
    return ranked 


def mse_ranking(prediction, actual):
    act = sorted(actual.items(),key=operator.itemgetter(1),reverse=True)[:1000]
    pred = sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)[:1000]
    actrank = {a[0]:i  for i, a in enumerate(act)}  
    predrank = {a[0]:i  for i, a in enumerate(pred)} 
    mse_rank = 0
    for paper in actrank:
        if paper in predrank:
            diff = (actrank[paper] - predrank[paper])**2
            mse_rank += diff       
    return mse_rank


def allparams(G, uptill_year, filename):    
    with open(filename,'w') as out:
        out.write('')
    r_list = []
    a_list = []
    g_list = []
    d_list = []
    mse1_list = []
    mse2_list = []
    mse3_list = []
    mse4_list = []
    rhos = [0.6, 0.7, 0.5, 0.8, 0.4, 0.9, 0.3, 1, 0.2, 0.1]
    alphas = [0.85, 0.8, 0.9, 0.6, 0.3, 0.1]
    gammas = deltas = [0.9, 0.7, 0.5, 0.3, 0.1]
    allresults = pd.DataFrame()
    t1_start = time.time() 
    countdown = 0
    for alpha in alphas:
        for gamma in gammas:
            for delta in deltas:
                if (alpha+gamma+delta <=1):
                    countdown += 1
    rd = 0
    msearchive = []
    for alpha in alphas:
        for gamma in gammas:
            for delta in deltas:
                for rho in rhos:
                    t2_start = time.time() 
                    countdown -= 1
                    rd += 1
                    with open(filename,'a') as out:
                        out.write('round '+str(rd)+':\r\n')
                    if (alpha+gamma+delta <=1):
                        #try:
                        r_list.append(rho)
                        a_list.append(alpha)                        
                        g_list.append(gamma)
                        d_list.append(delta)
                        with open(filename,'a') as out:
                            out.write('rhos tried:\r\n' + str(r_list) +'\r\n')
                            out.write('alphas tried:\r\n' + str(a_list) +'\r\n')
                            out.write('gammas tried:\r\n' + str(g_list) +'\r\n')
                            out.write('deltas tried:\r\n' + str(d_list) +'\r\n')                        
                        if os.path.exists('alpha'+ "%.3f"%alpha +'_gamma'+ "%.3f"%gamma +'_delta'+ "%.3f"%delta +'__Dictionaries'):
                            pass
                        else:
                            os.mkdir('alpha'+ "%.3f"%alpha +'_gamma'+ "%.3f"%gamma +'_delta'+ "%.3f"%delta +'__Dictionaries')
                        folder = 'alpha'+ "%.3f"%alpha +'_gamma'+ "%.3f"%gamma +'_delta'+ "%.3f"%delta +'__Dictionaries'
                        
                        # Create the list of column names for the datafrane where the results will be stored:        
                        columnlabels = ['year', #0
                                        'rho','alpha','gamma', 'delta',#1-3
                                        #'(uptill) FutureRank MSE', #4
                                        '(uptill) FutureClusterRank_1 MSE', #5
                                        '(uptill) FutureClusterRank_2 MSE', #6
                                        #'(uptill) PlainPageRank MSE'
                                         ]  #7    
                    
                        querytime = uptill_year + 3       
                        
                        Node_year_dict = nx.get_node_attributes(G,'Year')
                        Clus_size_dict = nx.get_node_attributes(G,'ClusterSize'+str(uptill_year))
                        print('____checkpoint_01')
                        
                        # Create ditionaries for paper ages and cluster size quotient at query time:
                        R_time = {k:np.exp(-rho*(querytime - Node_year_dict[k])) for k in Node_year_dict.keys()}  
                        R_size_1 = {k:1/(1+Clus_size_dict[k]) for k in Clus_size_dict.keys()}
                        R_size_2 = {k:1/np.log(1+Clus_size_dict[k]) for k in Clus_size_dict.keys()}
                        
                        # Get input graphs:
                        #graphforprediction_uptill = getgraphforpred(G, 2006, 2013)
                        graphforprediction_uptill = getgraphforpred(G, uptill_year)
                        graphactual_next3years = getactualgraph_next3years(G, uptill_year)
                        print('____checkpoint_02')
                        
                        # Run the 4 algorithms tp get scores:   
                        #Rp_CT_uptill =     futurerank_CT_scipy(graphforprediction_uptill,     personalization = R_time, alpha = alpha, gamma = gamma) 
                        Rp_clusV1_uptill =     futureclusterank_CT_scipy(graphforprediction_uptill, clusterization = R_size_1, personalization = R_time,  alpha = alpha, gamma = gamma, delta = delta)
                        Rp_clusV2_uptill =     futureclusterank_CT_scipy(graphforprediction_uptill, clusterization = R_size_2, personalization = R_time,  alpha = alpha, gamma = gamma, delta = delta)
                        #Rp_control_uptill = nx.pagerank(graphforprediction_uptill, alpha = alpha)
                        print('____checkpoint_03')
                    
                        Node_title_dict = nx.get_node_attributes(G,'title')
                        Node_year_dict = nx.get_node_attributes(G, 'Year')
                        print('____checkpoint_04')
                        
                        Rp_actual_next3years = nx.pagerank(graphactual_next3years)    
                    
                        
                        #mse1 = mse_futurerank_till = mse_ranking(Rp_CT_uptill, Rp_actual_next3years)
                        mse2 = mse_futureclusterrankv1_till   = mse_ranking(Rp_clusV1_uptill, Rp_actual_next3years)
                        mse3 = mse_futureclusterrankv2_till   = mse_ranking(Rp_clusV2_uptill, Rp_actual_next3years)
                        #mse4 = mse_control_till = mse_ranking(Rp_control_uptill, Rp_actual_next3years)
                        #msegroup = [mse1, mse2, mse3, mse4]
                        #mseaverage = sum(msegroup)/len(msegroup)
                        #mse1_list.append(mse1)
                        mse2_list.append(mse2)
                        mse3_list.append(mse3)
                        #mse4_list.append(mse4)                       
                        with open(filename,'a') as out:
                            #out.write('MSE1:\r\n' + str(mse1_list) +'\r\n')
                            out.write('MSE2:\r\n' + str(mse2_list) +'\r\n')
                            out.write('MSE3:\r\n' + str(mse3_list) +'\r\n')
                            #out.write('MSE4:\r\n' + str(mse4_list) +'\r\n\n\n')
                        print('____checkpoint_05') 
                    
                        #rankedA = json.dumps(rankdict(Rp_CT_uptill))
                        rankedB = json.dumps(rankdict(Rp_clusV1_uptill))
                        rankedC = json.dumps(rankdict(Rp_clusV2_uptill))
                        #rankedD = json.dumps(rankdict(Rp_control_uptill))
                        
                        rankedI = json.dumps(rankdict(Rp_actual_next3years))
                        print('____checkpoint_06')
                          
                        #dA = json.dumps(Rp_CT_uptill)
                        #rA = rankedA         #12-13
                        dB = json.dumps(Rp_clusV1_uptill)
                        rB = rankedB         #14-15
                        dC = json.dumps(Rp_clusV2_uptill)
                        rC = rankedC         #16-17
                        #dD = json.dumps(Rp_control_uptill)
                        #rD = rankedD         #18-19
                        dI = json.dumps(Rp_actual_next3years)
                        rI = rankedI         #28-29
                        print('____checkpoint_07')
                        
                        #with open(folder+'/FutureRank_paperIDdict.txt','w') as f:
                        #    f.write(dA)
                        #with open(folder+'/FutureRank_Rankingdict.txt','w') as f:
                        #    f.write(rA)   
                        with open(folder+'/FutureClusterRankv1_paperIDdict.txt','w') as f:
                            f.write(dB)
                        with open(folder+'/FutureClusterRankv1_Rankingdict.txt','w') as f:
                            f.write(rB)  
                        with open(folder+'/FutureClusterRankv2_paperIDdict.txt','w') as f:
                            f.write(dC)
                        with open(folder+'/FutureClusterRankv2_Rankingdict.txt','w') as f:
                            f.write(rC)  
                        #with open(folder+'/PlainPageRank_paperIDdict.txt','w') as f:
                        #    f.write(dD)
                        #with open(folder+'/PlainPageRank_Rankingdict.txt','w') as f:
                        #    f.write(rD) 
                        with open(folder+'/ActualInNext2Years_paperIDdict.txt','w') as f:
                            f.write(dI)
                        with open(folder+'/ActualInNext2Years_Rankingdict.txt','w') as f:
                            f.write(rI)                         
                        print('____checkpoint_08')
                        
                        data = [(uptill_year, rho, alpha, gamma, delta, #0-3
                                #mse_futurerank_till, #4
                                mse_futureclusterrankv1_till,  #5
                                mse_futureclusterrankv2_till,  #6
                                #mse_control_till
                                )]#,              #7                           
                        param_results = pd.DataFrame.from_records(data, columns = columnlabels)
                        print('____checkpoint_09')
                        
                        allresults = allresults.append(param_results, ignore_index=True) 
                        t1 = time.time() - t1_start
                        t2 = time.time() - t2_start
                        print('Time for this round (hh:mm:ss.ms) {}'.format(t2))
                        print('Total Time elapsed  (hh:mm:ss.ms) {}'.format(t1))
                        est_time = t1/rd*countdown
                        print('Est Time Remaining  (hh:mm:ss.ms) {}'.format(est_time))
                        print('Countdown = '+ str(countdown) + ' left to go.')
                        #except:
                        #    winsound.Beep(4390, 300)
                        #    winsound.Beep(4390, 300)
                        #    winsound.Beep(4390, 300)
                        #    mse2_list.append('fail')
                        #    mse3_list.append('fail')                       
                        #    with open(filename,'a') as out:
                        #        #out.write('MSE1:\r\n' + str(mse1_list) +'\r\n')
                        #        out.write('MSE2:\r\n' + str(mse2_list) +'\r\n')
    return allresults

