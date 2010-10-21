
import svector  

#import numpy as np
#cimport numpy as np

cdef extern from "math.h":
  double log(double x)
  double exp(double x)
  double fmax(double a, double b)
  double fmin(double a, double b)
DEF INF = 1e90
DEF MAX_NODES = 100000
DEF MAX_EDGES = 1000000

#cdef np.ndarray[np.float64_t, ndim=1] node_weights = np.zeros(forest.nodelen, dtype=np.float64) 
#cdef np.ndarray[np.float64_t, ndim=1] edge_weights = np.zeros(forest.edgelen, dtype=np.float64) 
cdef double node_weights[MAX_NODES]
cdef double edge_weights[MAX_EDGES]
cdef double edge_merit[MAX_EDGES]
cdef double node_merit[MAX_NODES]
cdef double outside_nodes[MAX_NODES]


cdef inside_sum(forest, weights):
  "Assume nodes are in topological order"
  cdef double node_score, edge_score, score, m, M
  cdef unsigned int edge_pos, sub_pos, node_pos
  for node in forest:
    #assert not node_weights.has_key(node)
    node_pos = node.position_id
    node_score = node.fvector.dot(weights) # log space
    score = -INF 
    for edge in node.edges:
      edge_score = edge.fvector.dot(weights)  # log space
      edge_pos = edge.position_id
      #assert not edge_weights.has_key(edge)
      edge_weights[edge_pos] = 0.0 # log space 
      for sub in edge.subs:
        sub_pos = sub.position_id
        #assert node_weights.has_key(sub), "Not in topo order"
        edge_weights[edge_pos] += node_weights[sub_pos] # log times
      
      edge_weights[edge_pos] += edge_score + node_score # log times

      M = fmax(score, edge_weights[edge_pos])
      m = fmin(score, edge_weights[edge_pos])
      score = M + log(1.0 + exp(m - M)) # log-sum score += edge_weights
      
    node_weights[node_pos] = score 
  #return (node_weights, edge_weights)
  
cdef outside_sum(forest, weights):


  cdef double score, edge_m, m, M
  cdef unsigned int edge_pos, sub_pos, node_pos, i
  
  for i in range(forest.nodelen):
    outside_nodes[i] = -INF

  outside_nodes[forest.root.position_id] = 0.0 # log 1.0
  for node in forest.reverse():   ## top-down
    # merit - sum of all derivation through this node
    node_pos = node.position_id
    node_merit[node_pos] = node_weights[node_pos] + outside_nodes[node_pos] # log times
    
    for edge in node.edges:
      edge_pos = edge.position_id
      edge_m = outside_nodes[node_pos] + edge_weights[edge_pos] # log times
      edge_merit[edge_pos] = edge_m 
      
      for sub in edge.subs:
        sub_pos = sub.position_id
        score = edge_m - node_weights[sub_pos] # log-div
        #if outside_nodes.has_key(sub.position_id):

        # log plus
        M = fmax(score, outside_nodes[sub_pos])
        m = fmin(score, outside_nodes[sub_pos])
        outside_nodes[sub_pos] = M + log(1.0 + exp(m - M)) # log-sum outside_nodes[sub.position_id] = outside_nodes[sub.position_id] + score
        #else:
        #  outside_nodes[sub.position_id] = score
        
  #return outside_nodes, node_merit, edge_merit

def collect_marginals(forest, weights):
  cdef double partition

  inside_sum(forest, weights)
  outside_sum(forest, weights)

  partition = node_merit[forest.root.position_id]

  marginals = svector.Vector()

  for node in forest:
    #print exp(node_weights[node.position_id]), exp(outside_nodes[node.position_id]), exp(node_merit[node.position_id] - partition), node
    node_pos = node.position_id
    node_fvector = node.fvector
    marginals +=  node_fvector * exp(node_merit[node_pos] - partition)  
    assert exp(node_merit[node_pos] - partition)  <= 1.01
  #print edges
  for edge in forest.edgeorder:
    #print exp(edge_weights[edge.position_id]), exp(edge_merit[edge.position_id] - partition), edge.fvector["Basic/text-length"], edge
    #if edge_prob > cutoff:
    edge_pos = edge.position_id
    edge_fvector = edge.fvector
    assert exp(edge_merit[edge_pos] - partition) <= 1.01
    marginals +=  edge_fvector * exp(edge_merit[edge_pos] - partition)

  return marginals, partition
