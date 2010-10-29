
import create_graph
from lattice_extractor import *
import local_decomp 
import time
from svector import Vector

DEBUG = True

graph_file = "/tmp/graph"
word_file = "/tmp/words"
class TreeDecoder(object):
  """
  Enforce that the parse is a valid tree
  """  
  def __init__(self, forest, weights, lm, lm_file):
    """
    forest - the forst to parse for this instance
    weights - weight vector to use
    
    """

    self.weights = weights
    self.forest = forest.copy()
    self.lm_file = lm_file
    self.lm = lm
    self.setup_lm_decoder()

    # augment features in forest with edge names for lagrangians
    for node in self.forest:
      for edge in node.edges:
        for graph_id in self.edge_map.get(edge.position_id, []):
          edge.fvector["UNI:" + str(graph_id.id)] = 1.0
                  
    self.lagrangians = Vector()

  def setup_lm_decoder(self):
    graph = NodeExtractor().extract(self.forest)
    graph.filter((lambda n: isinstance(n, NonTermNode)))  
    self.graph = graph
    create_graph.write_files(self.forest, graph, self.lm)

    self.edge_map = graph.edge_map

    self.subproblem = local_decomp.PySubproblem(graph_file, word_file, self.lm_file)

    # map edge to lex nodes 
    self.lex_map = {}
    for edge_id in self.edge_map:
      self.lex_map[edge_id] = [node for node in self.edge_map[edge_id] if node.lex]

  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def delta_weights(self, updates,weights):

    start = time.time()      
    self.set_weights(weights)
    end = time.time()      
    print "python weight update", end - start

    # send the weights to the subsolver
    
    c_updates = []
    c_pos = []
    l =0
    for feat in updates:
      p = int(feat.split(":")[1])
      
      c_updates.append(-updates[feat])
      c_pos.append(p)
      l+=1
    end = time.time()      
    print "prep weight update", end - start
    
    if DEBUG:
      print "UPDATES ARE: "
      for f in updates:
        print f, updates[f]
    
    start = time.time()      
    self.subproblem.update_weights(c_pos, c_updates, l)
    end = time.time()
    print "c weight update", end - start, "len", l 
    

  def compute_primal(self, fvector, trans):
     rescore = self.weights.dot(fvector)
     rescore += self.weights['lm'] * -self.lm.word_prob(trans)
     return rescore

  def decode(self):


    # Add in the lagrangians
    start = time.time()
    cur_weights = self.weights.__copy__()
    cur_weights += self.lagrangians
    end = time.time()
    print "copy time", end - start
    # first solve the subproblem
    
    start = time.time()
    self.subproblem.solve()
    end = time.time()
    print "C time", end - start
    
    start = time.time()
    for node in self.forest:
      for edge in node.edges:
        # remove previous
        for f in edge.fvector:
          if f.startswith("FOR:"):
            del edge.fvector[f]

        for graph_node in self.lex_map.get(edge.position_id, []):
          graph_id = graph_node.id
          # check the best forward

          (forword, score) = self.subproblem.get_best_bigram(graph_id)
          assert forword != -1, str(self.graph.nodes[graph_id])
          #print graph_node, forword, self.graph.nodes[forword], score
          feature_name = "FOR:" + str(graph_id)+ ":"+str(forword) 
          edge.fvector[feature_name] = 1.0
          cur_weights[feature_name] = score

          if DEBUG:
            print feature_name
            print graph_node, self.graph.nodes[forword], score
            between = self.subproblem.get_best_nodes_between(graph_id,forword)
            print "Should be: ", self.subproblem.get_best_bigram_weight(graph_id,forword)
            for b in between:
              print "\t", b, -self.lagrangians["UNI:"+str(b)] 
        print edge, edge.fvector.dot(self.weights), edge.fvector.dot(self.lagrangians),  
    end = time.time()
    print "augment time", end - start
          

    start = time.time()
    (best, subtree, best_fv) = self.forest.bestparse(cur_weights, use_min=True)
    end = time.time()
    print "parse time", end - start
    
    
    ret = Vector()
    print "TREE DECODER \n\n"
    bi_pairs = []

    start = time.time()
    for feat in best_fv:
      if feat.startswith("FOR:"):
        # really need to add every intervening node
        
        end_at = int(feat.split(":")[-1])
        start_from = int(feat.split(":")[-2])

        #print str(self.words[p]), self.lagrangians["UNI" + str(p)], p
        # TODO add all nodes
        between = self.subproblem.get_best_nodes_between(start_from,end_at)
        bi_pairs.append((self.graph.nodes[start_from].lex,self.graph.nodes[end_at].lex))

        #print "START", self.graph.nodes[start_from]
        #print "END", self.graph.nodes[end_at]
        for node in between:
          ret["UNI:"+str(node)] -= best_fv[feat]
          if DEBUG:
            print "LM", self.graph.nodes[node]
          #print "BETWEEN", self.graph.nodes[node]
      elif feat.startswith("UNI"):
        p = int(feat.split(":")[1])
        #if p == 0: continue
        #print str(self.words[p]), self.lagrangians["UNI" + str(p)], p
        ret[feat] += best_fv[feat]
        if DEBUG:
          print "REAL",self.graph.nodes[p]
    end = time.time()
    print "extract time", end - start
      
        
    # add in the last node
    ret["UNI:"+str(self.graph.size()-1)] += 1 

    (forword,  score) = self.subproblem.get_best_bigram(0)    
    best += score
    between = self.subproblem.get_best_nodes_between(0,forword)
    bi_pairs.append((self.graph.nodes[0].lex,self.graph.nodes[forword].lex))
    for b in between:
      ret["UNI:"+str(b)] -= 1 
      

    print "END TREE DECODER \n\n"
    #path = self.extract_path(self.forest.root)
    #print subtree
    #print [ self.s_table.Find(p) for p in path]
    #print path
    if DEBUG:
      for f in ret:
        if ret[f] <> 0.0:
          print f, ret[f]
    for f in ret:
      if ret[f] == 0.0:
        del ret[f]

    print "Best score", best
    print subtree
    for (a,b) in bi_pairs:
      print a, b

    primal = self.compute_primal(best_fv, subtree)
    print "Primal", primal
    return ( ret, best, primal)


