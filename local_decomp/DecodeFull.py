
import create_graph
from lattice_extractor import *
import local_decomp 
import time
from svector import Vector
from util import *

DEBUG = False
TIMING = True
BEST = False

graph_file = "/tmp/graph"
word_file = "/tmp/words"

def assert_close(a,b):
  assert abs(a - b) < 1e-5, str(a) + " " + str(b)    

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
    self.forest = forest
    self.lm_file = lm_file
    self.lm = lm
    self.setup_lm_decoder()

    # augment features in forest with edge names for lagrangians
    for node in self.forest:
      for edge in node.edges:
        edge.fvector["FOR:" + str(edge.position_id)] = 1.0
        for graph_id in self.edge_map.get(edge.position_id, []):
          edge.fvector["1UNI:" + str(graph_id.id)] = 1.0
          edge.fvector["2UNI:" + str(graph_id.id)] = 1.0
                  
    self.lagrangians = Vector()

  def setup_lm_decoder(self):
    graph = NodeExtractor().extract(self.forest)
    graph.filter((lambda n: isinstance(n, NonTermNode)))  
    self.graph = graph
    self.word_map = create_graph.write_files(self.forest, graph, self.lm, reverse = True) #REVERSE!!!

    self.edge_map = graph.edge_map

    self.subproblem = local_decomp.PySubproblem(graph_file, word_file, self.lm_file)

    # map edge to lex nodes 
    self.lex_map = {}
    for edge_id in self.edge_map:
      self.lex_map[edge_id] = [node for node in self.edge_map[edge_id] if node.lex]
    #print self.lex_map
  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def delta_weights(self, updates,weights):

    start = time.time()      
    self.set_weights(weights)
    
    if TIMING:
      end = time.time()      
      print "python weight update", end - start

    # send the weights to the subsolver
    
    c_updates1 = []
    c_pos1 = []

    c_updates2 = []
    c_pos2 = []


    for feat in updates:
      p = int(feat.split(":")[1])
      
      if feat[0] == '1': 
        c_updates1.append(-updates[feat])
        c_pos1.append(p)
      elif feat[0] == '2':
        c_updates2.append(-updates[feat])
        c_pos2.append(p)
      else : 
        assert False

    if TIMING:
      end = time.time()      
      print "prep weight update", end - start
    
    if DEBUG:
      print "UPDATES ARE: "
      for f in updates:
        print f, updates[f]
    
    start = time.time()      
    self.subproblem.update_weights(c_pos1, c_updates1, len(c_pos1), 1)
    self.subproblem.update_weights(c_pos2, c_updates2, len(c_pos2), 0)
    if TIMING:
      end = time.time()
      print "c weight update", end - start 
    

  def compute_primal(self, fvector, trans):
     #print trans
     rescore = self.weights.dot(fvector)
     rescore += self.weights['lm'] * -self.lm.word_prob(trans)
     trans2 = ("<s> <s> " + trans  + " </s> </s>").split()
     
     if DEBUG:
       print "\n\nPRIMAL BIGRAMS\n\n"
       for i in range(len(trans2)-2):
         (a,b,c) = (trans2[i], trans2[i+1], trans2[i+2])
         print a, b, c, self.weights['lm'] * self.lm.word_prob_bystr(c, a + " " + b)
     
         #print trans2
     word_ind = [self.word_map[self.lm.word2index(t)] for t in trans2]
     #print word_ind
     if BEST:
       print "LM Score", self.subproblem.primal_score(word_ind)
       #print "PARSE Score", (fvector)
     return self.weights['lm'] *self.subproblem.primal_score(word_ind) + self.weights.dot(fvector)

  def debug_bigram(self, graph_node, forbigram, score):
    assert DEBUG
    graph_id = graph_node.id

    words = [strip_lex(graph_node.lex),strip_lex(self.graph.nodes[forbigram.w1].lex), strip_lex(self.graph.nodes[forbigram.w2].lex)]
    words.reverse()
    word_ind = [self.word_map[self.lm.word2index(t)] for t in words]
    print word_ind
    lm_score = self.weights['lm']* self.subproblem.primal_score(word_ind)
  

    print graph_node, self.graph.nodes[forbigram.w1], self.graph.nodes[forbigram.w2], score, lm_score
    
    lag = 0.0
    between1 = self.subproblem.get_best_nodes_between(graph_id,forbigram.w1, True)            
            #print "Should be: ", self.subproblem.get_best_bigram_weight(graph_id,forword)
    for b in between1:
      print "\t", b, -self.lagrangians["1UNI:"+str(b)] 
      lag += -self.lagrangians["1UNI:"+str(b)] 

    between2 = self.subproblem.get_best_nodes_between(forbigram.w1, forbigram.w2, False)
    for b in between2:
      print "\t", b, -self.lagrangians["2UNI:"+str(b)] 
      lag += -self.lagrangians["2UNI:"+str(b)] 
    assert_close(score - lag, lm_score)

  def decode(self):

    # Add in the lagrangians
    start = time.time()
    cur_weights = self.weights.__copy__()
    cur_weights += self.lagrangians
    if TIMING:
      end = time.time()
      print "copy time", end - start


    # first solve the subproblem    
    start = time.time()
    self.subproblem.solve()
    if TIMING:
      end = time.time()
      print "C time", end - start
    
    # now add the forward trigrams at each node
    start = time.time()
    for node in self.forest:
      for edge in node.edges:
        # remove previous
        #for f in edge.fvector:
        #
        #if f.startswith("FOR:" + str(edge.position_id)):
        #  del edge.fvector["FOR:" + str(edge.position_id))]

        feat = "FOR:" + str(edge.position_id)
        total_score = 0.0
        for graph_node in self.lex_map.get(edge.position_id, []):
          graph_id = graph_node.id
          # check the best forward
          (forbigram, score) = self.subproblem.get_best_trigram(graph_id)
          total_score += score
          
#print graph_node, forword, self.graph.nodes[forword], score
          #feature_name = "FOR:" + str(graph_id)+ ":"+str(forbigram.w1) + ":" + str(forbigram.w2) 
          #edge.fvector[feature_name] = 1.0
          #cur_weights[feature_name] = score

          if DEBUG:  
            assert forbigram.w1 != -1, str(self.graph.nodes[graph_id])
            assert forbigram.w2 != -1, str(self.graph.nodes[graph_id])
            self.debug_bigram(graph_node, forbigram, score)
        cur_weights[feat] = total_score
      #if DEBUG:  
      #  print edge, edge.fvector.dot(self.weights), edge.fvector.dot(self.lagrangians),  
    if TIMING:
      end = time.time()
      print "augment time", end - start
          

    start = time.time()
    (best, subtree, best_fv) = self.forest.bestparse(cur_weights, use_min=True)
    #print "BEST PARSE", best
    assert abs(best -best_fv.dot(cur_weights)) < 1e-4, str(best) + " " + str(best_fv.dot(cur_weights))
    #print subtree
    lagrangians_parse = 0.0
    lagrangians_other = 0.0

    
    if TIMING:
      end = time.time()
      print "parse time", end - start
    
    
    ret = Vector()
    #print "TREE DECODER \n\n"
    tri_pairs = []

    start = time.time()
    for feat in best_fv:
      if feat.startswith("FOR:"):
        edge_pos_id = int(feat.split(":")[-1])
        for graph_node in self.lex_map.get(edge_pos_id, []):
          graph_id = graph_node.id
          (forbigram, score) = self.subproblem.get_best_trigram(graph_id)          
          end_at = forbigram.w2
          mid_at = forbigram.w1
          start_from = graph_node.id

          between1 = self.subproblem.get_best_nodes_between(start_from,mid_at, True)
          tri_pairs.append((self.graph.nodes[start_from].lex, self.graph.nodes[mid_at].lex, self.graph.nodes[end_at].lex))
          for node in between1:
            ret["1UNI:"+str(node)] -= best_fv[feat]
            if DEBUG:
              print "LM", self.graph.nodes[node]

          between2 = self.subproblem.get_best_nodes_between(mid_at,end_at, False)
          #bi_pairs.append((self.graph.nodes[mid_at].lex, self.graph.nodes[end_at].lex))
          for node in between2:
            ret["2UNI:"+str(node)] -= best_fv[feat]
            if DEBUG:
              print "LM", self.graph.nodes[node]

      elif feat.startswith("1UNI") or feat.startswith("2UNI") :
        p = int(feat.split(":")[1])
        #if p == 0: continue
        #print str(self.words[p]), self.lagrangians["UNI" + str(p)], p
        # Trigram means there are two lmulti
        ret[feat] += best_fv[feat]
        if DEBUG:
          print "REAL", self.graph.nodes[p]
    if TIMING:
      end = time.time()
      print "extract time", end - start
      
    # BOUNDARY CONDITIONS
    bounds = [(0,1), (self.graph.size()-1,self.graph.size()-2) ]
    bounds.reverse()

    # END BOUNDARY
    # add in the last node, and second to last (trigram)
    # over counted at k
    feat = "2UNI:"+str(bounds[1][0])
    ret[feat] += 1 
    
    best += self.lagrangians[feat]  

    # over counted at k and j
    feat = "1UNI:"+str(bounds[1][1])
    ret[feat] += 1 
    best += self.lagrangians[feat]

    feat = "2UNI:"+str(bounds[1][1])
    ret[feat] += 1 
    best += self.lagrangians[feat]  

    # START BOUNDARY
    # first word <s>
    (forbigram,  score) = self.subproblem.get_best_trigram(bounds[0][0])    
    best += score

    tri_pairs.append((self.graph.nodes[bounds[0][0]].lex, self.graph.nodes[forbigram.w1].lex,self.graph.nodes[forbigram.w2].lex))
    #bi_pairs.append((self.graph.nodes[forbigram.w1].lex, self.graph.nodes[forbigram.w2].lex))
    
    #between1 = self.subproblem.get_best_nodes_between(0,forbigram.w1)
    #for b in between1:
      #ret["UNI:"+str(b)] -= 1 

    #print "Boundaries \n\n"
    
    between2 = self.subproblem.get_best_nodes_between(forbigram.w1, forbigram.w2, False)
    for b in between2:
      ret["2UNI:"+str(b)] -= 1 
      
      if DEBUG:
        print "\t", b, -self.lagrangians["2UNI:"+str(b)] 
        print "LM", self.graph.nodes[b]
      
    # second word <s>


    (forbigram,  score) = self.subproblem.get_best_trigram(bounds[0][1])
    best += score

    tri_pairs.append((self.graph.nodes[bounds[0][1]].lex, 
                      self.graph.nodes[forbigram.w1].lex,
                      self.graph.nodes[forbigram.w2].lex))
    
    between1 = self.subproblem.get_best_nodes_between(bounds[0][1], forbigram.w1, True)
    if DEBUG: self.debug_bigram(self.graph.nodes[1], forbigram, score)

    for b in between1:
      ret["1UNI:"+str(b)] -= 1

      if DEBUG:
        print "\t", b, -self.lagrangians["1UNI:"+str(b)]
        print "LM", self.graph.nodes[b]

    between2 = self.subproblem.get_best_nodes_between(forbigram.w1, forbigram.w2, False)
    for b in between2:
      ret["2UNI:"+str(b)] -= 1

      if DEBUG:
        print "\t", b, -self.lagrangians["2UNI:"+str(b)]
        print "LM", self.graph.nodes[b]


    #print "END TREE DECODER \n\n"
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

    primal = self.compute_primal(best_fv, subtree)    
    
    if BEST:
      print "Best score", best
      #print "RM Weights", best - ret.dot(self.lagrangians)
      #accounting = 0.0
      print subtree
    if DEBUG:
      for (a,b,c) in tri_pairs:
        print a, b, c, self.weights["lm"]* self.lm.word_prob_bystr(strip_lex(c), strip_lex(a) + " " + strip_lex(b))
        
    


    if BEST:
      print "Primal", primal
      print "final best", best
    return ( ret, best, primal)


