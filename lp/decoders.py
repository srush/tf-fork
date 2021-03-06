from svector import Vector
from forest import Forest
import fsa
import openfst
import time
import tree_extractor
from itertools import *
import math, sys
from util import *

def uninf(u):
  if math.isinf(u): return 1e10
  else: return u

class TreeDecoder(object):
  """
  Enforce that the parse is a valid tree
  """  
  def __init__(self, forest, weights, word_map, words, lit_words, s_table):    
    self.weights = weights
    self.forest = forest.copy()
    self.s_table = s_table
    self.lit_words = lit_words
    self.words = words

    # word_map from edge.position_id to word outputs
    self.word_map = word_map
    
    # augment features in forest with edge names for lagrangians
    for node in self.forest:
      for edge in node.edges:
        for sym in edge.rule.rhs:
          if is_lex(sym):
            edge.fvector["UNI" + str(self.lit_words[sym].id)] += 1.0

        for id in self.word_map.get(edge.position_id, []):
          edge.fvector["UNI" + str(id)] = 1.0
                  
    self.lagrangians = {"alpha" : Vector(), "beta" : Vector()}


  def get_node_index(self, node):
    return self.node_map[node.position_id]

  def bad_approx(self):
    pass
  
  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def delta_weights(self, updates,weights):
    self.set_weights(weights)

  def get_uni_feats(self, vector):
    feats = []
    for feat in vector:
      if feat.startswith("UNI"):
        feats.append(int(feat[3:]))
    return feats
  
  def extract_path(self, node):
    print node
    full_path = []
    
    # node_output = self.get_uni_feats(node.fvector)
#     for p in node_output:
#       if self.words[p][1] == "DOWN":
#         full_path.append(p)
    edge = node.bestedge
    word_uni = self.word_map.get(edge.position_id, [])
    def uni_sort(a,b):
      return cmp((int(self.words[a][0].split("+++")[2]), self.words[a][1]),
                 (int(self.words[b][0].split("+++")[2]), self.words[b][1])) 
    word_uni.sort(uni_sort)
    #print [self.words[a] for a in word_uni]
    pre_symbols = [s for s in word_uni if int(self.words[s][0].split("+++")[2]) < 0]
    for pre in pre_symbols:
      full_path.append(tree_extractor.SRC_NODE)
      full_path.append(pre)  
    
    symbols = [s for s in word_uni if int(self.words[s][0].split("+++")[2]) >= 0]
    lex = 0
    nts_num = 0
    for sym in edge.rule.rhs:
      if is_lex(sym):
        #pass
        full_path.append(tree_extractor.PRE_WORD)
        full_path.append(symbols[lex])
        lex +=1
        
        id = self.s_table.Find(super_strip(sym))
        if id == -1:
          id = self.s_table.Find("<unk>")
        print sym, super_strip(sym), id
        full_path.append(id)
        #lex +=1 
      else:
        print sym
        full_path.append(tree_extractor.SRC_NODE)
        full_path.append(symbols[lex])
        lex +=1
        
        full_path += self.extract_path(edge.subs[nts_num])
        nts_num += 1
        full_path.append(tree_extractor.SRC_NODE)
        full_path.append(symbols[lex])

        lex += 1
#     for p in node_output:
#       if self.words[p][1] == "UP":
#         full_path.append(p)

    #edge word (no longer have this)
    # full_path.append(word_uni[lex])
    
    return full_path
    
      
  def decode(self):
    cur_weights = self.weights.__copy__()
    cur_weights += self.lagrangians["alpha"] 
    (best, subtree, best_fv) = self.forest.bestparse(cur_weights, use_min=True)
    #print subtree
    
    ret = {"alpha":Vector(), "beta":Vector()}
    print "TREE DECODER \n\n"

    for feat in best_fv:
      
      # can only turn on each element once 
      if feat.startswith("UNI"):
        p = int(feat[3:])
        if p == 0: continue
        print str(self.words[p]), self.lagrangians["alpha"]["UNI" + str(p)], p
        if p > tree_extractor.PRE_WORD:
          ret["alpha"][feat] += best_fv[feat]
        #print self.words[p]


    print "END TREE DECODER \n\n"
    path = self.extract_path(self.forest.root)
    print subtree
    print [ self.s_table.Find(p) for p in path]
    #print path
    return (best, ret, (subtree, best_fv, path)) 


  def oracle(self, (subtree, fv, _)):
    #print_weights(fv)
    pure = self.weights.dot(fv)
    with_lagrangians = pure + self.lagrangians["alpha"].dot(fv) + self.lagrangians["beta"].dot(fv)
    return (pure, with_lagrangians)



class LMDecoderFST(object):
  def __init__(self, fst, just_tree_fst, just_lm_fst, just_count_fst, words, lm, lm_weight, non_neg, s_table, count):
    self.output_bound = (count -1) * 2
    self.pruner = None
    print "Copying"
    self.fst = fst
    self.s_table =s_table
    self.just_tree_fst = just_tree_fst
    self.just_lm_fst = just_lm_fst
    self.just_count_fst = just_count_fst
    self.non_neg = non_neg
    self.original_fst = fst
    
    #print "Original Size %s"% (self.fst.NumStates())
    #print fsa.count_arcs(self.fst)
    #openfst.Prune(self.fst, 5.0)
    #print fsa.count_arcs(self.fst)
    #openfst.Minimize(self.fst)
    #print "Post Prune Original Size %s"% (self.fst.NumStates())
    self.last_prune = 1e90
    self.lagrangians = {"alpha": Vector(), "beta": Vector()}
    #self.state_hash = fsa.collect_fst_hash(self.fst)

    self.words = words
    #openfst.ArcSortInput(self.fst)

    self.lm = lm
    self.lm_weight = lm_weight


    self.arcs = {}
    
    self.viterbi = True
    self.last_sym = s_table.AvailableKey() - tree_extractor.SRC_NODE + 1
    n = 0 

    print "eqs"
    self.temp_fst = self.fst

    self.first_best = None
    self.cur_best = 1e90
    #self.just_wefight_fst = None
    self.first_weights = True
    self.round = 0
    self.beamCard = 200
    self.beamRad  = 0.3
    print "Done"
    # print "FSAing"


#     self.just_weight_fst = openfst.StdVectorFst()


    
#     self.single_state = self.just_weight_fst.AddState()
#     self.just_weight_fst.SetStart(self.single_state)
#     self.just_weight_fst.SetFinal(self.single_state, 0.0)
#     self.just_weight_fst.SetInputSymbols(s_table)
#     self.just_weight_fst.SetOutputSymbols(s_table)
    
#     self.just_weight_fst.AddArc(self.single_state,
#                                 openfst.StdArc(fsa.RHO, fsa.RHO, 0.0,
#                                                self.single_state))
#     for w in words:
#       self.just_weight_fst.AddArc(self.single_state,
#                              openfst.StdArc(w, w, 0.0,
#                                             self.single_state))
#       self.arcs[w] = n
#       n += 1

#     openfst.ArcSortInput(self.just_weight_fst)

  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def bad_approx(self):
    self.beamCard += 200
    self.beamRad += 0.2
    print "Bad Approx", self.beamCard, self.beamRad

  def full_create_weight_fst(self, updates):
    for feat in updates["alpha"]:  
      pos = int(feat[3:])
      if pos == 0: continue
      if self.non_neg:
        assert self.lagrangians["alpha"][feat] >= 0.0
      fsa.update_weight(self.just_weight_fst, self.single_state, self.arcs[pos], self.lagrangians["alpha"][feat])
    
  def create_weight_fst(self, updates):
    self.weight_fst = openfst.StdVectorFst()
    self.weight_fst.SetInputSymbols(self.s_table)
    self.weight_fst.SetOutputSymbols(self.s_table)
    self.single_state = self.weight_fst.AddState()
    self.weight_fst.SetStart(self.single_state)
    self.weight_fst.SetFinal(self.single_state, 0.0)

    
    for feat in updates["alpha"]:
      
      pos = int(feat[3:])
      if pos == 0: continue
      #fsa.update_weight(self.weight_fst, self.single_state, self.arcs[pos], self.lagrangians["alpha"][feat])
      if self.non_neg:
        assert self.lagrangians["alpha"][feat] >= 0.0
      self.weight_fst.AddArc(self.single_state,
                             openfst.StdArc(pos, pos, updates["alpha"][feat], self.single_state))

    self.weight_fst.AddArc(self.single_state,
                           openfst.StdArc(fsa.RHO, fsa.RHO, 0.0, self.single_state))
    openfst.ArcSortInput(self.weight_fst)
    
  def delta_weights(self, updates, weights):
    self.set_weights(weights)

  def delta_weights_old(self, updates, weights):
    self.set_weights(weights)
    #
    self.full_create_weight_fst(updates)
    if self.non_neg:
      print "Intersecting"
      #openfst.ArcSortInput(self.weight_fst)
      self.temp_fst = fsa.rho_compose(self.fst, False, self.just_weight_fst, True, False)
    else:
      self.create_weight_fst(updates)
      self.temp_fst = fsa.rho_compose(self.temp_fst, False, self.weight_fst, True, True)
    
    #self.temp_fst = self.fst

    #self.temp_fst = openfst.StdVectorFst()    
    # self.temp_fst = openfst.StdComposeFst(self.fst, self.weight_fst, opts)
    #openfst.Intersect(self.fst, self.just_weight_fst, self.temp_fst)
    #self.temp_fst = self.fst #openfst.Intersect(self.fst, True, self.weight_fst, False)

    #arc_changes = []
    #print "Calc weights"
    #for feat in updates["alpha"]:
      #pos = int(feat[3:])
      #transitions = self.state_hash.get(pos, []) 
      #print feat, self.words[pos], len(transitions)
      #for ((state, arc), initial_weight) in transitions:
        #arc_changes.append((state, arc, initial_weight + self.lagrangians["alpha"][feat]))
#     print "Changing"
    #for state, arc, weight in arc_changes:
      #fsa.update_weight(self.fst, state, arc, weight)
    #self.temp_fst = self.fst


  def set_heuristic_fsa(self, simple_tree, orig_uni_tree, simple_count, step_count, nts, last_sym):    
    """
    simple_tree - FSA of trans tree with zero weights
    orig_uni_tree - FSA of trans tree with unigram weights
    simple_count - FSA with counting
    step_count - FSA with 1.0 penalty for each step
    nts- map of all non-terms
    """

    uni_tree = orig_uni_tree
    #openfst.RmEpsilon(orig_uni_tree)
    #openfst.Determinize(orig_uni_tree, uni_tree)
    #openfst.Minimize(uni_tree)
    t1 = fsa.rho_compose(uni_tree, False, simple_count, True, True)
    t2 = fsa.rho_compose(simple_tree, False, step_count, True, True)
    
    assert uni_tree.NumStates() == simple_tree.NumStates(), str(uni_tree.NumStates()) + " "+ str(simple_tree.NumStates())
    assert step_count.NumStates() == simple_count.NumStates()

    self.heuristic_count_fst = openfst.StdVectorFst()
    
    self.heuristic_fst = openfst.StdVectorFst()
    openfst.Determinize(t1, self.heuristic_fst)
    openfst.Determinize(t2, self.heuristic_count_fst)
    openfst.Connect(self.heuristic_count_fst)
    openfst.Connect(self.heuristic_fst)
    openfst.TopSort(self.heuristic_count_fst)
    openfst.TopSort(self.heuristic_fst)
    assert self.heuristic_count_fst.NumStates() == self.heuristic_fst.NumStates()

    self.reverse_heuristic_fst = openfst.StdVectorFst()
    openfst.Reverse(self.heuristic_fst, self.reverse_heuristic_fst)

    self.nts = set([nt[1] for nt in nts])
    print "creating tables"
    
#     heuristic = {}
#     for step in range(1,self.output_bound+1):
#       for nt in self.nts:
#         heuristic.setdefault((nt, step), 1e10)
#         #print nt, step, heuristic[nt, step]

#     def unzip3(m):
#       n = len(m)
#       a = openfst.IntVector(n)
#       b = openfst.IntVector(n)
#       c = openfst.FloatVector(n)
#       for i,k in enumerate(m):
#         #print k
#         a[i] = int(k[0])
#         b[i] = int(k[1])
#         #print i, k[0], k[1], self.heuristic_count_fst.InputSymbols().Find(k[0]), float(m[k])
#         c[i] = float(m[k])
#       return (a,b,c)
#     self.blanks = unzip3(heuristic)
#     self.orig_n = len(self.blanks[0])

    self.topo_dist = openfst.FloatVector()
    openfst.ShortestDistance(self.heuristic_count_fst, self.topo_dist, False)
    self.topo_dist = [int(uninf(u))+1 for u in self.topo_dist]

    self.topo_dist = openfst.IntVector(self.topo_dist)
    n = len(self.topo_dist)
    self.cache_step = [None] * n
#    for i in range(n):
#       for j in range(self.heuristic_count_fst.NumArcs(i)):
#         out = self.heuristic_count_fst.GetOutput(i,j)
#         next = self.heuristic_count_fst.GetNext(i,j)
#         if next >= n: continue
#         if self.topo_dist[next] > self.output_bound: continue
#         if out not in self.nts: continue
#         self.cache_heu.append((next, out, int(self.output_bound - self.topo_dist[next])+1))

    self.state_symbol = openfst.IntVector(n)
    for i in range(n):
      for j in range(self.heuristic_count_fst.NumArcs(i)):
         out = self.heuristic_count_fst.GetOutput(i,j)
         next = self.heuristic_count_fst.GetNext(i,j)
         if next >= n: continue
         if self.topo_dist[next] > self.output_bound+ 1: continue
         if out not in self.nts: continue
         self.cache_step[next] = (self.topo_dist[next], out)
         self.state_symbol[next] = out - tree_extractor.SRC_NODE

    self.heuristic_pruner = openfst.BeamPrune(self.reverse_heuristic_fst.NumStates(),
                                              self.output_bound+1,
                                              tree_extractor.SRC_NODE,
                                              self.last_sym)

  def calc_heuristic(self):
    tmp1 = openfst.StdVectorFst()
    tmp3 = openfst.StdVectorFst()

    print "len ", self.output_bound+1
    starttime2 = time.time()
    endtime2 = time.time()
    print "ShortViterbi", endtime2 - starttime2
    starttime2 = time.time()
    
    #tmp2 = fsa.rho_compose(self.heuristic_fst, False, self.just_weight_fst, True, False)
    beamCard = self.beamCard
    beamPrune = self.beamRad
    distances = openfst.FloatVector()
    lagrange_symbol = openfst.IntVector()
    lagrange_weight = openfst.FloatVector()
    for feat in self.lagrangians["alpha"]:
      pos = int(feat[3:])
      lagrange_symbol.append(pos)
      lagrange_weight.append(self.lagrangians["alpha"][feat])
    path = openfst.IntVector()

    start_time_heu = time.time()
    self.heuristic_pruner = openfst.BeamPrune(self.reverse_heuristic_fst.NumStates(),
                                              self.output_bound+1,
                                              tree_extractor.SRC_NODE,
                                              self.last_sym)
    end_time_heu = time.time()
    print "HEU CREATE time", end_time_heu - start_time_heu

    reached = openfst.IntVector()
    start_time_heu = time.time()
    distancesBestFinal = self.heuristic_pruner.BeamPruneAndOrder(self.reverse_heuristic_fst, tmp1,
                                            10.0, 100000, lagrange_symbol, lagrange_weight, path,
                                            distances, reached, True)
     
    print "Best Score ", distancesBestFinal
    rpath = list(path)
    #rpath.reverse()
    rpath = [r for r in rpath if r <> 0]


    print "\n\n\n"
    total = 0.0
    for p in rpath:
      if p == 0 or p == tree_extractor.SRC_NODE or p == tree_extractor.PRE_WORD: continue 
      total += self.lagrangians["alpha"]["UNI"+str(p)]
      print self.words[p], p, self.lagrangians["alpha"]["UNI"+str(p)]
    print "\n\n\n"

    print "Actual value ", self.score_path(rpath) + total
      
    end_time_heu = time.time()
    print "HEU RUN time", end_time_heu - start_time_heu
    endtime2 = time.time()
    print "Justbeam", endtime2 - starttime2

    
    starttime2 = time.time()
    
            
    #print [u for u in uni_dist]
    #print [t for t in topo_dist]

    # cutoff the first

    #uni_dist = [uninf(u) for u in distances][1:]
    #print uni_dist
    #assert len(uni_dist) == len(self.topo_dist), str(len(uni_dist)) + " "+ str(len(self.topo_dist))
    #assert len(topo_dist) == self.heuristic_count_fst.NumStates() , str(len(uni_dist)) + " "+ str(self.heuristic_count_fst.NumStates())
    #assert len(uni_dist) == self.heuristic_count_fst.NumStates() , str(len(uni_dist)) + " "+ str(self.heuristic_count_fst.NumStates())
    heuristic = {}
    #n = len(uni_dist)
    num = 0

    n = len(reached)
    
#     a = openfst.IntVector(n)
#     b = openfst.IntVector(n)
#     c = openfst.FloatVector(n)

#     start_time_heu = time.time()    
    #for i, r in enumerate(reached):
       #if math.isinf(u): continue
       #if not self.cache_step[r-1]: continue
       #(step, sym) = self.cache_step[r-1]
       #print >>sys.stderr,  int(step), self.fst.InputSymbols().Find(sym), sym- tree_extractor.SRC_NODE, distances[r]
#       a[i] = int(step)
#       b[i] = int(sym) - tree_extractor.SRC_NODE
#       c[i] = float(distances[r])
#     end_time_heu = time.time()
#     print "HEU FINISH time", end_time_heu - start_time_heu, len(reached)

    
    print self.last_sym
    print self.output_bound
    #a,b,c = self.blanks
    #a.resize(n + self.orig_n)
    #b.resize(n + self.orig_n)
    #c.resize(n + self.orig_n)

    
#     self.cache_heu = []
#     n = len(self.topo_dist)
#     for i in range(n):
#       for j in range(self.heuristic_count_fst.NumArcs(i)):
#         out = self.heuristic_count_fst.GetOutput(i,j)
#         next = self.heuristic_count_fst.GetNext(i,j)
#         if next >= n: continue
#         if self.topo_dist[next] > self.output_bound: continue
#         if out not in self.nts: continue

#         num += 1
#         a[num+self.orig_n] = out
#         b[num+self.orig_n] = topo
#         c[num+self.orig_n] = float(uni_dist[next])

#        self.cache_heu.append((next, out, int(self.output_bound - self.topo_dist[next])+1))


#     for next, out, topo in self.cache_heu:
#       num +=1
#       #print self.heuristic_count_fst.InputSymbols().Find(out), out, uni_dist[next], self.output_bound - topo_dist[next]
#       a[num+self.orig_n] = out
#       b[num+self.orig_n] = topo
#       c[num+self.orig_n] = float(uni_dist[next])

    endtime2 = time.time()

    return distances, reached


  def beam_search(self):
    if True:
      print "Beam", self.output_bound
      beamed_fst = openfst.StdVectorFst()
      inter = openfst.StdVectorFst()


#       print "Heu"
#       starttime2 = time.time()
#      a,b,c = self.calc_heuristic()
#       endtime2 = time.time()
#       print "done heu", endtime2 - starttime2

      starttime2 = time.time()


      
      # If we haven't created our pruning options, make them here
      if self.pruner == None:
        self.pruner = openfst.BeamPrune(self.original_size, self.output_bound+1, tree_extractor.SRC_NODE, self.last_sym)
        print self.output_bound+1
        print self.last_sym


      beam = self.beamRad
      beamCard = self.beamCard

      # Holds the distances to each node
      distances = openfst.FloatVector()

      # arrays for the extra lagrangian values
      self.lagrange_symbol = openfst.IntVector()
      self.lagrange_weight = openfst.FloatVector()
      for feat in self.lagrangians["alpha"]:
        pos = int(feat[3:])
        self.lagrange_symbol.append(pos)
        self.lagrange_weight.append(self.lagrangians["alpha"][feat])
      
      
      path = openfst.IntVector()


      # Calculate the heuristic values for the search  
      start_time_heu  = time.time()
      distances,reached = self.calc_heuristic()
      end_time_heu  = time.time()

      
      print "HEU time", end_time_heu - start_time_heu
      self.pruner.SetHeuristicBack(distances, reached, self.topo_dist, self.state_symbol)
      #self.pruner.SetHeuristic(openfst.IntVector(), openfst.IntVector(),openfst.FloatVector())

      #for r in reached:
      #  print r, distances[r]
      #self.pruner.SetHeuristic(openfst.IntVector(), openfst.IntVector(), openfst.FloatVector())
      print "Heuristic set"
      reached = openfst.IntVector()
      #fsa.print_fst(openfst.StdVectorFst(self.temp_fst))
      #if self.round % 10 <> 0:

      # distBestFinal = self.pruner.BeamPruneAndOrder(self.temp_fst,
#                                                 beamed_fst, beam, beamCard, 
#                                                 self.lagrange_symbol, self.lagrange_weight, path, distances,
#                                                 reached, True)
      
      #else:
      bestFinal = self.pruner.AStar(self.temp_fst,
                                    self.lagrange_symbol, self.lagrange_weight, path, distances,
                                    reached)
#                                      openfst.IntVector(), openfst.FloatVector(), path)
      distBestFinal = distances[bestFinal]
      endtime2 = time.time()
      print "Done", distBestFinal
      print "BEAMING", endtime2 - starttime2
            
      print "DONE"

    # the shortest path
    path = list(path)
    path.reverse()
    return (path, distBestFinal)
    
    
  def decode(self):
    "create an fst out of the lagrangians, and compose it"
    self.round += 1
    print "LM DECODER \n\n"
    (path,best) = self.beam_search()
    print "ShortestPath "
    

    # Done searching, now process all this information
    return self.process_shortest(path, best)

  def process_shortest(self, path, total):
    output = Vector()
    sent = []
    i = 0



    for p in path:
        if p == 0 or p == tree_extractor.SRC_NODE or p == tree_extractor.PRE_WORD: continue 

        
        sent.append(self.words[p])
        print self.words[p], p #shortest.GetWeight(i,j),  shortest.GetWeight(i,j)- self.lagrangians["alpha"]["UNI" + str(p)], self.lagrangians["alpha"]["UNI" + str(p)], p

          
        if p > tree_extractor.PRE_WORD:
          output["UNI"+str(p)] += 1.0

    fsent = [s  for s in sent if isinstance(s, str)] 
    print " ".join(map(super_strip, fsent))

    print "The total is", total
      
    return (total, {"alpha":output, "beta":Vector()}, (fsent, None, None))

        
  def score_lm(self, w1, w2):
    return self.lm_weight * self.lm.word_prob_bystr(super_strip(w2), super_strip(w1))

  def score_path(self, path):
    inter_fsa = openfst.StdVectorFst()
    short = openfst.StdVectorFst()
    chain_fsa = fsa.make_chain_fsa(path, self.original_fst.InputSymbols())
    #print "CHAIN"
    #fsa.print_fst(chain_fsa)
    path = openfst.IntVector()
    distances = openfst.FloatVector()
    reached = openfst.IntVector()

    openfst.Intersect(self.original_fst, chain_fsa, inter_fsa)


    openfst.ShortestPath(inter_fsa,short, 1)
    #self.pruner.BeamPruneAndOrder(short,
    #                              openfst.StdVectorFst(), self.beamRad, self.beamCard, 
    ##                              self.lagrange_symbol, self.lagrange_weight, path, distances,
    #                              reached)
    assert short.NumStates() <> 0
    
    lm2 = fsa.get_weight(short)
    return lm2
  
  def oracle(self, (subtree, fv, path)):
    lm2 = 0
    #print path
    #print [self.words[a] for a in path]
    print "First check"
    fsa.check_fst(path, self.just_tree_fst, self.words)
    print "second check"
    #lm_path = [p for p in path if p <> tree_extractor.SRC_NODE if "+++" not in self.words[p][0] ]
    #fsa.check_fst(lm_path, self.just_lm_fst, self.words)
    print "third check"
    #edge_path = [p for p in path if "edge" in self.words[p][0] ]
    #s_path = [self.words[p] for p in edge_path ]
    #print s_path
    #print sum([n for (_, (n,_)) in s_path])
    #fsa.check_fst(edge_path, self.just_count_fst, self.words)
    
    print "final check"
    lm2 = self.score_path(path)
    print "LM from fst is ", lm2
    print "LM from model is", 0.141 * self.lm.word_prob(subtree)
    #sent = ["<s> <s>"] + subtree.split() + ["</s>"]
#print sent
 #for w1, w2 in izip(sent, sent[1:]):
       #lm2 += self.score_lm(w1,w2)
      #print w1, w2, self.score_lm(w1,w2)

    #print "With %s"% lm2

    without_lagrangians = self.lagrangians["alpha"].dot(fv) + self.lagrangians["beta"].dot(fv) 
    #print "Just %s"% without_lagrangians
      
    return (lm2, lm2 + without_lagrangians)
    #return (lm2-without_lagrangians, lm2)

  def oracle_bigrams(self, sent):
    lm2 = 0
    sent = ["<s>"] + sent + ["</s>"]
    for w1, w2 in izip(sent, sent[1:]):
      lm2 += self.score_lm(w1,w2)
    return lm2
