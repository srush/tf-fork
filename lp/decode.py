"""
Implements the two decoders for fast, dual decomposition with lm
"""

from __future__ import division



import sys
from itertools import *
sys.path.append('..')
import gflags as flags
from ngram import Ngram
from model import Model
from forest import Forest
from svector import Vector
import copy
import openfst
import tree_extractor
import decoders
import fsa as FSA
import gc, cPickle, time
FLAGS = flags.FLAGS

INF = 1e90

UP = 1
DOWN = 0


def dict_multi(key_val_list):
  dict = {}
  for (key, val) in key_val_list:
    dict.setdefault(key, [])
    dict[key].append(val)
  return dict

def span_rel(span1, span2):
  if is_under(span2, span1):
    return "above"
  elif is_under(span1, span2):
    return "under"
  elif (span1[1] <= span2[0]) or (span1[0] >= span2[1]):
    return "side"
  else: return None


def is_under(span1, span2):
  return span1[0] <= span2[0] and span1[1] >= span2[1]

def print_weights(weights):
  print "Size %s" % len(weights)
  print "Norm %s" % sum([weights[f] for f in weights])
  for feat in weights:
    if weights[feat] <> 0.0:
      print feat + "\t" + str(weights[feat])


def is_around(span1, span2):
  # ok , so either on one side
  one_side =  (span1[1] <= span2[0]) or (span1[0] >= span2[1]) 
  # or strictly bigger
  bigger = is_under(span2, span1)
  return one_side or bigger

def is_lex(word):
  return word[0]=="\""

def strip_lex(word):
  if word == "\"": return word
  elif word == "\"\\\"\"": return "\""
  return word.strip("\"")


def get_sym_pos(word):
  assert word[0] == "x"
  return int(word[1:])


def crunch(object):
  return str(object).replace(" ", "+")

def make_edge_str(edge):
  return crunch("EdgeID/" + str(edge.head.span) + "/" + crunch(edge.rule.lhs)+ "/" + ",".join(edge.rule.rhs))

def add_edge_pos_ind(edgestr, ind):
  return "Next" + edgestr + "/" + str(ind)

def is_edge_str(str):
  return str.startswith("EdgeID")


def edge_get_back_bigrams(edge, only_left):
  if is_lex(edge.rule.rhs[0]):
    yield 0
  if not only_left:
    rhs = edge.rule.rhs
    for i,word in enumerate(rhs[1:], 1):
      if is_lex(word) and not is_lex(rhs[i-1]):
        yield i


def collect_fst_hash(fst):
  state_hash = {}
  for i in range(fst.NumStates()): 
    for j in range(fst.NumArcs(i)):
      state_hash.setdefault(fst.GetOutput(i, j), [])
      state_hash[fst.GetOutput(i, j)].append(((i,j), fst.GetWeight(i,j)))
  return state_hash

class TreeDecoder(object):
  """
  Enforce that the parse is a valid tree
  """
  def __init__(self, forest, weights, unigram = {}, table = {}):
    
    self.weights = weights
    self.forest = forest.copy()
    self.table = table

    # augment features in forest with edge names for lagrangians
    for node in self.forest:
      for edge in node.edges:
        s = make_edge_str(edge)
        edge.fvector[s] = 1.0
        l1 = len(filter(is_lex, edge.rule.rhs))
        l2 = len(unigram.get(edge.position_id, []))
        #assert l1 == l2, "%s %s %s %s"%(l1,l2, edge.rule.rhs, unigram.get(edge.position_id, []))  
        print edge.rule.rhs, unigram.get(edge.position_id, [])
        for id in unigram.get(edge.position_id, []):
          edge.fvector["UNI" + str(id)] = 1.0
        
        for ind in edge_get_back_bigrams(edge, False):
          edge.fvector[add_edge_pos_ind(s, ind)] = 1.0
          
    self.lagrangians = {"alpha" : Vector(), "beta" : Vector()}
  
  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def delta_weights(self, updates,weights):
    self.set_weights(weights)

  def decode(self):
    cur_weights = self.weights.__copy__()
    cur_weights += self.lagrangians["alpha"] 
    (best, subtree, best_fv) = self.forest.bestparse(cur_weights, use_min=True)
    #print subtree
    
    ret = {"alpha":Vector(), "beta":Vector()}
    for feat in best_fv:
      # can only turn on each element once 
      if feat.startswith("UNI"):
        ret["alpha"][feat] += best_fv[feat]
        #print self.table[int(feat[3:])]
        #print feat
    #print "sanity %s" % (self.weights.dot(best_fv) + self.lagrangians["alpha"].dot(best_fv) + self.lagrangians["beta"].dot(best_fv)) 
    print "Subtree", subtree
    
    return (best, ret, (subtree, best_fv, None)) 

#   def decode(self):
#     cur_weights = self.weights.__copy__()
#     cur_weights += self.lagrangians["alpha"] + self.lagrangians["beta"] 
#     (best, subtree, best_fv) = self.forest.bestparse(cur_weights, use_min=True)
#     print subtree
#     ret = {"alpha":Vector(), "beta":Vector()}
#     for feat in best_fv:
#       # can only turn on each element once 
#       if is_edge_str(feat):
#         ret["alpha"][feat] += best_fv[feat]
#       if feat.startswith("Next"):
#         ret["beta"][feat] += best_fv[feat]

#     best += self.lagrangians["beta"]["NextEndTOKEN"]
#     ret["beta"]["NextEndTOKEN"] = 1.0
#     best_fv["NextEndTOKEN"] = 1.0

#     #print "sanity %s" % (self.weights.dot(best_fv) + self.lagrangians["alpha"].dot(best_fv) + self.lagrangians["beta"].dot(best_fv)) 

#     return (best, ret, (subtree, best_fv, None)) 

  def oracle(self, (subtree, fv, _)):
    #print_weights(fv)
    pure = self.weights.dot(fv)
    with_lagrangians = pure + self.lagrangians["alpha"].dot(fv) + self.lagrangians["beta"].dot(fv)
    return (pure, with_lagrangians)
  
class LMDecoder(object):
  """
  Upperbound the LM score of a covering
  """
  def __init__(self, forest, lm, lm_weight):
    self.lm = lm
    self.forest = forest.copy()
    self.lm_weight = lm_weight

    # augment features in forest with edge names for lagrangians
    for node in self.forest:
      for edge in node.edges:
        s = make_edge_str(edge)
        edge.fvector[s] = 1.0
        #for ind in edge_get_back_bigrams(edge):
        #  edge.fvector[add_edge_pos_ind(s, ind)] = 1.0

    self.lagrangians = {"alpha" : Vector(), "beta" : Vector()}
  
  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def score_lm(self, w1, w2):
    #return 1.0
    return self.lm_weight * self.lm.word_prob_bystr(strip_lex(w2), strip_lex(w1))

  def oracle(self, (subtree, fv, _)):
    lm2 = 0
    sent = ["<s>"] + subtree.split() + ["</s>"]
    for w1, w2 in izip(sent, sent[1:]):
      print w1, w2, self.score_lm(w1,w2)
      lm2 += self.score_lm(w1,w2)
      
    with_lagrangians = self.lagrangians["alpha"].dot(fv)
    return (lm2, lm2+with_lagrangians)

  def oracle_bigrams(self, fv, bigrams):
    lm2 = 0

    for (w1, w2) in bigrams: 
      lm2 += self.score_lm(w1, w2)
    #print "ORACLE SAYS %s"%lm2
    with_lagrangians = self.lagrangians["alpha"].dot(fv) + self.lagrangians["beta"].dot(fv) 

    

    return lm2, lm2 + with_lagrangians

  def decode(self):
    #print "starting to decode"
    weights = self.lagrangians["alpha"].__copy__()
    for feat in weights:
      assert is_edge_str(feat)
      #print feat, weights[feat]

    beta = self.lagrangians["beta"]
    weights["lm2"] = 1.0
    weights["extra_beta"] = 1.0
    
    temp_weights= Vector()
    temp_weights["lm2"] = 1.0
    
    
    # for each edge compute the best bigrams 
    for node in self.forest:
      for edge in node.edges:
        lm, ex_beta, next = self.best_next_bigrams(edge, beta)
        edge.fvector["lm2"] = lm
        edge.fvector["extra_beta"] = ex_beta
        assert edge.fvector[make_edge_str(edge)] == 1.0
        for feat in next:
          edge.fvector[feat] += 1.0

    (best, subtree, best_fv) = self.forest.bestparse(weights, use_min=True)

    assert abs(best- weights.dot(best_fv)) < 1e-3, str(best) + " " + str(weights.dot(best_fv)) 
    # get to hallucinate one free edge to start with
    best_first = None
    best_first_score = INF
    best_bi = None
    for node in self.forest:
      for edge in node.edges:
        if is_lex(edge.rule.rhs[0]):
          s = add_edge_pos_ind(make_edge_str(edge), 0)
          score = self.score_lm("<s>", edge.rule.rhs[0])
          score += beta[s]

          if score < best_first_score:
            best_first = s
            best_first_score = score
            best_first_beta = beta[s]
            best_bi = "Bi<s>++"+edge.rule.rhs[0]
    best += best_first_score

    best_fv[best_bi] = 1.0
    best_fv["extra_beta"] += best_first_beta
    best_fv["lm2"] += best_first_score - best_first_beta
    
    assert best_first 
    # end dream
    
    print subtree
    ret = {"alpha":Vector(), "beta":Vector()}

    bigrams = []
    all_alpha = 0.0
    for feat in best_fv:
      if is_edge_str(feat):
        #print "ALPHA", feat, best_fv[feat] * self.lagrangians["alpha"][feat]
        ret["alpha"][feat] += best_fv[feat]
      if feat.startswith("Next"):
        ret["beta"][feat] += best_fv[feat]
      if feat.startswith("Bi"):
        bigrams.extend([feat[2:].split("++")] * int(best_fv[feat]))
    ret["beta"][best_first] += 1.0

    best_fv[best_first] += 1.0
    #print "I THINK IT IS", str(temp_weights.dot(best_fv) + best_first_score - best_first_beta)
    #assert abs(((best - best_first_score + best_first_beta) - ret["alpha"].dot(self.lagrangians["alpha"]) - ret["beta"].dot(self.lagrangians["beta"]) ) - temp_weights.dot(best_fv)) < 1e-4, str(best - best_first_score) + " " + str(temp_weights.dot(best_fv)) + " "+str((best - best_first_score + best_first_beta) - ret["alpha"].dot(self.lagrangians["alpha"]) - ret["beta"].dot(self.lagrangians["beta"])) 
    

    
    # remove next pseudo features
    for node in self.forest:
      for edge in node.edges:
        for feat in edge.fvector:
          if feat.startswith("Next") or feat.startswith("Bi"):
            del edge.fvector[feat]
    
    return best, ret, (subtree, best_fv, bigrams)

    #self.compute_best_bigram(forest.root, self.lm, alpha, beta)

  def get_all_edges(self, span, is_rightmost):
    for node in self.forest:
      rel = span_rel(span,node.span)
      if is_rightmost and rel == "side":
        for edge in node.edges:
          for ind in edge_get_back_bigrams(edge, True):
            yield (ind, edge)
      if rel == "above" and is_rightmost:
        for edge in node.edges:
          for i, sub in enumerate(edge.subs):
            rel = span_rel(span, sub.span)
            if rel == "above":
              for j, w in enumerate(edge.rule.rhs):
                if is_lex(w) and j >0 and edge.rule.rhs[j-1] == "x"+str(i):
                  yield (j, edge)
              break
      if (rel == "under" or span == node.span ) and not is_rightmost:
        for edge in node.edges:
          for ind in edge_get_back_bigrams(edge, True):
            yield (ind, edge)
        
    
            
    
  def best_next_bigrams(self, edge, beta):
    rhs = edge.rule.rhs
    lm2 = 0.0
    ex_beta = 0.0
    lm_next = []
    
    #first check for internal bigrams
    for w1, w2 in izip(rhs, rhs[1:]):
      if is_lex(w1) and is_lex(w2):
        lm2 += self.score_lm(w1,w2)
        lm_next.append("Bi"+w1+"++"+w2)

    def max_lm(w, pos_next):
      best_next = None
      best_score = INF
      for (ind, next) in pos_next:
        #print ind , next
        lm = self.score_lm(w, next.rule.rhs[ind])
        ex_beta = beta[add_edge_pos_ind(make_edge_str(next), ind)]
        score = lm + ex_beta
        if score < best_score:
          best_lm = lm
          best_beta = ex_beta
          best_score = score
          best_next = [add_edge_pos_ind(make_edge_str(next), ind),
                       "Bi"+w+"++"+next.rule.rhs[ind]]
          #assert best_next <> None
      if best_next == None:
        return None
      return (best_lm, best_beta, best_next)

    # now check for holes (skip first, since we only care about forward bigrams)
    for i, w in enumerate(rhs[1:],1):
      if not is_lex(w) and is_lex(rhs[i-1]):
        sym_pos = get_sym_pos(w)
        
        # get all edges my size or below, that start with a word
        pos_next = self.get_all_edges(edge.subs[sym_pos].span, False)

        (best_lm, best_beta, best_next) = max_lm(rhs[i-1], pos_next)
        lm2 += best_lm
        ex_beta += best_beta
        lm_next.extend(best_next)
      
      
    # finally check the last word
    if is_lex(rhs[-1]):
      
      # get all edges not all my size or below that start with a word
      pos_next = self.get_all_edges(edge.head.span, True)
      ret  = max_lm(rhs[-1], pos_next)
      if ret <> None:
        best_lm, best_beta, best_next = ret
      # also could be last word
      last_lm =self.score_lm(rhs[-1], "</s>")
      last_beta = beta["NextEndTOKEN"]
      score = last_lm + last_beta
        
      if not ret or score < best_lm + best_beta:
        lm2 += last_lm
        ex_beta += last_beta
        lm_next.append("NextEndTOKEN")
        lm_next.append("Bi"+rhs[-1]+"++</s>")
      else:
        lm2 += best_lm
        ex_beta += best_beta
        lm_next.extend(best_next)
    #print edge, edge.rule.rhs
    #print lm_next

    return lm2, ex_beta, lm_next


class LMDecoderFST(object):
  def __init__(self, fst, chinese_len, all_trans, words, edge_to_states, states_to_edge, output_to_states, states_to_output, lm ,lm_weight):
    self.fst = fst 
    print "Length is %s"%chinese_len  
    self.len = chinese_len
    self.lagrangians = {"alpha": Vector(), "beta": Vector()}
    self.all_trans = all_trans
    self.words = words

    self.edge_to_states = edge_to_states
    self.states_to_edge = states_to_edge
    self.output_to_states = output_to_states
    self.states_to_output = states_to_output

    self.lm = lm 
    self.lm_weight = lm_weight 

    self.weight_fst = openfst.StdVectorFst()
    self.single_state = self.weight_fst.AddState()
    self.weight_fst.SetStart(self.single_state)
    self.weight_fst.SetFinal(self.single_state, 0.0)

    self.tmp = []
    self.states= {}
    for s in self.all_trans:
      self.states[s] = self.weight_fst.AddState() 
      #self.weight_fst.AddArc(self.single_state, openfst.StdArc(s, s, self.lagrangians["alpha"]["UNI"+str(s)], self.states[s]))
      #self.weight_fst.AddArc(self.states[s], openfst.StdArc(openfst.epsilon, openfst.epsilon, 0.0, self.single_state))
      self.weight_fst.AddArc(self.single_state, openfst.StdArc(s, s, self.lagrangians["alpha"]["UNI"+str(s)], self.single_state))

    self.counter_fst = openfst.StdVectorFst()
    first = self.counter_fst.AddState()
    self.counter_fst.SetStart(first)
    self.counter_fst.SetFinal(first, 0.0)
    states = [first]
    
    for i in range(int(int(self.len) * 4.0) + 2):
      states.append( self.counter_fst.AddState())
      self.counter_fst.SetFinal(states[-1], 0.0)
      for s in range(len(self.output_to_states)):
        
        self.counter_fst.AddArc(states[-2], openfst.StdArc(s, s, 0.0, states[-1]))
        #self.counter_fst.AddArc(states[-1], openfst.StdArc(s+1, s+1, 0.0, states[-1]))

    

    last = self.counter_fst.AddState()
    #for word in self.all_trans:
    for s in range(len(self.output_to_states)):
        
        self.counter_fst.AddArc(states[-1], openfst.StdArc(s, s, 0.0, last))
        self.counter_fst.AddArc(last, openfst.StdArc(s, s, 0.0, last))
        #self.counter_fst.AddArc(last, openfst.StdArc(s+1, s+1, 0.0, last))

      #openfst.Intersect(self.fst, self.counter_fst, run_fst)
    openfst.Minimize(self.counter_fst)
    intersect_fst = openfst.StdVectorFst()
    print "Intersecting"
    openfst.Intersect(self.fst, self.counter_fst, intersect_fst)
    #intersect_fst = self.fst
    self.intersect_fst = openfst.StdVectorFst()
    
    openfst.Determinize(intersect_fst, self.intersect_fst)
    openfst.Minimize(self.intersect_fst)
    self.intersect_fst = intersect_fst

    #print "Intersection size %s" % self.intersect_fst.NumStates()

    self.state_hash = collect_fst_hash(self.intersect_fst)
    
    
  def set_weights(self, lagrangians):
    self.lagrangians = lagrangians

  def delta_weights(self, updates, weights):
    self.set_weights(weights)

#     self.weight_fst = openfst.StdVectorFst()
#     self.single_state = self.weight_fst.AddState()
#     self.weight_fst.SetStart(self.single_state)
#     self.weight_fst.SetFinal(self.single_state, 0.0)
#     self.states= {}

#     for s in self.all_trans:
#       self.states[s] = self.weight_fst.AddState() 
#       #self.weight_fst.AddArc(self.single_state, openfst.StdArc(s, s, self.lagrangians["alpha"]["UNI"+str(s)], self.states[s]))
#       #self.weight_fst.AddArc(self.states[s], openfst.StdArc(openfst.epsilon, openfst.epsilon, 0.0, self.single_state))
#       self.weight_fst.AddArc(self.single_state, openfst.StdArc(s, s, self.lagrangians["alpha"]["UNI"+str(s)], self.single_state))

#     return
    print "Adding updates"
    for feat in updates["alpha"]:
      pos = int(feat[3:])
      #print pos
      print self.states_to_output
      transitions = self.state_hash.get(self.states_to_output[pos], []) # hidden_state
      
      for ((state, arc), initial_weight) in transitions:
       
        iter = openfst.StdMutableArcIterator(self.intersect_fst, state)
        iter.Seek(arc)
        old_arc= iter.Value()
      
#       #print "Adding arc %s %s %s"%(pos, feat, self.lagrangians["alpha"][feat])
      
#       #arc = self.weight_fst.GetArc(self.single_state, self.states[pos])
#       # so ridic. need to make a temp arc because pyopenfst sucks
      #self.temp_arcs.append( openfst.StdArc(0,0,self.lagrangians["alpha"][feat], 0))
#       #print temp_arc.weight.Value()
      #tmp = openfst.StdArc(0,0,self.lagrangians["alpha"][feat], 0)
      #.weight = copy.copy(tmp.weight)
      #print "position " + str(pos+1)
        #assert old_arc.ilabel == pos + 1 
        #assert old_arc.olabel == pos + 1
        #assert old_arc.nextstate == pos+1
        arc = openfst.StdArc(old_arc.ilabel,old_arc.olabel, initial_weight + self.lagrangians["alpha"][feat], old_arc.nextstate)
        iter.SetValue(arc)
    print "Done adding updates"
      #del tmp
      #print pos, pos+1, self.lagrangians["alpha"][feat], temp_arc.weight.Value(), self.fst.GetArc(pos, pos +1).weight.Value()
      
      #assert self.weight_fst.GetWeight(self.single_state, self.states[pos]) == self.lagrangians["alpha"][feat], self.weight_fst.GetWeight(self.single_state, self.states[pos]) 
#       # self.matcher.SetState(self.single_state);
# #       assert matcher.Find(pos)
# #       arc = matcher.Value()
# #arc.weight = self.lagrangians["alpha"][feat]
#       #self.weight_fst.DeleteArcs(self.single_state, openfst.StdArc(pos, pos, self.lagrangians["alpha"][feat], self.single_state))
#       #self.weight_fst.AddArc(self.single_state, openfst.StdArc(pos, pos, self.lagrangians["alpha"][feat], self.single_state)


  def decode(self):
    "create an fst out of the lagrangians, and compose it"
    
    #openfst.ArcSortInput(self.weight_fst)

    temp_fst = openfst.StdVectorFst()
    run_fst = openfst.StdVectorFst()
    det_run_fst = openfst.StdVectorFst()
  
    print "Intersecting"
    #openfst.Intersect(self.counter_fst, self.fst, run_fst)
    #openfst.Intersect(self.intersect_fst, self.weight_fst, run_fst)
    #openfst.Determinize(run_fst, det_run_fst)
    #openfst.Minimize(det_run_fst)
    print "Done Intersecting"
    
    shortest = openfst.StdVectorFst()

    print "ShortestPath %s %s"%(self.fst.NumStates(), self.intersect_fst.NumStates())
    openfst.ShortestPath(self.intersect_fst, shortest, 1)
    print "Done ShortestPath"
    openfst.TopSort(shortest)

    output = Vector()
    total = 0.0
    withv = 0.0
    without = 0.0
    sent = []
    i = 0
    print shortest.NumStates()
    for i in range(shortest.NumStates()):    
      for j in range(shortest.NumArcs(i)):
        p = shortest.GetOutput(i,j)
        (a,b) = self.output_to_states[p]
        total += shortest.GetWeight(i,j)
        print p, (a,b), shortest.GetWeight(i,j)
        
        if a == b: # fake var
          withv += shortest.GetWeight(i,j)
          continue
        else:
          without+= shortest.GetWeight(i,j)
          
        output["UNI"+str(b)] += 1.0
        if b <>0:
          sent.append(self.words[b])
        #print "UNI"+str(shortest.GetOutput(i,j)), 1.0
        #if i:
    total += shortest.FinalWeight(i)
    print "with %s" %withv
    print "without %s "% without
    print " ".join(map(strip_lex, sent))
    return (total, {"alpha":output, "beta":Vector()}, (sent, None, None))


  def score_lm(self, w1, w2):
    #return 1.0
    return self.lm_weight * self.lm.word_prob_bystr(strip_lex(w2), strip_lex(w1))

  def oracle(self, (subtree, fv, _)):
    lm2 = 0
    sent = ["<s>"] + subtree.split() + ["</s>"]
    for w1, w2 in izip(sent, sent[1:]):
      lm2 += self.score_lm(w1,w2)
      print w1, w2, self.score_lm(w1,w2)
    print "Without %s"% lm2

    with_lagrangians = self.lagrangians["alpha"].dot(fv) + self.lagrangians["beta"].dot(fv) 
    print "Just %s"% with_lagrangians
    return (lm2, lm2+with_lagrangians)

  def oracle_bigrams(self, sent):
    lm2 = 0
    sent = ["<s>"] + sent + ["</s>"]
    for w1, w2 in izip(sent, sent[1:]):
      lm2 += self.score_lm(w1,w2)
    return lm2

    
class SimpleDual(object):
  def __init__(self, s1, s2, non_neg = False):
    self.s1 = s1
    self.s2 = s2
    self.weights = {"alpha":Vector(), "beta":Vector()}
    self.round = 0
    self.old_dual=[]
    self.old_primal=[]
    self.lowest_primal = 1e20
    self.highest_dual = 0.0
    self.round = 1
    self.nround = 0
    self.should_print = False
    self.non_neg = non_neg
  def update_weights(self, subgrad):

    size = 0.0
    for feat in subgrad["alpha"]:
      if subgrad["alpha"][feat] == 0.0:
        del subgrad["alpha"][feat]
      else: size += 1

    if len(self.old_dual) > 2 and self.old_dual[-1] < self.old_dual[-2]:
      self.round += 1 
    elif len(self.old_dual) == 1: 
      self.base_weight = (self.old_primal[-1] - self.old_dual[-1]) / max(float(size),1.0)
      
    
    self.nround += 1
    alpha = (self.base_weight) * (0.98) ** (5* float(self.round))
      
    print "ALPHA %s" % alpha 
    updates = {"alpha" : alpha * subgrad["alpha"],
               "beta" :  alpha * subgrad["beta"]}

        

    self.weights["alpha"] += updates["alpha"]
    self.weights["beta"] += updates["beta"]

    self.send_weights(updates)

    if self.should_print:
      print "FULL ALPHA:"
      self.print_weights(self.weights["alpha"])
      print 
      #print "FULL BETA:"
      #self.print_weights(self.weights["beta"])
      
      print "ALPHA:"
      self.print_weights(subgrad["alpha"])
      print 
      #print "BETA:"
      #self.print_weights(subgrad["beta"])

  def send_weights(self, updates):
    self.s1.delta_weights(updates,self.weights)
    
    #self.s2.set_weights({
    #  "alpha":-self.weights["alpha"],
    #  "beta": -self.weights["beta"]})
    self.s2.delta_weights({
      "alpha": -updates["alpha"],
      "beta": -updates["beta"]},
      {"alpha": -self.weights["alpha"],
      "beta": -self.weights["beta"]}
                          )


  
  def print_weights(self, weights):
    print_weights(weights)

  def set_alpha(self, v):
    self.weights["alpha"] =v
    self.send_weights()
    
  def set_beta(self, v):
    self.weights["beta"] =v
    self.send_weights()
    
  def run_one_round(self):
    obj,  sub, der = self.s1.decode()
    obj2, sub2, der2  = self.s2.decode()

    print "\n\n\n"

#     print "Alpha Bi"
#     self.print_weights(sub2["alpha"])

#     print "Beta Bi"
#     self.print_weights(sub2["beta"])

#     print "FV"
#     self.print_weights(der2[1])
        
    self.old_dual.append(obj2+obj)
    
    if self.old_dual[-1] > self.highest_dual:
      self.highest_dual = self.old_dual[-1]
    
    print "Round : %s"% self.nround
    print "Dual  %s %s %s" % (obj, obj2, str(obj2+obj))

    if self.should_print:
      print "alpha"
      self.print_weights(sub["alpha"])
      #print "beta"
      #self.print_weights(sub["beta"])
      print 
      print "alpha"
      self.print_weights(sub2["alpha"])
      #print "beta"
      #self.print_weights(sub2["beta"])
    
    (primal, oracle) = self.s1.oracle(der)
    (primal2, oracle2) = self.s2.oracle(der)
    print "Primal1 %s %s %s" %(primal, primal2, str(primal+primal2))
    print "Oracle1 %s %s %s" %(oracle, oracle2, str(oracle+oracle2))
    
    self.old_primal.append(primal+primal2)
    if self.old_primal[-1] < self.lowest_primal:
      self.lowest_primal = self.old_primal[-1]
    
    print "Best Gap %s"%(self.lowest_primal - self.highest_dual)
    print "Best Primal %s"%(self.lowest_primal)
    print "Highest Dual %s"%(self.highest_dual)

    
    #assert (self.lowest_primal - self.highest_dual) > -1e-3, "%s %s" % (self.lowest_primal, self.highest_dual)
#     should = (obj - (self.weights["alpha"].dot(sub["alpha"]) + self.weights["beta"].dot(sub["beta"])))
#     assert abs(primal- should) < 1e-4, "%s %s "%(primal, should)
#     assert abs(oracle - obj) < 1e-4, "%s %s" % (oracle, obj) 

#     (primal,oracle) = self.s1.oracle(der2)
#     (primal2, oracle2) = self.s2.oracle(der2)
#     print "Primal2 %s %s %s" %(primal, primal2, str(primal+primal2))
#     print "Oracle2 %s %s %s" %(oracle, oracle2, str(oracle+oracle2))

    # Can't check this, not a feasible solution
    dual_check = self.s2.oracle_bigrams(der2[0])
    
    should = (obj2 - (self.s2.lagrangians["alpha"].dot(sub2["alpha"]) + self.s2.lagrangians["beta"].dot(sub2["beta"])))
    #assert abs(dual_check - should) < 1e-4, "%s %s" %(obj2, should, dual_check)
    #print "S2 feas %s %s %s %s" % (obj2, should, non_feas_primal, self.s2.lagrangians["alpha"].dot(sub2["alpha"]))
    #assert abs(non_feas_primal- should) < 1e-4, "%s %s %s" % (obj2, should, non_feas_primal)
    #assert abs(non_feas_oracle - obj2) < 1e-4, "%s %s" % (non_feas_oracle, obj2) 



    subgrad = {"alpha" : (sub["alpha"] - sub2["alpha"]),
               "beta" : (sub["beta"] - sub2["beta"])
               }
    
    change = sum([abs(subgrad["alpha"][f]) for f in subgrad["alpha"]])    


    

    print "CHANGE IS %s"%change

    if self.old_dual[-1] > self.old_primal[-1] + 0.001 : # self.lowest_primal + 0.001:
      # bad news, need more exact solution
      pass
      #self.s1.bad_approx()
      #self.s2.bad_approx()
      #return True

    if change == 0.0 and \
       self.old_primal[-1] > (self.lowest_primal + 1e-5):
      self.s1.bad_approx()
      self.s2.bad_approx()
      return True
    self.update_weights(subgrad)
    return (change <> 0.0 and self.nround < 200)

  def run(self):
    while self.run_one_round():
      pass




class LMExtractor(object):
  def __init__(self, lm, weight):
    self.counter = 0
    self.memo = {}
    self.lm = lm
    self.lm_weight = weight
    self.states = {}
    self.fst = openfst.StdVectorFst()
    self.states_by_id = {}
    self.edge_to_states = {}
    self.states_to_edge = {}
    self.output_to_states = []
    self.states_to_output = {}
    
  def extract(self, forest):
    self.memo = {}
    self.states = {}
    initial= self.create_state("<s>")
    initial.make_initial()
    _,end = self.extract_lm(forest.root, set([initial]))


    #last = self.create_state("</s>")
    #last.make_final()
    
    #self.fst.AddArc(initial.id, openfst.epsilon, initial.id, 1.0, last.id)    
    for n in end:
      n.make_final(self.score(n.word, "</s>"))

    # for _,state in self.states.iteritems():
#       for edge, cost in state.edges.iteritems():
#         print "%s %s %s %s"%(state.id, self.states[edge].id, 0, cost) 

#     print "%s %s"%(last.id, 0)


    for _,state in self.states.iteritems():
      #print state.name
      for edge, cost in state.edges.iteritems(): pass
        #print "\t%s %s %s"%(state.name, self.states[edge].name, cost) 
    #print

    return self.fst, self.states_by_id, self.edge_to_states

  def score(self, w1, w2):
    #print w1, w2
    #print self.lm_weight
    #print self.lm.word_prob_bystr(strip_lex(w2), strip_lex(w1))
    return self.lm_weight * self.lm.word_prob_bystr(strip_lex(w2), strip_lex(w1))

  def create_state(self, word, edge = None):
    
    newstate= LMState(self.fst, word, self.counter, self.output_to_states, self.states_to_output)
    self.states_by_id[newstate.fst_id] = word
    if edge <> None:
      newstate.edge = edge
      self.edge_to_states.setdefault(edge, [])
      self.edge_to_states[edge].append(newstate.fst_id)
      self.states_to_edge[newstate.fst_id] = edge
    self.states[newstate.name] = newstate
    self.counter += 1
    return newstate

  def extract_lm(self, node, node_previous_states):
    if self.memo.has_key(node):
      start, end = self.memo[node]
      for s in start:
        for n in node_previous_states:
          n.add_edge(s, self.score(n.word, s.word))
      return start, end

    beginning_states = set()
    end_states = set()

    for edge in node.edges:
      rhs = edge.rule.rhs
      previous_states = node_previous_states # shallow copy
      
      for i,sym in enumerate(rhs):
        if is_lex(sym):
          new_state = self.create_state(sym, edge.position_id)
          for state in previous_states:
            state.add_edge(new_state, self.score(state.word, new_state.word))
          previous_states = set([new_state])
          if i == 0:
            beginning_states.add(new_state)
        else: # it's a symbol
          pos = get_sym_pos(sym)
          prev, ending = self.extract_lm(edge.subs[pos], previous_states)
          for state in previous_states:
            for next in prev:
              state.add_edge(next, self.score(state.word, next.word))
          previous_states = ending
      end_states |= previous_states   

    self.memo[node] = beginning_states, end_states
    return beginning_states, end_states


          
        

#-----------------------------------------------------------
def read_vec(f):
  v = Vector()
  for l in open(f):
    a,b = l.strip().split("\t")
    v[a] = float(b)
  return v


def print_fst(fst, table, states):
  total = 0.0
  for i in range(fst.NumStates()):
    print "*",i
    #if states and table.has_key(i):
    #  print table[i]
    
    for j in range(fst.NumArcs(i)):
      total += fst.GetWeight(i,j)
      print "\t",i, j, fst.GetInput(i,j),fst.GetOutput(i,j),fst.GetWeight(i,j),fst.GetNext(i,j)
      if states:
        if fst.GetOutput(i,j) == 0:
          continue
          if fst.GetOutput(fst.GetNext(i,j),0) == 0 or fst.NumArcs(fst.GetNext(i,j)) == 0:
            
            print "\t", "warning double epsilon"
          else:
            print "\t", "Next", table[fst.GetOutput(fst.GetNext(i,j),0)]
        elif fst.GetOutput(i,j) == FSA.RHO:
          print "\t", "RHO"
        else:
          print "\t",table[fst.GetOutput(i,j)]
  print "Total Cost: %s" % total


  

if __name__ == "__main__":



    from ngram import Ngram
    from model import Model
    from forest import Forest

    flags.DEFINE_integer("beam", 100, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 0, "debug level")
    flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost>)")
    flags.DEFINE_boolean("cube", True, "using cube pruning to speedup")
    flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")
    flags.DEFINE_integer("ratio", 3, "the maximum items (pop from PQ): ratio*b", short_name="r")
    flags.DEFINE_boolean("extract", False, "use extractor")
    flags.DEFINE_boolean("dual_fst", False, "DD using fst")
    flags.DEFINE_boolean("dual_fst2", False, "DD using fst")
    
    flags.DEFINE_boolean("shift_weights", False, "put the weights all on the lm")

    flags.DEFINE_string("alpha", None, "the maximum items (pop from PQ): ratio*b")
    flags.DEFINE_string("beta", None, "the maximum items (pop from PQ): ratio*b")
  
    flags.DEFINE_string("lm_fsa", None, "the language model FSA")
    flags.DEFINE_string("lm_symbol", None, "the language model symbol table")
    flags.DEFINE_string("lm_heuristic", None, "the language model heuristic table")
    
    

    argv = FLAGS(sys.argv)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram()
    
      
    f = Forest.load("-", is_tforest=True, lm=lm)
    original_weights = weights.__copy__()

    new_weights = Vector()

    # length penalty
    length_weight = weights["text-length"]
    #weights["text-length"] = 0.0
    #del weights["text-length"]
    
    lm_fsa = openfst.Read(FLAGS.lm_fsa)    
    


    lm_uni_heuristic = {}
    lm_uni_heuristic_set = {}
    #gc.disable()
#     for i,l in enumerate(open(FLAGS.lm_heuristic)):
#       if i %10000 ==0 : print i
#       tmp = l.strip().split()
#       word_pair = (tmp[1], tmp[2])
#       lm_uni_heuristic.setdefault(tmp[0], {})
#       lm_uni_heuristic_set.setdefault(tmp[0], set())
#       lm_uni_heuristic[tmp[0]][word_pair] = float(tmp[3])
#       lm_uni_heuristic_set[tmp[0]].add(word_pair)

    start = time.time()
    #heu_handle = open(FLAGS.lm_heuristic, 'rb')
    #(lm_uni_heuristic, lm_uni_heuristic_set) = cPickle.load(heu_handle)
    print "Time was ", time.time() - start 

    #gc.enable()
    



    for i, forest in enumerate(f, 1):
      if FLAGS.dual_fst2:
        print "Forest ", i
        forest.number_nodes()
        if len(forest) > 15: continue
        #for node in forest :
        #  for edge in node.edges:
        #    print edge.position_id


        s_table = openfst.SymbolTable.ReadText(FLAGS.lm_symbol)
        
        print "Extracting Tree"
        ex = tree_extractor.NodeExtractor(False, s_table, 0.0)
        fsa = ex.extract(forest)

        # build one with a unigram for pruning
        no_uni_ex = tree_extractor.NodeExtractor(False, s_table, 0.0)
        no_uni_fsa = no_uni_ex.extract(forest)



        # for debugging
#         tmp1 = {}
#         tmp2 = {}
#         for word in ex.real_word:
#           if lm_uni_heuristic.has_key(word):
#             tmp1[word] = lm_uni_heuristic[word]
#             tmp2[word] = lm_uni_heuristic_set[word]
#         hand = open("example/heuristic.first", 'wb')
#         cPickle.dump((tmp1, tmp2), hand) 
#         exit()
        
        print "Extracting LM"
        #lm_ex = tree_extractor.LMExtractor(lm, original_weights["lm"], ex.s_table)# , length_weight)
        #lm_fsa = lm_ex.extract(ex.words, ex.nt_states)


        lm_fsa.SetInputSymbols(ex.s_table)
        lm_fsa.SetOutputSymbols(ex.s_table)



        count_ex = tree_extractor.TreeCounterFSA(ex.s_table)
        def count_nodes(n):
          t = 0 
          e = n.edges[0]
          t = e.rule.tree_size()
          for s in e.subs:
            t += count_nodes(s)
          return t
        count = count_nodes(forest.root)
        count_fsa = count_ex.extract(count, ex.words, ex.nt_states, 0.0, new_weights, 0.0)
        count_step_fsa = count_ex.extract(count, ex.words, ex.nt_states, 1.0, new_weights, 0.0)

        #count_fsa.Write("/tmp/count.fsa") 
        print "FOREST SIZE", count
        
        table = dict([(output, word) for (word, output) in ex.words] +[(output, word) for (word, output) in ex.nt_states])

        #print_fst(count_fsa, table, True)
        #print_fst(lm_fsa, table, True)
        #print_fst(lm_fsa, table, True)
        #node_map = dict_multi([(nt[0].position_id, output) for (nt, output) in ex.nt_states] )
        #for n in node_map:
          #assert len(node_map[n]) == 2
        #print node_map
        word_map = dict_multi(#[(int(word[0].split("+++")[0]), output) for (word, output) in ex.word_set.iteritems()] +
                              [(int(word[0].split("+++")[-2]), output) for (word, output) in ex.nt_states] )
        
        minimize = True

        openfst.RmEpsilon(fsa)
        #openfst.RmEpsilon(lm_fsa)
        if minimize:
        
          det_fsa = openfst.StdVectorFst()
          #min_lm_fsa = openfst.StdVectorFst()
          #det_lm_fsa = openfst.StdVectorFst()
          #openfst.Prune(lm_fsa, 10.0)  
          
          #openfst.Determinize(lm_fsa, det_lm_fsa)

          print "Minimizing fsa"
          openfst.Determinize(fsa, det_fsa)
          openfst.Minimize(det_fsa)
          print "Minimizing lm"
          #openfst.Minimize(det_lm_fsa)
          #openfst.Prune(det_lm_fsa, 5.0)
          #openfst.Minimize(det_lm_fsa)
        else:
          
          det_fsa = fsa
          
        openfst.ArcSortInput(fsa)
        openfst.ArcSortInput(lm_fsa)
        det_lm_fsa = lm_fsa
        print "Intersecting %s %s %s" % (det_fsa.NumStates(), det_lm_fsa.NumStates(), count_fsa.NumStates()) 
        #openfst.Prune(det_lm_fsa, 3.0)
        fsa2 = openfst.StdVectorFst()
        fsa3 = openfst.StdVectorFst()

        print "Intersect 1"
        #FSA.Intersect(det_fsa, det_lm_fsa, fsa2)
        fsa2 = FSA.rho_compose(det_fsa, False, det_lm_fsa, True, True)
#        tree_count_fsa = FSA.rho_compose(det_fsa, False, count_fsa, True, True)

        shortest = openfst.StdVectorFst()
#        openfst.ShortestPath(det_fsa, shortest, 1)
#        openfst.TopSort(shortest)
        #shortest.Write("/tmp/short.fsa")
        #FSA.print_fst(shortest)
        #fsa2 = det_fsa
        #fsa2 = det_fsa
        #print "Size %s" %(fsa2.NumStates())
        #openfst.ArcSortInput(det_lm_fsa)
        #fsa2 = FSA.rho_compose(det_fsa, False, det_lm_fsa, True)

        # opts = openfst.StdRhoComposeOptions()
#         opts.gc = False
#         fsa2 = openfst.StdComposeFst(det_lm_fsa, det_fsa, opts)

        #fsa2 = det_fsa
        

        bests = {}
        totals = {}

        if minimize:
          print "Minimizing FSA2"
          #det_fsa2 = openfst.StdVectorFst()

          #det_fsa2 = fsa2
          det_fsa2 = openfst.StdVectorFst()
          openfst.Determinize(fsa2, det_fsa2)
          openfst.Connect(det_fsa2)
          #print "LM Original Inter size", det_fsa2.NumStates()

          openfst.Minimize(det_fsa2)
          #openfst.Push(det_fsa2)
          #openfst.Minimize(det_fsa2)
          #openfst.Prune(det_fsa2, 5.0)

          
#           leading = {} 
#           for i in range(det_fsa2.NumStates()):
#             for j in range(det_fsa2.NumArcs(i)):
#               output = det_fsa2.GetOutput(i,j)
#               next = det_fsa2.GetNext(i,j)
#               leading.setdefault(next, set())
#               leading[next].add(output)

#           for i in range(det_fsa2.NumStates()):
#             for j in range(det_fsa2.NumArcs(i)):
#               output = det_fsa2.GetOutput(i,j)
#               if output < 3000000 and output> 0:
#                 #assert len(leading[i]) == 1
#                 #ls = tuple(leading[i])[0]
#                 for l in leading[i]: 
#                   #bests[l] = min(bests.get(l, 1e90), det_fsa2.GetWeight(i,j))
#                   bests[l] = bests.get(l, 0.0) + det_fsa2.GetWeight(i,j)
#                   totals[l] = totals.get(l, 0) + 1

#           for l in bests:
#             bests[l] = bests[l] / float(totals[l])
            
          #print "LM Inter size", det_fsa2.NumStates()
          
          det_tree_count_fsa = openfst.StdVectorFst()
#          
#          openfst.Minimize(det_tree_count_fsa)
#          openfst.Connect(det_tree_count_fsa)

          #openfst.RmEpsilon(count_fsa)
          #openfst.Minimize(count_fsa)        
          print "ArcSortInput"
          openfst.ArcSortInput(count_fsa)
          #openfst.ArcSortInput(count_fsa1)
        else:
          det_fsa2 = fsa2
        #print_fst(fsa, table, True)
        #det_fsa3 = FSA.rho_compose(fsa2, True, count_fsa, True)

        
        #openfst.RmEpsilon(det_fsa2)
        #print "Size of 2 %s %s"%(det_fsa2.NumStates(), det_tree_count_fsa.NumStates())
        #original_size = det_fsa2.NumStates()
        fsa3 = FSA.rho_compose(det_fsa2, False, count_fsa, True, True)
        #fsa3 = openfst.StdVectorFst()
        #openfst.Intersect(det_fsa2, det_tree_count_fsa, fsa3)
        #fsa3 = FSA.rho_compose(det_fsa2, False, count_fsa, True, False)
        
        #fsa3 = det_fsa2
        
        #openfst.Intersect(count_fsa, det_fsa2, fsa3)        
        #fsa3 = det_fsa2
        #fsa3 = det_fsa2d
        
        #print "Size %s" %fsa3.NumStates()

        if  minimize:
          print "minimize fsa3"
          #det_fsa3 = openfst.StdVectorFst(fsa3)
          print "determine"
          #rev_fsa3 = openfst.StdVectorFst()
          #openfst.Reverse(det_fsa3, rev_fsa3)
          #det_fsa3 = fsa3
          #openfst.RmEpsilon(fsa3)
          det_fsa3 = fsa3
          #openfst.Determinize(fsa3, det_fsa3)
          #det_fsa3 = fsa3
          #det_fsa3 = openfst.StdVectorFst(fsa3)
          #openfst.Connect(det_fsa3)
          #
          #openfst.Prune(det_fsa3, 5.0)  
          #openfst.Minimize(det_fsa3)
          
          print >>sys.stderr, "Removing epsilons"
          #openfst.RmEpsilon(det_fsa3)
          #openfst.Prune(det_fsa3, 10.0)
        else:
          det_fsa3 = fsa3
        #print "done minimizing", det_fsa3.NumStates()  
        #openfst.ArcSortInput(det_fsa3)
        #openfst.Prune(det_fsa3, 5.0)

        #print "Final Size %s" %det_fsa3.NumStates()
        # DEBUGGING
        #shortest = openfst.StdVectorFst()

        #openfst.ShortestPath(det_fsa3, shortest, 1)

        print "Shortest Path Done"
        openfst.TopSort(shortest)
        print "Top Sorting"
        #check_fst(shortest, det_lm_fsa, table)

        #short = openfst.StdVectorFst()
        #openfst.Intersect(lm_fsa, shortest, short)
        #print "Short length", short.NumStates()
        #print_fst(shortest, table, True)
        #print_fst(det_fsa3, table, True)
        
        #det_fsa3.Write("/tmp/final.fsa")
        #openfst.TopSort(det_fsa3)
        lm_decoder = decoders.LMDecoderFST(det_fsa3, det_fsa, lm_fsa, count_fsa, table, lm, original_weights['lm'], False, ex.s_table, count)
        lm_decoder.original_size = 10000000 # int((original_size * count_fsa.NumStates()))
        print "Set Heuristic"

        uni_ex = tree_extractor.NodeExtractor(False, s_table, 0.0)
        #uni_ex.set_uni_model(lm, lm_uni_heuristic, lm_uni_heuristic_set, original_weights["lm"], ex.real_word)
        uni_ex.set_uni_model(lm, original_weights["lm"])

        #uni_ex.set_uni_model(lm, bests, original_weights["lm"])
        uni_fsa = uni_ex.extract(forest)


        
        lm_decoder.set_heuristic_fsa(no_uni_fsa, uni_fsa, count_fsa, count_step_fsa, ex.nt_states, ex.s_table.AvailableKey())
        
                                  #forest.len, table.keys(), table, extract.edge_to_states, extract.states_to_edge, extract.output_to_states,extract.states_to_output, lm, weights["lm"])
        #score, words = lm_decoder.decode()
        tree_decoder = decoders.TreeDecoder(forest, weights, word_map, table, ex.word_set, ex.s_table)
        

        dual = SimpleDual(tree_decoder, lm_decoder, False)
        dual.should_print = False

        #start_vec = Vector()
        #for vals in table:
        #   start_vec["UNI" + str(vals)] = -1.0
           
        #dual.weights["alpha"] = start_vec
        
        #dual.send_weights({"alpha":start_vec, "beta":Vector()})
        print dual.run()

      else:
        treedecode = TreeDecoder(forest, weights)
        lmdecode = LMDecoder(forest, lm, weights["lm"])
        dual = SimpleDual(treedecode, lmdecode)

        if FLAGS.alpha:
          dual.set_alpha(read_vec(FLAGS.alpha))
        if FLAGS.beta:
          dual.set_beta(read_vec(FLAGS.beta))
        print dual.run()

