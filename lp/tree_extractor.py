import sys
import fsa
import openfst
from openfst import *
sys.path.append('../Features/')
sys.path.append('Features/')
from ruletree import RuleTree
import sys
from util import *
def is_lex(word):
  return word[0]=="\""

def strip_lex(word):
  if word == "\"": return word
  elif word == "\"\\\"\"": return "\""
  return word.strip("\"")

def super_strip(word):
  return strip_lex(word.split("+++")[0])

def get_sym_pos(word):
  assert word[0] == "x"
  return int(word[1:])


# class LMNodeExtractor(object):
#   def __init__(self):
#     self.words = []
#     self.nt_states = []
#     self.word_set = set()
#   def extract(self, forest):

#     self.fsa = openfst.StdVectorFst() 

#     # memoize the state at each node
#     self.memo = {}
    
#     self.states = {}

#     # dummy start state
#     start, end = self.extract_fsa(forest.root)
#     start.set_start()
#     end.set_final(0.0)


#     return self.fsa

#   def create_state(self, word, is_word):    
#     newstate= fsa.NamedState(self.fsa, word)
#     assert word not in self.word_set

#     if is_word:
#       self.words.append((word, newstate.id))
#       self.word_set.add(word)
#     else :
#       self.nt_states.append((word, newstate.id)) 

#     return newstate

#   def extract_fsa(self, node):
#     # memoization
#     if self.memo.has_key(node.position_id):
#       return self.memo[node.position_id]
     
#     down_state = self.create_state((node, DOWN), False)
#     up_state = self.create_state((node, UP), False)
  
#     self.memo[node.position_id] = (down_state, up_state)

#     for edge in node.edges:
#       rhs = edge.rule.rhs
#       previous_state = down_state
      
#       for i,sym in enumerate(rhs):
#         if is_lex(sym):
#           new_state = self.create_state(sym+"+++"+str(edge.position_id)+"+++"+str(i), True)
#           previous_state.add_edge(new_state, 0.0)
#           previous_state = new_state          
#         else: # it's a symbol

#           pos = get_sym_pos(sym)
#           down_sym, up_sym = self.extract_fsa(edge.subs[pos])
#           previous_state.add_edge(down_sym, 0.0)
#           previous_state = up_sym
          
#       previous_state.add_edge(up_state, 0.0)
#     return self.memo[node.position_id]

SRC_NODE = 3000000
PRE_WORD = 3000001

class NodeExtractor(object):
  "Class for creating the FSA that represents a translation forest (forest lattice) "

  def __init__(self, unique_words, s_table, length_penalty):
    self.words = []
    self.nt_states = []
    self.word_set = {}
    self.edge_set = []
    self.unique_words = unique_words
    self.s_table = s_table
    self.unigram_on = False
    self.length_penalty = length_penalty
    self.real_word = set()
    #self.s_table.AddSymbol("*RHO*", fsa.RHO)
    #self.s_table.AddSymbol("*EPS*", openfst.epsilon)
    #self.s_table.AddSymbol("*SRC*", SRC_NODE)
  def extract(self, forest):
    
    self.fsa = openfst.StdVectorFst() 

    # memoize the state at each node
    self.memo = {}
    self.states = {}

    start, end = self.extract_fsa(forest.root)

    start.set_start()
    end.set_final(0.0)
    self.fsa.SetInputSymbols(self.s_table)
    self.fsa.SetOutputSymbols(self.s_table)

    print "Number of unique words", len(self.real_word)

    return self.fsa

  def set_uni_model_bak(self, orig_lm, lm, lm_set, lm_weight, word_set):
    #self.lm = lm
    self.lm_weight = lm_weight
    self.unigram_on = True
    self.lm = {}
    self.lm_choice = {}
    word_pair_set = set()
    for w1 in word_set:
      for w2 in word_set:
        word_pair_set.add((w1, w2))

    for word in word_set:
      self.lm[word] = -1e90
      valid_words = word_pair_set & lm_set.get(word, set())
      if len(valid_words)== 0:
        self.lm[word] = orig_lm.word_prob_bystr(word, "")
        self.lm_choice[word] = "fail"
      else:
        for pre_word in valid_words:
          if lm[word][pre_word] > self.lm[word]:
            self.lm[word] = lm[word][pre_word]
            self.lm_choice[word] = pre_word
      print word, self.lm_choice[word], self.lm[word],  orig_lm.word_prob_bystr(word, "")
    

  def set_uni_model_bak2(self, orig_lm, bests, lm_weight):
    #self.lm = lm
    self.lm_weight = lm_weight
    self.unigram_on = True
    self.lm = {}
    self.lm = bests
    for s in bests:
      word = self.s_table.Find(s)
      #self.lm[word] = bests[s]
      print >> sys.stderr, word, s, self.lm[s], lm_weight * orig_lm.word_prob_bystr(word, "")

  def set_uni_model(self, orig_lm, lm_weight):
    #self.lm = lm
    self.lm_weight = lm_weight
    self.unigram_on = True
    self.lm = orig_lm


  def score_uni(self, w):
    #assert self.lm_weight * self.lm.word_prob_bystr(super_strip(w), '') >= 0.0
    return self.lm_weight * self.lm.word_prob_bystr(super_strip(w), '')
    #return self.lm_weight *
    #return self.lm[super_strip(w)]

  def score_uni_edge(self, sym):
    print self.s_table.Find(sym)
    return self.lm.get(sym, 1e10)

  def create_state(self, word, is_word, extra=None):
    "create the fsa states associated with nt"
    if is_word and self.unigram_on:
      weight = self.score_uni(word)  + self.length_penalty
    else:
      weight = 0.0 + self.length_penalty
      
    if is_word and self.word_set.has_key(word):
      other_state = self.word_set[word]
      newstate= fsa.NamedState(self.fsa, word, other_state.id, weight=weight)
    else:
      if is_word:
        self.real_word.add(super_strip(word))
        id = self.s_table.Find(super_strip(word))
        print word, id
        if id == -1:
          id = self.s_table.Find("<unk>")
        newstate= fsa.NamedState(self.fsa, word, id, weight=weight)
        
      else:
        id = self.s_table.Find(str(word))
        if id == -1:
          id = self.s_table.AddSymbol(str(word))
        newstate= fsa.NamedState(self.fsa, word, id)
    #assert word not in self.word_set

      if is_word:
        self.words.append((word, newstate.id))
        self.word_set[word] = newstate
      else :
        self.nt_states.append((word, newstate.id)) 
        newstate.add_required_output(SRC_NODE)

    if is_word:
      wextra = (word+extra,DOWN)
      id = self.s_table.Find(str(wextra))
      if id == -1:
        id = self.s_table.AddSymbol(str(wextra))
      self.nt_states.append((wextra, id))

      #if self.unigram_on:
      #  newstate.set_weight(self.score_uni_edge(id))

      newstate.add_required_output(id)
      newstate.add_required_output(PRE_WORD)
      self.word_set[wextra] = id 
    return newstate

  def extract_fsa(self, node):
    "Constructs the segment of the fsa associated with a node in the forest"
    # memoization if we have done this node already
    if self.memo.has_key(node.position_id):
      return self.memo[node.position_id]

    # Create the FSA state for this general node (non marked)
    # (These will go away during minimization)
    down_state = fsa.BasicState(self.fsa, (node, DOWN)) #self.create_state((node, DOWN), False)
    up_state = fsa.BasicState(self.fsa, (node, UP)) #self.create_state((node, UP), False)
    self.memo[node.position_id] = (down_state, up_state)
    

    for edge in node.edges:
      previous_state = down_state
      # start experiment
      # Enumerate internal (non-local terminal) nodes on left hand side 
      lhs = edge.rule.lhs
      
      lhs_treelet = RuleTree.from_lhs_string(edge.rule.lhs)
      def non_fringe(tree):
        "get the non terminals that are not part of the fringe"
        if not tree.subs:
          return []
        return [tree.label] + sum(map(non_fringe, tree.subs), [])
      lhs_internal = sum(map(non_fringe,lhs_treelet.subs), [])
      print "INTERNAL", lhs_internal
      for i, nt in enumerate(lhs_internal):
        extra = "+++"+str(edge.position_id)+"+++"+str(i-10)
        fake_down_state = self.create_state((str(nt)+extra, DOWN), False)
        fake_up_state = self.create_state((str(nt)+extra, UP), False)        
        previous_state.add_edge(fake_down_state, 0.0)
        fake_down_state.add_edge(fake_up_state, 0.0)
        previous_state = fake_up_state
      
      # end experiment


      rhs = edge.rule.rhs
      
      # always start with the parent down state ( . P ) 
      
      nts_num =0 
      for i,sym in enumerate(rhs):
        extra = "+++"+str(edge.position_id)+"+++"+str(i)

        # next is a word ( . lex ) 
        if is_lex(sym):

          if self.unique_words:
            new_state = self.create_state((sym+extra, DOWN), True)

          else:
            new_state = self.create_state(sym, True, extra)

          previous_state.add_edge(new_state, 0.0)

          # Move the dot ( lex . )
          previous_state = new_state          
        else:
          # it's a symbol

          # local symbol name (lagrangians!)
          to_node = edge.subs[nts_num]
          nts_num += 1
          
          # We are at (. N_id ) need to get to ( N_id .) 

          # First, Create a unique named version of this state (. N_id) and ( N_id . )
          # We need these so that we can assign lagrangians
          local_down_state = self.create_state((str(to_node)+extra, DOWN), False)
          local_up_state = self.create_state((str(to_node)+extra, UP), False)

          down_sym, up_sym = self.extract_fsa(to_node)
          
          previous_state.add_edge(local_down_state, 0.0)
          local_down_state.add_edge(down_sym, 0.0)
          up_sym.add_edge(local_up_state, 0.0)

          # move the dot
          previous_state = local_up_state


      # for nt in lhs_internal:
#         extra = "+++"+str(edge.position_id)+"+++-1"
#         local_up_state = self.create_state((str(nt)+extra, UP), False)        
#         previous_state.add_edge(local_up_state,0.0)
#         previous_state = local_up_state


      #extra = "+++"+str(edge.position_id)+"+++"+str(i + 1)
      
      #end_hyp_edge = self.create_state(("edge"+extra, (edge.rule.tree_size(), edge.fvector["text-length"], edge.fvector) ), False)
      #previous_state.add_edge(end_hyp_edge, 0.0)
      #previous_state = end_hyp_edge

      # Finish by connecting back to parent up
      previous_state.add_edge(up_state, 0.0)
    return self.memo[node.position_id]



class LMExtractor(object):
  """
  This class creates a vanilla trigram LM fsa from a bag of words 
  """
  
  def __init__(self, lm, lm_weight, s_table):
    "A language model a a relative weighting"
    self.lm = lm
    self.lm_weight = lm_weight
    self.s_table = s_table
  def score(self, w1, w2, w3):
    "Score a trigram"
    lm = self.lm.word_prob_bystr(strip_lex(w3), strip_lex(w1) + " " + strip_lex(w2))
    #print >>sys.stderr, strip_lex(w3), strip_lex(w1) + " " + strip_lex(w2),  lm, self.lm_weight * lm
    
    return self.lm_weight * lm
  
  def extract(self, words, ignore_out):
    print >>sys.stderr, "LM Extractor %s %s" % (len(words), len(ignore_out)) 

    # the fsa
    self.fsa = openfst.StdVectorFst() 

    initial = self.fsa.AddState()
    self.fsa.SetStart(initial)
    states = {}


    # RHO means anything allowed
    self.fsa.AddArc(initial, fsa.simple_arc(fsa.RHO, 0.0, initial))

    # initial arcs
    for (word1, output) in words:

      # add a starting arc to each word
      score = self.score("<s>", "<s>", word1)
      i_pair = ("<s>", word1)
      #print >>sys.stderr, i_pair, score, output

      states[i_pair] = self.fsa.AddState()
      self.fsa.AddArc(initial, fsa.simple_arc(output, score, states[i_pair]))
      self.fsa.AddArc(states[i_pair], fsa.simple_arc(fsa.RHO, 0.0, states[i_pair]))


      for (word2, output) in words:
        # add a second starting arc to each word pair
        score = self.score("<s>", word1, word2)
        pair= (word1, word2)
        states[pair] = self.fsa.AddState()
        self.fsa.AddArc(states[i_pair], StdArc(output, output, score, states[pair])) 


    print >>sys.stderr, "starting inner"
    for (word1,_) in words:
      for (word2,_) in words:
        # connect each word pair
        pair = (word1, word2)
        word_state = states[pair]
        for (to_word, word_output) in words:
          to_word_state = states[word2, to_word]
          score = self.score(word1, word2, to_word)
          self.fsa.AddArc(word_state, fsa.simple_arc(word_output, score, to_word_state))

        # connect to the final state
        self.fsa.AddArc(word_state, fsa.simple_arc(fsa.RHO, 0.0, word_state))
        end_score = self.score(word1, word2, "</s>")
        self.fsa.SetFinal(word_state, end_score)
    print >>sys.stderr, "done inner"
    # create symbol table for reading
    
    print >>sys.stderr, "Done"
    self.fsa.SetInputSymbols(self.s_table)
    self.fsa.SetOutputSymbols(self.s_table)
    
    return self.fsa



# class LMExtractor(object):
#   def __init__(self, lm, weight, text_weight):
#     self.counter = 0
#     self.memo = {}
#     self.lm = lm
#     self.lm_weight = weight
#     self.text_weight = text_weight
#     self.states = {}
#     self.fsa = openfst.StdVectorFst()
    
#   def extract(self, forest, words, ignore_out):
#     self.words = dict(words)
#     self.ignore_out = ignore_out
    
#     self.memo = {}
#     self.states = {}
#     #initial= self.create_state("<s>")
#     initial= fsa.BasicState(self.fsa, "<s>")
#     initial.set_start()
#     self.fsa.SetStart(initial.state)
#     self.add_blank_edges(initial)

#     begin,end = self.extract_lm(forest.root)

#     for p in begin:
#       initial.add_edge(p, self.score(initial.name, p.name))

#     for n in end:
#       n.set_final(self.score(n.name, "</s>"))
#       #n.set_final(0.0)
      
#     return self.fsa

#   def score(self, w1, w2):
#     #print super_strip(w1), super_strip(w2)
#     extra = self.text_weight if w2 <> "</s>" else 0.0
#     lm = self.lm.word_prob_bystr(super_strip(w2), super_strip(w1))
#     #print super_strip(w1), super_strip(w2), lm 
#     return self.lm_weight * lm + extra

#   def add_blank_edges(self, state):
#     # RHO eats everything
#     #self.fsa.AddArc(state.out_state, StdArc(fsa.RHO, fsa.RHO, 0.0, state.out_state))
#     for (_, ignore) in self.ignore_out:
#       # non-word, stay in the same state
#       self.fsa.AddArc(state.out_state, StdArc(ignore, ignore, 0.0, state.out_state))
#       #self.fsa.AddArc(state.ou_state, StdArc(ignore, ignore, 0.0, state.in_state))

#   def create_state(self, word, edge_pos = None):
#     if edge_pos:
#       newstate= fsa.NamedState(self.fsa, word, self.words[(word, DOWN)])
#     else:
#       newstate= fsa.NamedState(self.fsa, word, openfst.epsilon)
#     self.add_blank_edges(newstate)    
#     return newstate

#   def extract_lm(self, node):
#     if self.memo.has_key(node):
#       start, end = self.memo[node]
#       return start, end

#     beginning_states = set()
#     end_states = set()

#     for edge in node.edges:
#       rhs = edge.rule.rhs
#       previous_states = set() # shallow copy
      
#       for i,sym in enumerate(rhs):
#         if is_lex(sym):
#           new_state = self.create_state(sym+"+++"+str(edge.position_id)+"+++"+str(i), True)
#           for state in previous_states:
#             state.add_edge(new_state, self.score(state.name, new_state.name))
#           previous_states = set([new_state])
#           if i == 0:
#             beginning_states |= previous_states

#         else: # it's a symbol
#           pos = get_sym_pos(sym)
#           prev, ending = self.extract_lm(edge.subs[pos])
          
#           if i == 0:
#             beginning_states |= prev

#           for state in previous_states:
#             for next in prev:
#               state.add_edge(next, self.score(state.name, next.name))
#           previous_states = ending

#       end_states |= previous_states   

#     self.memo[node] = beginning_states, end_states
#     return beginning_states, end_states


class CounterFSA(object):
  def __init__(self): pass

  def extract(self, l, words, ignore_out, length_bonus):

    self.counter_fst = openfst.StdVectorFst()
    first = self.counter_fst.AddState()
    self.counter_fst.SetStart(first)
    self.counter_fst.SetFinal(first, 0.0)
    states = [first]
    print "length limit %s" % (l*2.0 + 10) 
    for i in range(l*2.0 + 10):
      states.append(self.counter_fst.AddState())
      self.counter_fst.SetFinal(states[-1], 0.0)

      for (_, word_output) in words:
        self.counter_fst.AddArc(states[-2], StdArc(word_output, word_output, length_bonus, states[-1]))
      #self.counter_fst.AddArc(states[-2], StdArc(0, 0, 0.0, states[-1]))

      # RHO arc eats anything else
      #self.counter_fst.AddArc(states[-2], StdArc(fsa.RHO, fsa.RHO, 0.0, states[-2]))

      for (_, ignore) in ignore_out:
        # non-word, stay in the same state
        self.counter_fst.AddArc(states[-2], StdArc(ignore, ignore, 0.0, states[-2]))
      
    self.counter_fst.SetNotFinal(states[-1])
    self.counter_fst.SetNotFinal(states[-2])

    return self.counter_fst


class TreeCounterFSA(object):
  def __init__(self, s_table):
    self.s_table = s_table

  def extract(self, original_length, words, ignore_out, step_weight, weights, length_penalty):
    # ignore TOP UP and TOP DOWN
    l = original_length -1
    
    #self.text_weight = text_weight

    # create and initialize the fst
    self.counter_fst = openfst.StdVectorFst()
    first = self.counter_fst.AddState()
    self.counter_fst.SetStart(first)
    states = [first]

    # Counting fsa has 2*l states (up and down for each NT)
    for i in range(2*l+1):
      states.append(self.counter_fst.AddState())

    for i in range(len(states)-1):
      #self.counter_fst.SetFinal(states[-1], 0.0)

      # Don't count words
      self.counter_fst.AddArc(states[i], fsa.simple_arc(fsa.RHO, 0.0, states[i]))
      

      #for (_, word_output) in words:
        #self.fsa.AddArc(first, StdArc(fsa.RHO, fsa.RHO, 0.0, first))
        #self.counter_fst.AddArc(states[i], StdArc(word_output, word_output, 0.0, states[i]))

      # for each non word
      #for ((s, motion), sym) in ignore_out:
        # All non words increase count by 1
        #self.counter_fst.AddArc(states[i], fsa.simple_arc(sym, 0.0, states[i+1]))

      self.counter_fst.AddArc(states[i], fsa.simple_arc(SRC_NODE, step_weight, states[i+1]))
      self.counter_fst.AddArc(states[i], fsa.simple_arc(PRE_WORD, length_penalty, states[i]))
      #self.counter_fst.SetFinal(states[i], 0.0)
#      for (edge_jump, output) in edge_jump:
        
      #self.counter_fst.AddArc(states[-2], StdArc(0, 0, 0.0, states[-1]))

      # RHO arc eats anything else
      #self.counter_fst.AddArc(states[-2], StdArc(fsa.RHO, fsa.RHO, 0.0, states[-2]))

    # Last node is a sink, so second to last is final 
    self.counter_fst.SetFinal(states[-2], 0.0)

    
    #self.counter_fst.SetNotFinal(states[-1])
    #self.counter_fst.SetNotFinal(states[-2])

    self.counter_fst.SetInputSymbols(self.s_table)
    self.counter_fst.SetOutputSymbols(self.s_table)

    return self.counter_fst



