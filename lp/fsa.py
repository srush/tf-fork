from openfst import *
RHO = 2000000 # label that matches anything else


def rho_compose(fst1, has_rho1, fst2, has_rho2, should_cache):
  
  opts = StdRhoComposeOptions()
  if should_cache:
    opts.gc = False
  else:
    opts.gc_limit = 0 
    opts.gc = True
  #
  if has_rho1:
    opts.matcher1 = StdRhoMatcher(fst1, MATCH_INPUT, RHO);
  else:
    opts.matcher1 = StdRhoMatcher(fst1, MATCH_INPUT, kNoLabel);
  if has_rho2:
    opts.matcher2 = StdRhoMatcher(fst2, MATCH_INPUT, RHO)
  else:
    opts.matcher2 = StdRhoMatcher(fst2, MATCH_INPUT, kNoLabel)
  return StdComposeFst(fst1, fst2, opts)

def collect_fst_hash(fst):
  "Create a map from output symbol to edge"
  output_hash = {}
  for i in range(fst.NumStates()): 
    for j in range(fst.NumArcs(i)):
      output_hash.setdefault(fst.GetOutput(i, j), [])
      output_hash[fst.GetOutput(i, j)].append(((i,j), fst.GetWeight(i,j)))
  return output_hash

def update_weight(fsa, state, arc_index, weight):
  iter = StdMutableArcIterator(fsa, state)
  iter.Seek(arc_index)
  old_arc= iter.Value()
  
  arc = StdArc(old_arc.ilabel,old_arc.olabel, weight, old_arc.nextstate)
  iter.SetValue(arc)
  


# class FSAWrapper(object):
#   "FSA with a couple extra thing"

#   def __init__(self):
#     self.fsa = StdVectorFst()

#   def shrink():

  #def AddArc(self, from_state, symbol, weight, to_state):
  #  self.fst.AddArc(from_state.id, StdArc(symbol, symbol, weight, to_state.id))
    
  #def CreateState():

def simple_arc(label, weight, next):
  return StdArc(label, label, weight, next)

class BasicState(object):
  def __init__(self, fsa, name):
    self.fsa = fsa
    self.name = name
    self.state = self.fsa.AddState()
    self.out_state = self.state
    self.in_state = self.state
    
  def add_edge(self, to_state, weight):
    self.fsa.AddArc(self.state, StdArc(epsilon, epsilon, weight, to_state.in_state))
    
  def set_final(self, weight):
    self.fsa.SetFinal(self.state, weight)

  def set_start(self):
    self.fsa.SetStart(self.state)

    
class NamedState(object):
  def __init__(self, fsa, name, id = None, weight = 0.0):
    self.fsa = fsa
    self.name = name
    self.in_state = self.fsa.AddState()
    self.out_state = self.fsa.AddState()
    if id == None:
      self.id = self.out_state
    else:
      self.id = id
    self.fsa.AddArc(self.in_state, StdArc(self.id, self.id, weight, self.out_state))

  def set_weight(self, weight):
    update_weight(self.fsa, self.in_state, 0,  weight)
    

  def add_required_output(self, output):
    "Warning: Call directly after constructor"
    tmp_state = self.fsa.AddState()
    self.fsa.AddArc(tmp_state, simple_arc(output, 0.0, self.in_state))
    self.in_state = tmp_state
    
  def add_edge(self, to_state, weight):
    self.fsa.AddArc(self.out_state, StdArc(epsilon, epsilon, weight, to_state.in_state))
    
  def set_final(self, weight):
    self.fsa.SetFinal(self.out_state, weight)

  def set_start(self):
    self.fsa.SetStart(self.in_state)

def check_fst(path, fst, table):
  total = 0.0
  weight = 0.0
  icur = fst.Start()
  for i in range(len(path)):
    #print "Looking for ",path[i]
    #print "other", icur
    j = 0
    #print path[i], table[path[i]]

    #print "\t",i, j, path.GetInput(i,j),path.GetOutput(i,j),path.GetWeight(i,j),path.GetNext(i,j)
    #path_output = path.GetOutput(i,j)
    #if path.GetOutput(i,j) == 0:
      #print "\t", "Next", table[path.GetOutput(path.GetNext(i,j),0)]
      
    #else:
    #  print "\t",table[path.GetOutput(i,j)]

    found = False
    #print fst.NumArcs(icur)
    for jcur in range(fst.NumArcs(icur)):
      #print icur, jcur, i, path[i], fst.GetOutput(icur, jcur)
      if fst.GetOutput(icur, jcur) == path[i]:
        found = True
        weight += fst.GetWeight(icur, jcur)
        icur = fst.GetNext(icur, jcur)
        break
    if not found:
      print "FAIL"
      print "Looking for ",path[i]
      print "other", icur
      print "Got to", i
      print path[i], fst.InputSymbols().Find(path[i])

      print_fst_state(fst, icur, table)
      assert False,  "%s , %s"%(icur, path[i])
    #print_fst_state(fst, icur, table)
      #return -1000
  #if not fst.IsFinal(icur):
  #  return -1000
  assert fst.IsFinal(icur)
  weight += fst.FinalWeight(icur)
  return weight


def print_fst_state(fst, state, table):
  i = state
  print "*",i
    #if states and table.has_key(i):
    #  print table[i]
    
  for j in range(fst.NumArcs(i)):
    print "\t",i, j, fst.GetInput(i,j),fst.GetOutput(i,j),fst.GetWeight(i,j),fst.GetNext(i,j)
    if fst.GetOutput(i,j) == 0:
      print "\t", "Next", table[fst.GetOutput(fst.GetNext(i,j),0)]
    elif fst.GetOutput(i,j) == RHO:
      print "\t", "RHO"
    else:
      print "\t",table[fst.GetOutput(i,j)]


def count_arcs(fst):
  total  = 0
  for i in range(fst.NumStates()):
    total += fst.NumArcs(i)
  return total


def print_fst(fst):
  total = 0.0
  for i in range(fst.NumStates()):
    print "*",i, fst.IsFinal(i), fst.FinalWeight(i)
    #if states and table.has_key(i):
    #  print table[i]
    
    for j in range(fst.NumArcs(i)):
      total += fst.GetWeight(i,j)
      print "\t",i, j, fst.GetInput(i,j),fst.GetOutput(i,j),fst.GetWeight(i,j),fst.GetNext(i,j)
      if True:
        if fst.GetOutput(i,j) == 0:
          continue
          if fst.GetOutput(fst.GetNext(i,j),0) == 0 or fst.NumArcs(fst.GetNext(i,j)) == 0:
            
            print "\t", "warning double epsilon"
          else:
            print "\t", "Next", fst.InputSymbols().Find(fst.GetOutput(fst.GetNext(i,j)))
        elif fst.GetOutput(i,j) == RHO:
          print "\t", "RHO"
        else:
          print "\t",fst.InputSymbols().Find(fst.GetOutput(i,j))
  print "Total Cost: %s" % total


def make_chain_fsa(syms, s_table):
  sent_fst = StdVectorFst()
  ssent = syms
  sent_fst.AddState()
  for w in ssent:
    sent_fst.AddState()
  for i,w in enumerate(ssent):
    
    sent_fst.AddArc(i, simple_arc(w, 0.0, i+1))
  sent_fst.SetStart(0)
  sent_fst.SetFinal(len(ssent), 0.0)

  sent_fst.SetInputSymbols(s_table)
  sent_fst.SetOutputSymbols(s_table)

  return sent_fst

def get_weight(fst):
  total = 0.0
  TopSort(fst)
  for i in range(fst.NumStates()):    
    for j in range(fst.NumArcs(i)):
      total += fst.GetWeight(i,j)
  last = fst.NumStates()-1
  total += fst.FinalWeight(last)
  return total
