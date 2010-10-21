
from util import *


class Graph(object):
  
  def __init__(self):
    self.nodes = []
    self.first = None
    self.id = 0
  def register_node(self, node):
    self.nodes.append(node)
    self.id += 1 
    return self.id
  
  def __iter__(self):
    for n in self.nodes:
      yield n

class LatNode(object):
  def __init__(self, graph):
    self.edges = []
    self.id = graph.register_node(self)
    
  def add_edge(self, to_node):
    self.edges.append(to_node)

    
class NonTermNode(LatNode):
  def __init__(self, graph, forest_node, dir):
    LatNode.__init__(self, graph)
    self.forest_node = forest_node
    self.dir = dir
  def __str__(self):
    return "NT %s %s"%(self.forest_node, self.dir)

class LexNode(LatNode):
  def __init__(self, graph, lex):
    LatNode.__init__(self, graph)
    self.lex = lex
  def __str__(self):
    return "Lex %s "%(self.lex)
    

class InternalNode(LatNode):
  def __init__(self, graph, rule, pos, name, dir):
    LatNode.__init__(self, graph)
    self.name = name
    self.rule = rule

    self.dir = dir
    if self.dir == UP:
      self.pos = pos +1
    else :
      self.pos = pos
  def __str__(self):
    rhs = self.rule.rhs[:]
    rhs.insert(self.pos, ".")
    rhsstr = " ".join(rhs)
    return "Int %s %s %s"%(rhsstr, self.name, self.dir)




  
class NodeExtractor(object):
  "Class for creating the FSA that represents a translation forest (forest lattice) "

  def __init__(self):
    pass
 
  def extract(self, forest):
    self.memo = {}
    self.graph = Graph()
    (first, _) = self.extract_fsa(forest.root)
    self.graph.first = first
    return self.graph

  def extract_fsa(self, node):
    "Constructs the segment of the fsa associated with a node in the forest"

    # memoization if we have done this node already
    if self.memo.has_key(node.position_id):
      return self.memo[node.position_id]

    # Create the FSA state for this general node (non marked)
    # (These will go away during minimization)
    down_state = NonTermNode(self.graph, node, DOWN)
    up_state = NonTermNode(self.graph, node, UP) 
    self.memo[node.position_id] = (down_state, up_state)
    
    for edge in node.edges:
      previous_state = down_state
    
      rhs = edge.rule.rhs
      
      # always start with the parent down state ( . P )       
      
      for i,sym in enumerate(rhs):

        # next is a word ( . lex ) 
        if is_lex(sym):
          new_state = LexNode(self.graph, sym)

          previous_state.add_edge(new_state)

          # Move the dot ( lex . )
          previous_state = new_state          

        else:
          # it's a symbol

          # local symbol name (lagrangians!)
          pos = get_sym_pos(sym)
          to_node = edge.subs[pos]

          # We are at (. N_id ) need to get to ( N_id .) 

          # First, Create a unique named version of this state (. N_id) and ( N_id . )
          # We need these so that we can assign lagrangians
          local_down_state = InternalNode(self.graph, edge.rule, i, to_node, DOWN)
          local_up_state = InternalNode(self.graph, edge.rule, i , to_node, UP)

          down_sym, up_sym = self.extract_fsa(to_node)
          
          previous_state.add_edge(local_down_state)
          local_down_state.add_edge(down_sym)
          up_sym.add_edge(local_up_state)

          # move the dot
          previous_state = local_up_state
          
      previous_state.add_edge(up_state)
    return self.memo[node.position_id]
