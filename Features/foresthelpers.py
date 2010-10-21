import sys, time, math
sys.path.append('..')
from  forest import Forest
from prune import inside_outside
import fast_inside_outside
logs = sys.stderr

def general_prune(forest, node_pruning, edge_pruning, copy=True):
    ''' General version of forest pruning.
        node_pruning and edge_pruning let you specify pruning predicates
    '''

    survived_nodes = set()
    
    def check_subs(edge):
        ''' check if every tail falls in the beam. '''
        for sub in edge.subs:
            if sub not in survived_nodes:
                return False
        return True
    
    start_time = time.time()
    
    new_forest = forest.copy() if copy else forest
    
    oldsize = new_forest.size()
    
    newnodes = {}
    neworder = []

    kleinedges = 0
    for node in new_forest:
        iden = node.iden
        if not node_pruning(node):
            node.edges = [e for e in node.edges if not edge_pruning(e) and check_subs(e)]
            if node.edges == []: ## N.B.: isolated node: unreachable from bottom-up!
                print >> logs, "WARNING: isolated node found!"
                del node
            else:
                survived_nodes.add(node)
                newnodes[iden] = node
                neworder.append(node)                
                #print "Keeping edges %s"%[str(edge.rule) for edge in node.edges ]
        else:
            kleinedges += len(node.edges)
            del node

    del new_forest.nodes
    del new_forest.nodeorder
    
    new_forest.nodes = newnodes
    new_forest.nodeorder = neworder

    new_forest.rehash() ## important update for various statistics
    
    newsize = new_forest.size()
    
    print >> logs, "%s (len: %d), %4d nodes, %5d edges remained. prune ratio = %.1f%%, %.1f%% (%.1f%%) edges/len: %.1f" \
          % (new_forest.tag, len(new_forest.sent),
             newsize[0], newsize[1], 
             (oldsize[0] - newsize[0])*100.0 / oldsize[0], (oldsize[1] - newsize[1])*100.0 / oldsize[1],
             kleinedges*100.0/oldsize[1],
             newsize[1]/len(new_forest.sent))    
    
    print >> logs, "done in %.2lf secs" % (time.time() - start_time)
    return new_forest

def inside_outside_prune(forest, weights, gap = None, ratio = None):
  all_edges = inside_outside(forest, weights)
  threshold = forest.root.merit + (gap if gap is not None else 1e+9)

  if ratio is not None:
      all_edges.sort()
      allowed = int(len(forest.sent) * ratio) # allowed size
      threshold = min(threshold, all_edges[min(allowed, len(all_edges)-1)][0])

  # remove unreachable nodes or nodes below the cutoff
  def node_pruning(node):
      return hasattr(node, "unreachable") or node.merit > threshold

  def edge_pruning(edge):
      return edge.merit > threshold

  return general_prune(forest, node_pruning, edge_pruning, copy = False)

def inside_outside_sum(forest, weights):

  nodes, edges = fast_inside_outside.inside_sum(forest, weights)
  node_outside, node_merit, edge_merit = fast_inside_outside.outside_sum(forest, weights, nodes, edges)
  return node_merit, edge_merit


def collect_marginals(forest, weight):
  "Grab all the expected features from the forest"
  all_edges = inside_outside_sum(forest, weights)
  full_inside = forest.root.merit

  all_feats = Vector()

  for node in forest.nodes:
    all_feats += math.exp(node.merit) * node.fvector
    
  for edge in all_edges :
    all_feats += math.exp(edge.merit) * edge.fvector  


  return all_feats

if __name__ == "__main__":

    # confirm that pruning is working
    import prune
    import gflags as flags
    FLAGS=flags.FLAGS

    # reimplementation of prune.py
    flags.DEFINE_float("prob", None, "score threshold", short_name="p")
    flags.DEFINE_float("ratio", None, "ratio of |hyperedges|/|sent|", short_name="r")
    flags.DEFINE_boolean("oracle", False, "compute oracle after pruning")
    flags.DEFINE_boolean("out", True, "output pruned forest (to stdout)")
    flags.DEFINE_string("suffix", None, "suffix for dumping (1.<suffix>)", short_name="s")
    flags.DEFINE_integer("startid", 1, "start id for dumping")

    from ngram import Ngram # defines --lm and --order    
    from model import Model
    from bleu import Bleu
    argv = FLAGS(sys.argv)

    if FLAGS.prob is None and FLAGS.ratio is None:
        print >> logs, "Error: must specify pruning threshold by -p or ratio by -r" + str(FLAGS)
        sys.exit(1)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram() # if FLAGS.lm is None then returns None
    if lm:
        weights["lm1"] = weights["lm"] * FLAGS.lmratio
    
    onebestscores = 0
    onebestbleus = Bleu()
    myscores = 0
    myoraclebleus = Bleu()    
    
    total_nodes = total_edges = old_nodes = old_edges = 0
    
    for i, forest in enumerate(Forest.load("-", lm=lm), 1):
        if forest is None:
            print
            continue

        #f1 = forest.copy()
        #f2 = forest.copy()
        
        #prune.prune(f1, weights, FLAGS.prob, FLAGS.ratio)
        #inside_outside_prune(f2, weights, FLAGS.prob, FLAGS.ratio)

        #assert f1.size() == f2.size()

        node_merit, edge_merit = inside_outside_sum(forest, weights)
        sum = 0.0
        beta = node_merit[forest.root.position_id]
        print beta
        print node_merit[forest.root.position_id]
        #for score, edge in all_edges: pass
        
          #print math.exp(-score), edge
          #sum += score / beta
          #print score, edge, edge.merit
        for node in forest.nodeorder:
          #print math.exp(-score), edge
          #sum += score / beta
          
          #print node_inside[node.position_id], node_outside[node.position_id],
          print node_merit[node.position_id], node_merit[node.position_id] / beta, node
            #print node.insidepaths, node.outsidepaths, node.insidepaths * node.outsidepaths
            #for edge in node.edges:
            #  print edge, edge.merit/beta
              
        print sum
        


