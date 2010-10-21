import sys, os
import gflags as flags
import itertools
from local_features import *
import features
from svector import Vector
from features import FeatureExtractor
sys.path.append('..')
import utility
import forest
FLAGS=flags.FLAGS


def read_features(file):
  feature2ind = {}
  ind2feature = {}
  features = utility.getfile(file, 0)
  for l in features:
    i, feature = l.strip().split("\t")
    feature2ind[feature] = int(i)
    ind2feature[int(i)] = feature
  return (feature2ind, ind2feature)

class FeatureAdder(object):
  def __init__(self, tdm):
    self.tdm = tdm

    # Local (edges)
    self.local_fe = FeatureExtractor(get_all_feature_classes(LocalFeatureClass))


    # Local Node (nodes)
    self.node_fe = FeatureExtractor(get_all_feature_classes(NodeFeatureClass))

    # Local RHS
    rhs_fc  = [fc for fc in get_all_feature_classes(LocalFeatureClass) if fc.feature_side & LocalContext.TARGET ]
    self.rhs_fe = FeatureExtractor(rhs_fc)


  def add_features(self, tforest, just_list = False):
    allfeats = set()
    for node in tforest.nodes.values():
      node_local_context = LocalNodeContext(node, tforest.sent)
      node_features = self.node_fe.extract_all(node_local_context)

      if just_list:
        allfeats |= set([f.split('=')[0] for f in node_features])        
      else:
        #node.fvector = " ".join([f for f in node_features])
        node.fvector = Vector(" ".join([f for f in node_features]))
            
      for edge in node.edges:
        local_context = LocalContext(node, edge, edge.rule, tforest.sent)
        features = self.local_fe.extract_all(local_context)

        #local_context.set_cluster_level(self.tdm, 4)
        #features.extend(self.rhs_fe.extract_all(local_context))
        
        #local_context.set_cluster_level(self.tdm, 6)
        #features.extend(self.rhs_fe.extract_all(local_context))

        if just_list:
          allfeats |= set([f.split('=')[0] for f in features])
        else:
          # hack, add in features
          # edge.fvector = " ".join([f for f in features])
          edge.fvector = Vector(" ".join([f for f in features]))
          edge.rule.fields = Vector(" ".join(features))
    return allfeats  
          

def main():
  flags.DEFINE_boolean("feature_map", False, "create a feature map for all features in the data")  
  argv = FLAGS(sys.argv)
  
  (trans_forest_filename, with_feature_filename) = argv[1:]
  trans_forests = forest.Forest.load(trans_forest_filename, True)
  tdm = TargetDataManager()
  feature_adder = FeatureAdder(tdm)
  outfile = utility.getfile(with_feature_filename, 1)
  allfeats = set()
  allfeats.add("lm1")
  allfeats.add("lm")
  for i, tforest in enumerate(trans_forests, 1):
    print >>sys.stderr, "processed sent %s " % i
    
    if FLAGS.feature_map:
      allfeats |= feature_adder.add_features(tforest, just_list = True)
    else:
      feature_adder.add_features(tforest)
      tforest.dump(outfile)

  if FLAGS.feature_map:
    for i, feat in enumerate(allfeats):
      print >>outfile, str(i)+ "\t" + feat
if __name__ == "__main__":
  #import cProfile
  #cProfile.run('main()')
  
  main()

