import sys, os
import cPickle
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


def main():
  flags.DEFINE_boolean("unpickle", False, "create a feature map for all features in the data")  
  argv = FLAGS(sys.argv)
  
  (trans_forest_filename, with_feature_filename) = argv[1:]

  if not FLAGS.unpickle:
    
    trans_forests = forest.Forest.load(trans_forest_filename, True)
    outfile = open(with_feature_filename, 'wb')
    for i, tforest in enumerate(trans_forests, 1):
      cPickle.dump(tforest, outfile, -1)
    outfile.close()
  else:
    try:
      f = open(trans_forest_filename, 'rb')
      outfile = open(with_feature_filename, 'w')
      fore = cPickle.load(f)
      while fore:
        fore.dump(outfile)
        fore = cPickle.load(f)
    except EOFError:
      pass

    
if __name__ == "__main__":
  #import cProfile
  #cProfile.run('main()')
  
  main()
