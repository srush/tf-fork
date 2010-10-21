
from __future__ import division

import sys, os
from math import *
import itertools
import trainer
import distributed_trainer
import math, time
sys.path.append('..')
sys.path.append('../wrappers/')
import oracle
import forest
import gflags as flags
import glob
import utility
import local_features
from evaluator import Evaluator
from cyksearch import CYKDecoder
from ngram import Ngram
from model import Model
from forest import Forest
from svector import Vector
from bleu import Bleu
import prune
from add_features import FeatureAdder
import sgd
import crf as CRF
import gc
import fast_inside_outside
import oracle
import add_features
import distributed_crf as distCRF
from cubepruning import CubePruning, FeatureScorer 
import cPickle
FLAGS = flags.FLAGS

logs = sys.stderr
INF = 1e90

def parse_diff(weights, fv1 , fv2):
  new_fv = fv1 - fv2
  ls = []
  for feat in new_fv:
    
    if new_fv[feat] <> 0.0 and weights[feat] <> 0.0:
      ls.append( (weights[feat], feat, new_fv[feat]))
  ls.sort()

  for weight, feat, diff in ls[0:20]:
    print feat, weight, diff

  ls.reverse()
  print 
  for weight, feat, diff in ls[0:20]:
    print feat, weight, diff

    
def pickle_trans_load(filenames, feature_adder = None, should_prune = False, weights = None, lm = None):  
  for file in filenames:
    try:
      print file
      f = open(file, 'rb')
      fore = cPickle.load(f)
      
      while fore:#forest.Forest.load(file, True, lm=lm):
        if feature_adder:
          feature_adder.add_features(fore)
        
        if should_prune:
          prune.prune(fore, weights, None, 10)
        yield fore
        fore = cPickle.load(f)
    except EOFError:
      pass

def trans_load(filenames, feature_adder = None, should_prune = False, weights = None, lm = None):
  for file in filenames:
    for fore in forest.Forest.load(file, True, lm=lm):
      if feature_adder:
        feature_adder.add_features(fore)
        
      if should_prune:
        prune.prune(fore, weights, None, 10)
      yield fore
      
def write_model(filename, weights):
  for name, weight in weights.iteritems():
    print "%s\t%s"%( name, weight)
    
class TransDecoder(trainer.IDecoder):
  def __init__(self, weights, lm):
    self.lm = lm
    self.weights = weights

    self.prune_train = False
    self.feature_adder = None
    self.training_cache = {}
    self.use_pickle = False
    self.cache_input = False
    
  def get_model_weights(self):
    return self.weights
  
  def set_model_weights(self, new_weights):
    self.weights = new_weights
    self._refresh()
    
  def write_model(self, filename, weights):
    return write_model(filename, weights)
  
  def evalclass(self):
    return Bleu()

  def load(self, filenames):
    if self.use_pickle:
      for f in  pickle_trans_load(filenames, self.feature_adder, self.prune_train, self.weights, self.lm):
        yield f
    elif not self.cache_input:
      for f in  trans_load(filenames, self.feature_adder, self.prune_train, self.weights, self.lm):
        yield f
    else:
      if self.training_cache.has_key(filenames[0]):
        for sent in self.training_cache[filenames[0]]:
          yield sent
      else:
        self.training_cache[filenames[0]] = []
        for f in trans_load(filenames, self.feature_adder, self.prune_train, self.weights, self.lm):
          self.training_cache[filenames[0]].append(f)
          yield f
      
class PerceptronDecoder(TransDecoder):
  def _refresh():
    self.decoder = CYKDecoder(self.weights, self.lm)    

  @staticmethod
  def trim(fv):
    for f in fv:
      if math.fabs(fv[f]) < 1e-3:
        del fv[f]
    return fv


class LocalPerceptronDecoder(PerceptronDecoder):  
  def decode(self, forest, early_stop=False):    
    #decoding
    (score, best_trans, best_fv) = self.decoder.beam_search(forest, b=FLAGS.beam)     
    

    test_bleu = forest.bleu.rescore((best_trans))

    best_bleu = test_bleu
    oracle_fv = best_fv
    for k, (sc, trans, fv) in enumerate(forest.root.hypvec, 1):
      hyp_bleu = forest.bleu.rescore(trans)
      if hyp_bleu > best_bleu:
        oracle_fv = fv
        best_bleu = hyp_bleu

    delta_feats = self.trim(oracle_fv - best_fv)
    test_bleu = forest.bleu.rescore((best_trans))
    #print delta_feats
    return (forest.bleu, delta_feats)



class ChiangPerceptronDecoder(PerceptronDecoder):  
  
  def _add_back_language_model(self, sent):
    s = sent.split()
    score = 0.0
    score += self.lm.word_prob_bystr(s[0], '')
    score += self.lm.word_prob_bystr(s[1], s[0]) if len(s) > 2 else 0.0
    for i in range(2, len(s)):
      #print (s[i-2:i+1])
      score += self.lm.word_prob_bystr(s[i], s[i-2:i])
        
    return score

  def _add_back_uni_language_model(self, sent):
    s = sent.split()
    score = 0.0
    for i in range(len(s)):
      #this = lm.word2index(word)
      score += self.lm.word_prob_bystr(s[i],"")
      
    return score

  def decode(self, forest, early_stop=False):    
    #decoding
    
    oracle_bleu, oracle_trans, oracle_fv, _ = forest.compute_oracle(Vector(), model_weight=0.0, bleu_weight=1.0)

    (score, best_trans, best_fv) = self.decoder.beam_search(forest, b=FLAGS.beam)

    
    
    #self.write_model("", best_fv)
    best_lm = self._add_back_language_model(best_trans)
    #best_lm1 = self._add_back_uni_language_model(best_trans)
    #assert  best_lm == best_fv["lm"], "%s, %s "% (best_lm, best_fv["lm"])
    #assert  best_lm1 == best_fv["lm1"], "%s, %s "% (best_lm1, best_fv["lm1"])

    
    oracle_fv["lm"] = self._add_back_language_model(oracle_trans)
    #print "oracle", oracle_fv["lm1"]
    #oracle_fv["lm1"] = self._add_back_uni_language_model(oracle_trans)


    #print "test score " + str(score)
    #print "dot test " + str(self.weights.dot(best_fv))
    test_bleu = forest.bleu.rescore(best_trans)
    oracle_bleu = forest.bleu.rescore(oracle_trans)


    #print "oracle test " + str(self.weights.dot(oracle_fv))
    #print best_trans
    #print oracle_trans

#     best_bleu = test_bleu
#     oracle_fv = best_fv
#     for k, (sc, trans, fv) in enumerate(forest.root.hypvec, 1):
#       hyp_bleu = forest.bleu.rescore(trans)
#       if hyp_bleu > best_bleu:
#         oracle_fv = fv
#         best_bleu = hyp_bleu

    print >>sys.stderr, "Test: %s \n Oracle: %s"% ( best_trans, oracle_trans)
    print >>sys.stderr, "BLEU Test: %s | Oracle: %s"% ( test_bleu, oracle_bleu)
    print >>sys.stderr, "MODEL Test: %s | Oracle: %s"% ( self.weights.dot(best_fv), self.weights.dot(oracle_fv))

    #delta_feats = self.clip(self.trim(-oracle_fv + best_fv))
    delta_feats = (self.trim(-oracle_fv + best_fv))

    # self.write_model("",best_fv)
#     print " ----------------------"
#     self.write_model("",oracle_fv)
#     
    #self.write_model("", delta_feats)
    #print " ----------------------"  
    #self.write_model("", self.weights)
    #self.write_model("", best_fv)
    #self.write_model("", oracle_fv)

    test_bleu = forest.bleu.rescore((best_trans))
    return (forest.bleu, delta_feats)



class MarginalDecoder(TransDecoder):
  class FeatureAdder(FeatureScorer):
    def add(self, one, two):
      M = max(one[0], two[0])
      m = min(one[0], two[0])
      ret=  M + log(1.0 + exp(m - M))
      return (ret, one[1])

  def load_oracle(self, oracle_filenames):
    for file in oracle_filenames:
      for fore in forest.Forest.load(file, True, lm=self.lm):
        #fore.number_nodes()
        yield fore

  def compute_marginals(self, forest, oracle_forest):
    "computes the marginals of a -lm forest"
    

    #print >> logs, "Example TIME %s"%((end - start))

    
    #oracle_bleu, oracle_trans, oracle_fv, _ = oracle_forest.compute_oracle(Vector(), model_weight=0.0, bleu_weight=1.0)    
    def non_local_scorer(cedge, ders):
      hyp = cedge.assemble(ders)
      return ((0.0, Vector()),  hyp, hyp)
    #decoder = CubePruning(MarginalDecoder.FeatureAdder(self.weights), non_local_scorer, 20, 5, find_min=False)
    #best = decoder.run(forest.root)

#     example_marginals = Vector()
#     total = -INF
#     for i in range(min(200, len(best))):
#       M = max(best[i].score[0], total)
#       m = min(best[i].score[0], total)
#       total =  M + log(1.0 + exp(m - M))
      
#     #print "before"
#     print total
#     for i in range(min(200, len(best))):      
#       #print exp(best[i].score[0] -total)
#       example_marginals +=  exp(best[i].score[0] -total) * best[i].score[1]
#     #print "after"
#     partition = total

    #start = time.time()
    example_marg, partition  = fast_inside_outside.collect_marginals(forest, self.weights)
    #end = time.time()
    #print >> logs, "marg TIME %s"%((end - start))
    
    #print "Best Log Likelihood %s "%(best[0].score[0] - partition)
    #start = time.time()
    #oracle_forest, oracle_item = oracle.oracle_extracter(forest, self.weights, 5, 2, extract=1)
    #end = time.time()
    #print >> logs, "oracle forest %s"%((end - start))

    #start = time.time()
    oracle_marg, oracle_partition  = fast_inside_outside.collect_marginals(oracle_forest, self.weights)
    (oracle_best, oracle_subtree, oracle_best_fv)  = oracle_forest.bestparse(self.weights, use_min=False)
    #end = time.time()
    #print >> logs, "oracle TIME %s"%((end - start))
    #logs.flush()
    #self.write_model("", oracle_marg)

    #print "Best   Score: %s"% best 
    #print "Oracle Score: %s"% (self.weights.dot(oracle_fv))

    #for i in range(5):
      #print "Oracle Trans: %s %s" %(oracle_item[i].full_derivation, oracle_item[i].score)
      #print "Oracle BLEU Score: %s"% (forest.bleu.rescore(oracle_item[i].full_derivation))


#     forest.bleu.rescore(oracle_subtree)
#     print "Oracle Trans: %s %s" %(oracle_subtree, forest.bleu.score_ratio_str())
#     print "Best   Trans: %s"%best[0].full_derivation
#     forest.bleu.rescore(best[0].full_derivation)
#     print "Best BLEU   Score: %s"% (forest.bleu.score_ratio_str()) 
#     print oracle_partition -partition, oracle_partition, partition


    average= 0.0
#     for i in range(min(10, len(best))):
#       print "  Best   Trans: %s"%best[i].full_derivation
#       forest.bleu.rescore(best[i].full_derivation)
#       average += len(best[i].full_derivation.split())
#       print "  Best BLEU   Score: %s"% (forest.bleu.score_ratio_str())
#       print "  Best Score: %s"% (best[i].score[0]) 
    # print "Average Length %s"%(average / float(i))
    #print "Local Difference: %s"%(oracle_partition-partition)
    return example_marg, oracle_marg, oracle_partition-partition # log div
  
  def _refresh(self):
    self.decoder = CYKDecoder(self.weights, self.lm)
    
  def decode(self, forest, verbose = False):    
    #decoding
    # def non_local_scorer(cedge, ders):
#       hyp = cedge.assemble(ders)
#       return ((0.0, Vector()),  hyp, hyp)
#     decoder = CubePruning(MarginalDecoder.FeatureAdder(self.weights), non_local_scorer, 20, 5, find_min=False)
#     best = decoder.run(forest.root)

    
#     for i in range(min(10, len(best))):
#       print "  Best   Trans: %s"%best[i].full_derivation
#       forest.bleu.rescore(best[i].full_derivation)
#       print "  Best BLEU   Score: %s"% (forest.bleu.score_ratio_str())
#       print "  Best Score: %s"% (best[i].score[0]) 
      

#     test_bleu = forest.bleu.rescore((best[0].full_derivation))
    #oracle_forest, oracle_item = oracle.oracle_extracter(forest, self.weights, None, 5, 2, extract=1)
    (score , subtree, fv)  = forest.bestparse(self.weights, use_min=False)
    print "textlenght", fv["Basic/text-length"], self.weights["Basic/text-length"], score
    if verbose:
      
      print "Ref Tran %s"%forest.refs
      (oracle_score, oracle_subtree, oracle_fv,_) = forest.compute_oracle(self.weights)
      print "Oracle: "+ oracle_subtree


      #oracle_forest, oracle_item = oracle.oracle_extracter(forest, self.weights, None, 100, 2, extract=100)
      #(oracle_forest_score , oracle_forest_subtree, oracle_forest_fv)  = oracle_forest.bestparse(self.weights, use_min=False)

      # example_marg, partition  = fast_inside_outside.collect_marginals(forest, self.weights)
#       oracle_marg, oracle_partition  = fast_inside_outside.collect_marginals(oracle_forest, self.weights)
#       print "Oracle forest likelihood: ",oracle_partition - partition     
      
#       print "Oracle Forest Score: ", oracle_forest_score
#       print "Oracle Forest Results: ", oracle_forest_subtree
#       print "Oracle Forest Bleu: ", forest.bleu.rescore(oracle_forest_subtree)
#       parse_diff(self.weights,fv, oracle_forest_fv)

      print "Best: ", subtree
      print "Best Score: ", (score, self.weights.dot(oracle_fv))
      print "Local Score: ",forest.bleu.rescore(subtree) 
      # (score, best_trans, best_fv) = self.decoder.beam_search(forest, b=FLAGS.beam)     
      
      print "\n\n\n"
    #print delta_feats
    forest.bleu.rescore(subtree)
    return (forest.bleu, None)


  @staticmethod
  def test_marginals():
    pass


    
def main():
  gc.set_threshold(100000,10,10)
  flags.DEFINE_integer("beam", 100, "beam size", short_name="b")
  flags.DEFINE_integer("debuglevel", 0, "debug level")
  flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost>)")
  flags.DEFINE_boolean("cube", True, "using cube pruning to speedup")
  flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")
  flags.DEFINE_integer("ratio", 3, "the maximum items (pop from PQ): ratio*b", short_name="r")
  flags.DEFINE_boolean("dist", False, "ditributed (hadoop) training)")
  flags.DEFINE_string("prefix", "", "prefix for distributed training")
  flags.DEFINE_string("hadoop_weights", "", "hadoop weights (formatted specially)")
  flags.DEFINE_boolean("add_features", False, "add features to training data")
  flags.DEFINE_boolean("prune_train", False, "prune before decoding")
  flags.DEFINE_boolean("no_lm", False, "don't use the unigram language model")
  flags.DEFINE_boolean("pickleinput", False, "assumed input is pickled")
  flags.DEFINE_string("oracle_forests", None, "oracle forests", short_name="o")
  flags.DEFINE_string("feature_map_file", None, "file with the integer to feature mapping (for lbfgs)")
  flags.DEFINE_boolean("cache_input", False, "cache input sentences (only works for pruned input)")
  flags.DEFINE_string("rm_features", None, "list of features to remove")
  flags.DEFINE_boolean("just_basic", False, "remove all features but basic")

  argv = FLAGS(sys.argv)

  if FLAGS.weights:
    weights = Model.cmdline_model()
  else:
    vector = Vector()
    assert glob.glob(FLAGS.hadoop_weights)
    for file in glob.glob(FLAGS.hadoop_weights):
      for l in open(file):
        if not l.strip(): continue
        f,v = l.strip().split()
        vector[f] = float(v)
    weights = Model(vector)

  rm_features = set()
  if FLAGS.rm_features:  
    for l in open(FLAGS.rm_features):
      rm_features.add(l.strip())
    
  lm = Ngram.cmdline_ngram()
  if FLAGS.no_lm:
    lm = None
  
  if argv[1] == "train":
    local_decode = ChiangPerceptronDecoder(weights, lm)
  elif argv[1] == "sgd" or argv[1] == "crf":
    local_decode = MarginalDecoder(weights, lm)
  else:
    local_decode = MarginalDecoder(weights, lm)

  if FLAGS.add_features:
    tdm = local_features.TargetDataManager()
    local_decode.feature_adder = FeatureAdder(tdm) 
  local_decode.prune_train = FLAGS.prune_train
  local_decode.use_pickle = FLAGS.pickleinput
  local_decode.cache_input = FLAGS.cache_input
  print >>logs, "Cache input is %s"%FLAGS.cache_input
  if FLAGS.debuglevel > 0:
    print >>logs, "beam size = %d" % FLAGS.beam
    
  if argv[1] == "train":
    if not FLAGS.dist:
      perc = trainer.Perceptron.cmdline_perc(local_decode)
    else:
      train_files = [FLAGS.prefix+file.strip() for file in sys.stdin]
      perc = distributed_trainer.DistributedPerceptron.cmdline_perc(local_decode)
      perc.set_training(train_files)      
    perc.train()
  elif argv[1] == "sgd":
    crf = sgd.BaseCRF.cmdline_crf(local_decode)
    crf.set_oracle_files([FLAGS.oracle_forests])
    crf.train()
    
  elif argv[1] == "crf":
    if not FLAGS.dist:
      crf = CRF.LBFGSCRF.cmdline_crf(local_decode)
      crf.set_oracle_files([FLAGS.oracle_forests])
      crf.set_feature_mappers(add_features.read_features(FLAGS.feature_map_file))
      crf.rm_features(rm_features)
      if FLAGS.just_basic:
        print "Enforcing Basic"
        crf.enforce_just_basic()
      crf.train()
    else:
      #train_files = [FLAGS.prefix+file.strip() for file in sys.stdin]
      #oracle_files = [file+".oracle" for file in train_files]
      print >>sys.stderr, "DistributedCRF"
      crf = distCRF.DistributedCRF.cmdline_distibuted_crf(local_decode)
      #os.system("~/.python/bin/dumbo rm train_input -hadoop /home/nlg-03/mt-apps/hadoop/0.20.1+169.89/")      
      #os.system("~/.python/bin/dumbo put "+crf.trainfiles[0]+" train_input -hadoop /home/nlg-03/mt-apps/hadoop/0.20.1+169.89/")      
      crf.set_feature_mappers(add_features.read_features(FLAGS.feature_map_file))
      crf.rm_features(rm_features)
      if FLAGS.just_basic:
        print "Enforcing Basic"
        crf.enforce_just_basic()

      #crf.set_oracle_files(oracle_files)      
      crf.train()

  else:
    if not FLAGS.dist:
      print "Evaluating"
      eval = Evaluator(local_decode, [FLAGS.dev])
      eval.tune()
    else:
      dev_files = [FLAGS.prefix+file.strip() for file in sys.stdin]
      eval = Evaluator(local_decode, dev_files)
    print eval.eval(verbose=True).compute_score()
    
if __name__ == "__main__":
  main()
