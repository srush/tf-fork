"""
Manages evaluation of parameters for dev and test
"""



class Evaluator(object):
  def __init__(self, decoder, dev):
    
    self.decoder = decoder
    self.devfile = dev

  def eval(self, num_examples=None, verbose = False):
    print "EVAL"
    tot = self.decoder.evalclass()        
    print self.devfile
    for i, example in enumerate(self.decoder.load(self.devfile), 1):
      similarity, _ = self.decoder.decode(example, verbose=verbose)
      tot += similarity
      print "Bleu: %s" %tot.compute_score()
      print "Ratio: %s" %tot.score_ratio_str()
      if num_examples <> None and i == num_examples:
        break
    return tot

  def tune(self):
    lower = -1000.0
    upper = 1000.0
    cur = (lower + upper) / 2.0
    length_ratio = 0.0
    weights = self.decoder.get_model_weights()
    while True:
      print "Fail", length_ratio, cur 
      weights["Basic/text-length"] = cur
      self.decoder.set_model_weights(weights)
      bleu_score = self.eval(num_examples = 25)
      prec = bleu_score.compute_score()
      length_ratio = bleu_score.ratio()
      
      if length_ratio >= 0.99 and length_ratio <= 1.01: break
      
      if length_ratio > 1.01:
        upper = cur
        cur = (lower + cur) / 2.0
      else:
        lower = cur
        cur = (upper + cur) / 2.0
    return cur
      

class IDecoder(object):
    '''providing three functions: load(), decode(early_stop=False), has attribute model, evalclass.'''

    def load(self, filenames):
        raise NotImplementedError, "Need a way to load in examples"
        #for i, line in enumerate(open(filename), 1):
        #    yield DepTree.parse(line)

    def decode(self, reftree, early_stop=False):
        raise NotImplementedError, "Need to decode"
    
    def evalclass(self):
        raise NotImplementedError, "Need to eval"

def main():
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
  flags.DEFINE_string("oracle_forests", None, "oracle forests", short_name="o")
  flags.DEFINE_string("feature_map_file", None, "file with the integer to feature mapping (for lbfgs)")

  argv = FLAGS(sys.argv)
    
  vector = Vector()
  for file in glob.glob(FLAGS.hadoop_weights):
    for l in open(file):
      if not l.strip(): continue
      f,v = l.strip().split("\t")
      vector[f] = float(v)
    weights = Model(vector)
    
  lm = Ngram.cmdline_ngram()
  if FLAGS.no_lm:
    lm = None
  
  local_decode = MarginalDecoder(weights, lm)
    
  if not FLAGS.dist:
    eval = Evaluator(local_decode, [FLAGS.dev])
  else:
    dev_files = [FLAGS.prefix+file.strip() for file in sys.stdin]
    eval = Evaluator(local_decode, dev_files)
    #print eval.eval().compute_score()




if __name__ == "__main__":
  main()
