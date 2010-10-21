"""

"""

import sys
import os
import re
import subprocess
import logging
import itertools

code_dir = os.getenv("TRANSFOREST")
sys.path.append(code_dir+'/scripts/')
from shell_helpers import *
from encode import *


class Mapper(object):
  def __init__(self):
    self.shell = ShellRunner({})
    self.shell.make_local_tmp_dir()
    rule_file = os.getenv("RULE_FILE")
    self.shell.check_input(rule_file, ["rhs50", "count2"])

    self.tmp_ref = self.shell.subs["REF_FILE"] =  self.shell.tmpdir + "ref_file"
    self.tmp_parse = self.shell.subs["PARSE_FILE"] = self.shell.tmpdir + "parse_file"

    # transforest code directory
    self.shell.subs["P_DIR"] = os.getenv("TRANSFOREST")
      
    # location of the full rule file
    self.shell.subs["RULE_FILE"] = rule_file

    
    self.shell.subs["PYTHON"] = os.getenv("PYTHON")
    
    # write temporary parse and ref riles
    
    self.shell.subs["PFOREST"] = self.shell.tmpdir + "/tmp.pforest"

  def __call__(self, data):
    ref_handle = open(self.tmp_ref, 'w')
    parse_handle = open(self.tmp_parse, 'w')
    min_key = 1e90
    for key, value in data:
      (parse, ref) = value.strip().split("\t")
      print >>parse_handle, parse
      print >>ref_handle, ref
      min_key = min(min_key, key)
    ref_handle.close()
    parse_handle.close()
      
    # Follow README file

    # 1) convert a parse tree to a trivial parse forest
    self.shell.call("cat $PARSE_FILE | $PYTHON $P_DIR/tree.py --toforest > $PFOREST")

    # 2.1) filter the large rule set and output the small rule set, which is only used in current parse forest. 
    self.shell.call("cat $PFOREST | $PYTHON $P_DIR/forest.py --rulefilter $RULE_FILE -w $P_DIR/example/config.ini --max_height 3 > $LOCAL_TMPDIR/rules_count2_rhs50")

    # SKIP assume it is done previously
  
    # 2.2) filter the count=1 rules
    #self.shell.call("grep -v \"count1=1 \" $LOCAL_TMPDIR/rules >$LOCAL_TMPDIR/rules_count2")

    # 2.3) filter max(lhs)<=k
    #self.shell.call("cat $LOCAL_TMPDIR/rules_count2 | $PYTHON $P_DIR/rulefilter.py 50  > $LOCAL_TMPDIR/rules_count2_rhs50")
  
    # 2.4) convert a parse forest into a translation forest. NOTE -w won't be used 
    self.shell.call("cat $PFOREST | $PYTHON $P_DIR/forest.py -r $LOCAL_TMPDIR/rules_count2_rhs50 --max_height 3 -w \"gt_prob=-1\" $REF_FILE 1> $LOCAL_TMPDIR/processed.tforest")
  
    # 3) prune a translation forest
    self.shell.call("cat $LOCAL_TMPDIR/processed.tforest | $PYTHON $P_DIR/prune.py --lm $P_DIR/example/lm.3.sri  -r15 -w $P_DIR/example/config.ini > $LOCAL_TMPDIR/processed.pruned.tforest")

    #output without last \n
    for i,l in enumerate(encode(self.shell.open("$LOCAL_TMPDIR/processed.pruned.tforest"))):
      yield int(min_key), l

  #self.shell.call("rm $LOCAL_TMPDIR/*")






def starter(program):
  from dumbo.backends import get_backend
  def require(a):
    var = program.delopt(a)
    assert var, a
    return var
  
  original_parse_file = require("original_parse")
  original_ref_file = require("original_ref")
  rule_file = require("rule_file")
  

  shell = ShellRunner({})
  shell.subs["CHINESE"] = original_parse_file
  shell.subs["ENGLISH"] = original_ref_file
  shell.subs["P_DIR"] = os.getenv("TRANSFOREST")
  shell.subs["PYTHON"] = os.getenv("PYTHON")
  shell.subs["TMPDIR"] = os.getenv("TMPDIR")
  
  shell.call("$PYTHON $P_DIR/hadoop/zip.py $CHINESE $ENGLISH > $TMPDIR/combined.txt")


  backend = get_backend(program.opts)
  fs = backend.create_filesystem(program.opts)
  fs.put(shell.complete("$TMPDIR/combined.txt"), "combined.txt", program.opts)
  
  program.addopt("input","combined.txt")
  program.addopt("cmdenv","RULE_FILE=%s"%rule_file)
  program.addopt("cmdenv","PYTHON=%s"%os.getenv("PYTHON"))
  program.addopt("cmdenv","TRANSFOREST=%s"%os.getenv("TRANSFOREST"))
  program.addopt("cmdenv","LD_LIBRARY_PATH=%s"%os.getenv("LD_LIBRARY_PATH"))
  program.addopt("cmdenv","PYTHONPATH=%s"%os.getenv("PYTHONPATH"))  
def runner(job):  
  import dumbo
  dumbo.run(Mapper)


if __name__ == "__main__":
  import dumbo
  dumbo.main(runner,starter)
  
