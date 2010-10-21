import sys
import os
code_dir = os.getenv("TRANSFOREST")
sys.path.append(code_dir+'/scripts/')
from shell_helpers import *
from encode import *
from itertools import *

class Mapper(object):
  def __init__(self):
    self.shell = ShellRunner({})
    self.shell.make_local_tmp_dir()

    # transforest code directory
    self.shell.subs["TRANSFOREST"] = os.getenv("TRANSFOREST")
      
    self.shell.subs["PYTHON"] = os.getenv("PYTHON")

    self.shell.subs["TMP_FILE"] = self.tmp_file = self.shell.tmpdir + "forests"
    
    
  def __call__(self, data):
    self.min_key = 1e90
    def get_vals():
      for key, value in data:
        self.min_key = min(self.min_key, key)
        yield value

    tmp_handle = open(self.tmp_file, 'w')
    for l in decode(get_vals()):
      print >>tmp_handle, l

    tmp_handle.close()
      
    # Oracle-ize the forst 
    self.shell.call("export PYTHONPATH=%s;cd $TRANSFOREST;  $PYTHON $TRANSFOREST/Features/add_features.py --feature_map $TMP_FILE $LOCAL_TMPDIR/feature_set"%os.getenv("PPATH"))

    for l in self.shell.open("$LOCAL_TMPDIR/feature_set"):
      yield l.strip().split("\t")[1], 1

def reduce(key, values):
  yield key, 1

def starter(program):  
  program.addopt("cmdenv","PYTHON=%s"%os.getenv("PYTHON"))
  program.addopt("cmdenv","TRANSFOREST=%s"%os.getenv("TRANSFOREST"))
  program.addopt("cmdenv","LD_LIBRARY_PATH=%s"%os.getenv("LD_LIBRARY_PATH"))
  program.addopt("cmdenv","PPATH=%s"%os.getenv("PYTHONPATH"))
  
def runner(job):  
  import dumbo
  dumbo.run(Mapper, reduce)


if __name__ == "__main__":
  import dumbo
  dumbo.main(runner,starter)
  
