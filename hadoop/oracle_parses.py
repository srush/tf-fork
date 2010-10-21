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
        print >>sys.stderr, "key", key
        print >> sys.stderr, value[0:10], value.split("\t", 1)[1][0:10]
        yield value

    tmp_handle = open(self.tmp_file, 'w')
    for l in decode(get_vals()):
      print >>sys.stderr, "print"
      print >>tmp_handle, l
    tmp_handle.close()
      
    # Oracle-ize the forst
    print >>sys.stderr, os.getenv("PYTHONPATH")
    self.shell.call("export PYTHONPATH=%s;cd $TRANSFOREST; cat $TMP_FILE | $PYTHON $TRANSFOREST/Features/oracle.py -w $TRANSFOREST/example/config.ini --lm $TRANSFOREST/example/lm.3.sri --order 3 $LOCAL_TMPDIR/oracle"% os.getenv("PPATH"))

    # Add Features to oracle forst
    self.shell.call("export PYTHONPATH=%s;cd $TRANSFOREST;  $PYTHON $TRANSFOREST/Features/add_features.py  $LOCAL_TMPDIR/oracle $LOCAL_TMPDIR/oracle_features"%os.getenv("PPATH"))

    # Add features to the main sentence
    self.shell.call("export PYTHONPATH=%s;cd $TRANSFOREST;  $PYTHON $TRANSFOREST/Features/add_features.py $TMP_FILE $LOCAL_TMPDIR/features"%os.getenv("PPATH"))

    for i,(l1,l2) in enumerate(izip(encode(self.shell.open("$LOCAL_TMPDIR/features")),
                                    encode(self.shell.open("$LOCAL_TMPDIR/oracle_features")))):
      yield int(l.split("\t",1)[0]), l1 + "****" + l2


def starter(program):  
  program.addopt("cmdenv","PYTHON=%s"%os.getenv("PYTHON"))
  program.addopt("cmdenv","TRANSFOREST=%s"%os.getenv("TRANSFOREST"))
  program.addopt("cmdenv","LD_LIBRARY_PATH=%s"%os.getenv("LD_LIBRARY_PATH"))
  program.addopt("cmdenv","PPATH=%s"%os.getenv("PYTHONPATH"))
  
def runner(job):  
  import dumbo
  dumbo.run(Mapper)


if __name__ == "__main__":
  import dumbo
  dumbo.main(runner,starter)
  
