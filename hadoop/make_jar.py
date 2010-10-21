import os,sys


s = os.system
s("mkdir $TMPDIR/files")
tmpdir = os.getenv("TMPDIR")
argv = sys.argv[1:]
files = []

def read_item(f):
  collect = ""
  for l in f:
    if not l.strip():
      yield collect
      collect = ""
    else:
      collect += l 

for i, item in enumerate(read_item(open(argv[0])), 1):
  files.append("file.%s"%(i))
  f = open(tmpdir+"/files/"+files[-1], 'w')
  print >>f, item
  f.close()
  
allfiles = open("files.txt", 'w')
print >>allfiles, "\n".join(files)

s("cd $TMPDIR/files; jar cvf files.jar " + " ".join(files))
  
