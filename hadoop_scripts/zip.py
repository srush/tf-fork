import sys, itertools

files = sys.argv[1:]
handles = map(open, files)
for lines in itertools.izip(*handles):
  print "\t".join([l.strip() for l in lines])
