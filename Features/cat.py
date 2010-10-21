import sys


def file_inp():
  for l in sys.stdin:
    for item in open("file_archive/" + l.strip()):
      yield item.strip()
      
for item in file_inp():
  if item.strip():
    print item + "eaten"
