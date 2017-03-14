#!/usr/bin/python
# written by HyunMin-kor 
#---------------------------------------------------------------------- -------
# Usage  							              |
# python pyget.py URL : Download a file from URL to HDFS                      |
# python pyget.py URL FILENAME : DOWNload a file from URL to HDFS as FILENAME |
#------------------------------------------------------------------------------
import pydoop.hdfs as hdfs
import sys
import os
from urllib2 import urlopen

def DownloadFile(url,filename=None):
	CHUNK = 16 * 1024
	if filename is None:
		filename = url.split('/')[-1]
	else:
		filename = filename
	response = urlopen(url)
#	with open(filename, 'wb') as f:
	with hdfs.open(filename,"w") as f:
		while True:
			chunk = response.read(CHUNK)
			if not chunk: break
			f.write_chunk(chunk)
			f.flush()
	f.close()
	f.fs.close()
	
	return 0

# Main
if __name__ == "__main__":
	if len(sys.argv) == 1:
		print "Please Enter URL to download"
		exit(1)
	elif len(sys.argv) == 2:
		DownloadFile(sys.argv[1])
		print "Done"
		exit(2)
	elif len(sys.argv) == 3:
		DownloadFile(sys.argv[1],sys.argv[2])
		print "Done"
		exit(2)
	else:
		print "Too many arguments"
		exit(1)


