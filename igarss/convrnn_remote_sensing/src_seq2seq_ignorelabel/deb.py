import inspect
import re
import sys

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	GREENCLEAR = '\033[36m'
	BLUE = '\033[34m'


def prints(x,level_actual=1,level_constant=1,fname="debug"):
	#print("[@"+sys._getframe().f_code.co_name+"]")
	if level_actual>=level_constant:
		try:
			frame = inspect.currentframe().f_back
			s = inspect.getframeinfo(frame).code_context[0]
			r = re.search(r"\((.*)\)", s).group(1)
			if fname is not "debug":
				r = r[0:-6]
			print("{}[@{}] {} = {}{}".format(bcolors.OKGREEN,fname,r,x,bcolors.ENDC))
		except:
			print("Deb prints error. Value:",x)


#x=34
if __name__ == "__main__":
	#print(bcolors.WARNING+"asdf"+bcolors.ENDC)
	d={}
	d["f"]=[23,3]
	prints(d["f"])
