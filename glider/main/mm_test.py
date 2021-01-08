import os
import subprocess
import sys
import time
import socket
import signal


proc_first = None
proc_second = None

def signal_handler(signum,frame):
	sys.stderr.write('catch SIGINT, kill the subprocesses, program exit\n')
	proc_first.kill()
	proc_second.kill()
	sys.exit(0)

def get_open_port():
	sock = socket.socket(socket.AF_INET)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	sock.bind(('', 0))
	port = sock.getsockname()[1]
	sock.close()
	return str(port)

def Popen(cmd, **kwargs):
	print(cmd)
	return subprocess.Popen(cmd,**kwargs)

port = get_open_port()


mm_cmd = ['mm-delay','50', 'mm-link', '--meter-all',\
 'trace/60mbps.trace', 'trace/60mbps.trace','--uplink-queue=droptail', '--uplink-queue-args=packets=1', '--downlink-queue=droptail','--downlink-queue-args=packets=1']
# mm_cmd = ['mm-delay',str(int(delay/2))]

cmd = ['python','run_sender.py',port]

proc_first = Popen(cmd, preexec_fn=os.setsid)

# ensure quite listening
time.sleep(5)

# MAHIMAHI_BASE is a macro after you setting the mm-* commands
sh_cmd = 'python run_receiver.py $MAHIMAHI_BASE %s' % port
sh_cmd = ' '.join(mm_cmd) + " -- sh -c '%s'" % sh_cmd

"""take care!!! if you use mm-link ***, you should and -- before sh for seperate the commands,
   like the following
"""
# sh_cmd = ' '.join(mm_cmd) + " -- sh -c '%s'" % sh_cmd

proc_second = Popen(sh_cmd, shell=True)

# register signal handler

signal.signal(signal.SIGINT,signal_handler)
signal.pause()

# normally wait the subprocesses
try:
	proc_first.wait()
	proc_second.wait()
except Exception:
	pass
finally:
	pass



