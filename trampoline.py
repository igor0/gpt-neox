import subprocess
import sys

#cmd = sys.argv[1:]
cmd = ["./evaluate.sh"] + sys.argv[1:]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

for c in iter(lambda: proc.stdout.read(1), b''): 
    sys.stdout.buffer.write(c)

proc.communicate()[0]
exit(proc.returncode)
