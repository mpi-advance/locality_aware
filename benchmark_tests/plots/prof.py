computer = ""
procs = ""
ppn = ""
filename = ""

if 0:
    computer = "quartz"
    procs = [4, 16, 16, 32, 64]
    ppn = [4, 4, 16, 32, 4]
    filename = "bruck_allgather"

if 1:
    computer = "lassen"
    procs = [4, 8, 16, 32]
    #ppn = [4, 4, 16, 4]
    filename = "gpu_alltoall"

