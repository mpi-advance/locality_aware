import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import pyfancyplot as plt
import glob

class PingPong:
    active_procs = ""
    sizes = ""
    times = ""

    def __init__(self, num_active):
        self.active_procs = num_active
        self.sizes = list()
        self.times = list()

    def add_timing(self, size, time):
        if size in self.sizes:
            idx = self.sizes.index(size)
            self.times[idx] = time
        else:
            self.sizes.append(size)
            self.times.append(time)

## TODO : Change if new architecture
num_sockets = 4
GPN = 4

on_socket_pp = PingPong(1)
on_node_pp = PingPong(1)
off_node_pp = PingPong(1)

on_socket_multi = list()
on_node_multi = list()
off_node_multi = list()
off_node_even_sockets = list()

files = glob.glob("cpu_microbenchmarks.*.out")
for fn in files:
    file = open(fn, 'r')
    active_group = False
    group = ""
    active_method = False
    method = ""

    for line in file:
        if "Ping-Pong" in line:
            if "Even" in line or ("All" in line and "Active" in line): # Testing Multiproc, even (numa/socket/etc)
                active_method = False
                if "Off-Node MultiProc" in line and "Even NUMA" in line:
                    print(line)
                    active_group = True
                    group = off_node_even_sockets
                else:
                    active_group = False

            elif "MultiProc" in line: # Testing Multiproc, not even
                active_method = False
                if "On-NUMA" in line:
                    active_group = True
                    group = on_socket_multi
                elif "On-Node, Off-Socket" in line:
                    active_group = True
                    group = on_node_multi
                elif "Off-Node" in line:
                    active_group = True
                    group = off_node_multi
                else:
                    active_group = False

            else: # Testing single ping-pong, 1 ping proc, 1 pong proc
                active_group = False
                if "On-NUMA" in line:
                    active_method = True
                    method = on_socket_pp
                elif "On-Node, Off-Socket" in line:
                    active_method = True
                    method = on_node_pp
                elif "Off-Node" in line:
                    active_method = True
                    method = off_node_pp
                else: # Don't gather times for intra-NUMA
                    active_method = False

        elif active_group and "Active Procs" in line:
            num_active = (int)((line.split('\n')[0]).rsplit(' ')[-1])
            found = False
            for m in group:
                if m.active_procs == num_active:
                    method = m
                    found = True
                    break
            if not found:
                group.append(PingPong(num_active))
                method = group[-1]
            active_method = True
        elif active_method and "Size" in line:
            splitting = line.split(':')
            size = (int)(splitting[0].rsplit(' ')[-1])
            time = (float)(splitting[-1].split('\n')[0])
            method.add_timing(size, time/2.0) # Adding time / 2 (ping + pong)

    file.close()
        
#####################################
### Standard Inter-CPU Ping Pongs ###
#####################################
### Single Ping-Pong              ###
### Intra-socket, Intra-Node, vs  ###
### Inter-node                    ###
#####################################
plt.add_luke_options()
## On-Socket Times
plt.line_plot(on_socket_pp.times, on_socket_pp.sizes, color='r', label = "Intra-NUMA")
## On-Node Times
plt.line_plot(on_node_pp.times, on_node_pp.sizes, color='b', label = "Inter-NUMA")
## Off-Node Times
plt.line_plot(off_node_pp.times, off_node_pp.sizes, color='g', label = "Inter-Node")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("standard_inter_cpu.pdf")


#####################################
### Multiple Processes Per Region ###
#####################################
### Single Ping-Pong between procs###
### Intra-socket, Intra-Node      ###
#####################################
# On-Socket, MultiProc
plt.add_luke_options()
for i in range(len(on_socket_multi)):
    plt.line_plot(on_socket_multi[i].times, on_socket_multi[i].sizes, label = "%d Active Procs"%on_socket_multi[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("on_socket_multi.pdf")

# Off-Socket, MultiProc
plt.add_luke_options()
for i in range(len(on_node_multi)):
    plt.line_plot(on_node_multi[i].times, on_node_multi[i].sizes, label = "%d Active Procs"%on_node_multi[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("on_node_multi.pdf")


#####################################
### Multiple Processes Per Region ###
#####################################
### Inter-node                    ###
### Times and Bandwidth           ###
### Sizes per process vs per node ###
#####################################
# Off-Node, MultiProc
plt.add_luke_options()
for i in range(len(off_node_multi)):
    plt.line_plot(off_node_multi[i].times, off_node_multi[i].sizes, label = "%d Active Procs"%off_node_multi[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_multi.pdf")

# Off-Node, MultiProc
plt.add_luke_options()
for i in range(len(off_node_multi)):
    sizes = [s * off_node_multi[i].active_procs for s in off_node_multi[i].sizes]
    plt.line_plot(off_node_multi[i].times, sizes, label = "%d Active Procs"%off_node_multi[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Node (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_multi_pernode.pdf")

# Off-Node Bandwidth (MultiProc)
plt.add_luke_options()
for i in range(len(off_node_multi)):
    sizes = [s for s in off_node_multi[i].sizes]    
    bw = [(off_node_multi[i].sizes[j])/off_node_multi[i].times[j] for j in range(len(off_node_multi[i].sizes))]
    plt.line_plot(bw, sizes, label = "%d Active Procs"%off_node_multi[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Process (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_bw_perproc.pdf")

# Off-Node Bandwidth (MultiProc)
plt.add_luke_options()
for i in range(len(off_node_multi)):
    sizes = [s * off_node_multi[i].active_procs for s in off_node_multi[i].sizes]    
    bw = [(off_node_multi[i].sizes[j]*off_node_multi[i].active_procs)/off_node_multi[i].times[j] for j in range(len(off_node_multi[i].sizes))]
    plt.line_plot(bw, sizes, label = "%d Active Procs"%off_node_multi[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Node (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_bw_pernode.pdf")


#####################################
### Even Sockets                  ###
#####################################
### Inter-node                    ###
### Even number procs per NUMA    ###
### Uses all NICS                 ###
### Times and Bandwidth           ###
### Sizes per process vs per node ###
#####################################

# Off-Node, MultiProc, Even Sockets
plt.add_luke_options()
for i in range(len(off_node_even_sockets)):
    plt.line_plot(off_node_even_sockets[i].times, off_node_even_sockets[i].sizes, label = "%d Active PPS"%off_node_even_sockets[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_allsocketsactive_perproc.pdf")


# Off-Node, MultiProc, Even Sockets
plt.add_luke_options()
for i in range(len(off_node_even_sockets)):
    sizes = [s * off_node_even_sockets[i].active_procs for s in off_node_even_sockets[i].sizes]
    plt.line_plot(off_node_even_sockets[i].times, sizes, label = "%d Active PPS"%off_node_even_sockets[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Node (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_allsocketsactive_pernode.pdf")


# Off-Node Bandwidth (MultiProc, Even Sockets)
plt.add_luke_options()
for i in range(len(off_node_even_sockets)):
    sizes = [s  for s in off_node_even_sockets[i].sizes]        
    bw = [(off_node_even_sockets[i].sizes[j])/off_node_even_sockets[i].times[j] for j in range(len(off_node_even_sockets[i].sizes))]
    plt.line_plot(bw, off_node_even_sockets[i].sizes, label = "%d Active PPS"%off_node_even_sockets[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_bw_allsocketsactive_perproc.pdf")


# Off-Node Bandwidth (MultiProc, Even Sockets)
plt.add_luke_options()
for i in range(len(off_node_even_sockets)):
    sizes = [s * off_node_even_sockets[i].active_procs * num_sockets for s in off_node_even_sockets[i].sizes]        
    bw = [(off_node_even_sockets[i].sizes[j]*off_node_even_sockets[i].active_procs)/off_node_even_sockets[i].times[j] for j in range(len(off_node_even_sockets[i].sizes))]
    plt.line_plot(bw, off_node_even_sockets[i].sizes, label = "%d Active PPS"%off_node_even_sockets[i].active_procs)
plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("off_node_bw_allsocketsactive_pernode.pdf")




gpu_aware = False
c2c = False
active_group = False
active_method = False

gpu_on_node = PingPong(1)
gpu_off_node = PingPong(1)
c2c_on_node = PingPong(1)
c2c_off_node = PingPong(1)
gpu_multi = list()
gpu_multi_all = list()
c2c_multi = list()
c2c_multi_all = list()


files = glob.glob("gpu_microbenchmarks.*.out")
for fn in files:
    file = open(fn, 'r')
    for line in file:
        if "all GPUs active per node" in line:
            active_method = False
            if c2c:
                active_group = True
                group = c2c_multi_all
            elif gpu_aware:
                active_group = True
                group = gpu_multi_all
            else:
                active_group = False

        elif "Multiple Processes Per GPU" in line:
            active_method = False
            if c2c:
                active_group = True
                group = c2c_multi
            elif gpu_aware:
                active_group = True
                group = gpu_multi
            else:
                active_group = False

        elif "Ping-Pong" in line:
            active_group = False
            if c2c:
                if "On-Node, Off-Socket" in line:
                    active_method = True
                    method = c2c_on_node
                elif "Off-Node" in line:
                    active_method = True
                    method = c2c_off_node
                else:
                    active_method = Flase
            elif gpu_aware:
                if "On-Node, Off-Socket" in line:
                    active_method = True
                    method = gpu_on_node
                elif "Off-Node" in line:
                    active_method = True
                    method = gpu_off_node
                else:
                    active_method = Flase

        elif "Running" in line and "benchmarks" in line:
            if "GPU-Aware" in line:
                c2c = False
                gpu_aware = True
            elif "Copy-to-CPU" in line:
                c2c = True
                gpu_aware = False

        elif active_group and "Active Procs" in line:
            num_active = (int)((line.split('\n')[0]).rsplit(' ')[-1])
            found = False
            for m in group:
                if m.active_procs == num_active:
                    method = m
                    found = True
                    break
            if not found:
                group.append(PingPong(num_active))
                method = group[-1]
            active_method = True
        elif active_method and "Size" in line:
            splitting = line.split(':')
            size = (int)(splitting[0].rsplit(' ')[-1])
            time = (float)(splitting[-1].split('\n')[0])
            method.add_timing(size, time/2.0) # Adding time / 2 (ping + pong)

    file.close()


#####################################
### Standard Inter-GPU ###
#####################################
### Inter-node vs Intra-Node      ###
### GPU-Aware vs Copy2CPU         ###
#####################################

# Standard Inter-GPU Ping Pongs
plt.add_luke_options()
plt.line_plot(gpu_on_node.times, gpu_on_node.sizes, color='r', label = "Intra-Node")
plt.line_plot(gpu_off_node.times, gpu_off_node.sizes, color='b', label = "Inter-Node")
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("standard_inter_gpu.pdf")


# Standard Inter-GPU Ping Pongs
plt.add_luke_options()
plt.line_plot(gpu_on_node.times, gpu_on_node.sizes, label = "GPU-Aware Intra-Node")
plt.line_plot(gpu_off_node.times, gpu_off_node.sizes, label = "GPU-Aware Inter-Node")
plt.line_plot(c2c_on_node.times, c2c_on_node.sizes, label = "C2C Intra-Node")
plt.line_plot(c2c_off_node.times, c2c_off_node.sizes, label = "C2C Inter-Node")
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_aware_vs_c2c.pdf")



#####################################
### Multiple Processes per GPU    ###
#####################################
### Inter-node                    ###
### Sizes : per proc vs           ###
###         per GPU vs            ###
###         per node              ###
#####################################
# Multiple Processes Per GPU
plt.add_luke_options()
for i in range(len(gpu_multi)):
    plt.line_plot(gpu_multi[i].times, gpu_multi[i].sizes, label = "%d Active Procs"%gpu_multi[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_multi_perproc.pdf")

# Multiple Processes Per GPU
plt.add_luke_options()
for i in range(len(gpu_multi)):
    sizes = [s*gpu_multi[i].active_procs for s in gpu_multi[i].sizes]
    plt.line_plot(gpu_multi[i].times, sizes, label = "%d Active Procs"%gpu_multi[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per GPU (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_multi_pergpu.pdf")

# Multiple Processes Per GPU
plt.add_luke_options()
for i in range(len(gpu_multi_all)):
    plt.line_plot(gpu_multi_all[i].times, gpu_multi_all[i].sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_multi_allgpusactive_perproc.pdf")


#####################################
### Multiple Processes per GPU    ###
#####################################
### Inter-node                    ###
### All GPUs active (use all NICs)###
### Sizes per gpu vs per node     ###
#####################################

# Multiple Processes Per GPU
plt.add_luke_options()
for i in range(len(gpu_multi_all)):
    sizes = [s*gpu_multi_all[i].active_procs for s in gpu_multi_all[i].sizes]
    plt.line_plot(gpu_multi_all[i].times, sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
plt.add_anchored_legend(ncol=2, fontsize=16)
plt.add_labels("Message Size Per GPU (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_multi_allgpusactive_pergpu.pdf")

# Speedup at max size and max PPG
max_ppg = max(map(lambda x: gpu_multi_all[x].active_procs, range(len(gpu_multi_all))))
one_ppg_sizes = None
one_ppg_times = None
max_ppg_sizes = None
max_ppg_times = None
for i in range(len(gpu_multi_all)):
    if gpu_multi_all[i].active_procs == 1:
        one_ppg_sizes = [s*gpu_multi_all[i].active_procs for s in gpu_multi_all[i].sizes]
        one_ppg_times = gpu_multi_all[i].times
    elif gpu_multi_all[i].active_procs == max_ppg:
        max_ppg_sizes = [s*gpu_multi_all[i].active_procs for s in gpu_multi_all[i].sizes]
        max_ppg_times = gpu_multi_all[i].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print("gpu_multi_allgpusactive_pergpu: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

# Multiple Processes Per GPU
plt.add_luke_options()
for i in range(len(gpu_multi_all)):
    sizes = [s*gpu_multi_all[i].active_procs*GPN for s in gpu_multi_all[i].sizes]
    plt.line_plot(gpu_multi_all[i].times, sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per GPU (Bytes)", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_multi_allgpusactive_pernode.pdf")


#####################################
### Multiple Processes per GPU    ###
#####################################
### Inter-node                    ###
### All GPUs active (use all NICs)###
### Injection BW : per GPU,node   ###
#####################################

# Multiple Processes Per GPU -- Bandwidth
plt.add_luke_options()
for i in range(len(gpu_multi_all)):
    sizes = [s * gpu_multi_all[i].active_procs for s in gpu_multi_all[i].sizes]
    bw = [gpu_multi_all[i].sizes[j] * gpu_multi_all[i].active_procs / gpu_multi_all[i].times[j] for j in range(len(gpu_multi_all[i].sizes))]
    plt.line_plot(bw, sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per GPU (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_bw_allgpusactive_pergpu.pdf")

# Multiple Processes Per GPU -- Bandwidth
plt.add_luke_options()
for i in range(len(gpu_multi_all)):
    sizes = [s * gpu_multi_all[i].active_procs *GPN for s in gpu_multi_all[i].sizes]
    bw = [gpu_multi_all[i].sizes[j] * gpu_multi_all[i].active_procs * GPN / gpu_multi_all[i].times[j] for j in range(len(gpu_multi_all[i].sizes))]
    plt.line_plot(bw, sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per Node (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("gpu_bw_allgpusactive_pernode.pdf")


#####################################
### CPU/GPU Bandwidth Comparisons ###
#####################################
### Inter-node                    ###
### All GPUs active (use all NICs)###
### Injection BW : per node       ###
### Single Process per GPU        ###
#####################################

# CPU (Even Sockets) vs GPU Bandwidth (1 PPG)
plt.add_luke_options()

# 1 CPU core per socket
sizes = [s * off_node_even_sockets[0].active_procs * num_sockets for s in off_node_even_sockets[0].sizes]    
bw = [(off_node_even_sockets[0].sizes[j]*off_node_even_sockets[0].active_procs * num_sockets)/off_node_even_sockets[0].times[j] for j in range(len(off_node_even_sockets[0].sizes))]
plt.line_plot(bw, sizes, label = "1 CPU Core Per Socket")

# 16 cpu cores per socket
sizes = [s * off_node_even_sockets[1].active_procs * num_sockets for s in off_node_even_sockets[1].sizes]    
bw = [(off_node_even_sockets[1].sizes[j]*off_node_even_sockets[1].active_procs * num_sockets)/off_node_even_sockets[1].times[j] for j in range(len(off_node_even_sockets[1].sizes))]
plt.line_plot(bw, sizes, label = "%d CPU Cores Per Socket"%off_node_even_sockets[1].active_procs)

# 1 Proc per GPU
sizes = [s * gpu_multi_all[0].active_procs for s in gpu_multi_all[0].sizes]    
bw = [gpu_multi_all[0].sizes[j] * gpu_multi_all[0].active_procs * GPN / gpu_multi_all[0].times[j] for j in range(len(gpu_multi_all[0].sizes))]
plt.line_plot(bw, sizes, label = "1 GPU Per Socket")

plt.add_anchored_legend(ncol=3)
plt.add_labels("Message Size Per Node (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("injection_bandwidth_1ppg.pdf")



#####################################
### CPU/GPU Bandwidth Comparisons ###
#####################################
### Inter-node                    ###
### All GPUs active (use all NICs)###
### Injection BW : per node       ###
### 1 vs 16 processes per GPU     ###
#####################################

# CPU (Even Sockets) vs GPU Bandwidth (1 PPG)
plt.add_luke_options()

# 1 CPU core per socket
sizes = [s * off_node_even_sockets[0].active_procs * num_sockets for s in off_node_even_sockets[0].sizes]    
bw = [(off_node_even_sockets[0].sizes[j]*off_node_even_sockets[0].active_procs * num_sockets)/off_node_even_sockets[0].times[j] for j in range(len(off_node_even_sockets[0].sizes))]
plt.line_plot(bw, sizes, label = "1 CPU Core Per Socket")

# 16 cpu cores per socket
sizes = [s * off_node_even_sockets[1].active_procs * num_sockets for s in off_node_even_sockets[1].sizes]    
bw = [(off_node_even_sockets[1].sizes[j]*off_node_even_sockets[1].active_procs * num_sockets)/off_node_even_sockets[1].times[j] for j in range(len(off_node_even_sockets[1].sizes))]
plt.line_plot(bw, sizes, label = "%d CPU Cores Per Socket"%off_node_even_sockets[1].active_procs)

# 1 Proc per GPU
sizes = [s * gpu_multi_all[0].active_procs for s in gpu_multi_all[0].sizes]    
bw = [gpu_multi_all[0].sizes[j] * gpu_multi_all[0].active_procs * GPN / gpu_multi_all[0].times[j] for j in range(len(gpu_multi_all[0].sizes))]
plt.line_plot(bw, sizes, label = "1 Process Per GPU")

# 16 Proc per GPU
sizes = [s * gpu_multi_all[1].active_procs for s in gpu_multi_all[1].sizes]    
bw = [gpu_multi_all[1].sizes[j] * gpu_multi_all[1].active_procs * GPN / gpu_multi_all[1].times[j] for j in range(len(gpu_multi_all[1].sizes))]
plt.line_plot(bw, sizes, label = "%d Processes Per GPU"%gpu_multi_all[1].active_procs)

plt.add_anchored_legend(ncol=2)
plt.add_labels("Message Size Per Node (Bytes)", "Bandwidth (Bytes Per Second)")
plt.set_scale('log', 'log')
plt.save_plot("injection_bandwidth.pdf")
