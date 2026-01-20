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

GPN_SPX = 4
GPN_TPX = 12
GPN_CPX = 24

dict_gpu_on_node = {}
dict_gpu_off_node = {}
dict_c2c_on_node = {}
dict_c2c_off_node = {}
dict_gpu_multi = {}
dict_gpu_multi_all = {}
dict_c2c_multi = {}
dict_c2c_multi_all = {}

for mode in ["spx", "tpx", "cpx"]:
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


    files = glob.glob(("gpu" if mode == "spx" else mode) + "_microbenchmarks_n2.*.out")
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
                        active_method = False
                elif gpu_aware:
                    if "On-Node, Off-Socket" in line:
                        active_method = True
                        method = gpu_on_node
                    elif "Off-Node" in line:
                        active_method = True
                        method = gpu_off_node
                    else:
                        active_method = False

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
                size = (int)(splitting[1].rsplit(' ')[-1])
                time = (float)(splitting[-1].split('\n')[0])
                method.add_timing(size, time/2.0) # Adding time / 2 (ping + pong)

        file.close()

    dict_gpu_on_node[mode] = gpu_on_node
    dict_gpu_off_node[mode] = gpu_off_node
    dict_c2c_on_node[mode]  = c2c_on_node
    dict_c2c_off_node[mode] = c2c_off_node
    dict_gpu_multi[mode] = gpu_multi
    dict_gpu_multi_all[mode] = gpu_multi_all
    dict_c2c_multi[mode] = c2c_multi
    dict_c2c_multi_all[mode] = c2c_multi_all

for mode in ["tpx", "cpx"]:
    curr_gpn = GPN_TPX if mode == "tpx" else GPN_CPX if mode == "cpx" else GPN_SPX

    gpu_on_node = dict_gpu_on_node[mode]
    gpu_off_node = dict_gpu_off_node[mode]
    c2c_on_node = dict_c2c_on_node[mode]
    c2c_off_node = dict_c2c_off_node[mode]
    gpu_multi = dict_gpu_multi[mode]
    gpu_multi_all = dict_gpu_multi_all[mode]
    c2c_multi = dict_c2c_multi[mode]
    c2c_multi_all = dict_c2c_multi_all[mode]

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
    plt.save_plot(mode + "_standard_inter_gpu.pdf")


    # Standard Inter-GPU Ping Pongs
    plt.add_luke_options()
    plt.line_plot(gpu_on_node.times, gpu_on_node.sizes, label = "GPU-Aware Intra-Node")
    plt.line_plot(gpu_off_node.times, gpu_off_node.sizes, label = "GPU-Aware Inter-Node")
    plt.line_plot(c2c_on_node.times, c2c_on_node.sizes, label = "C2C Intra-Node")
    plt.line_plot(c2c_off_node.times, c2c_off_node.sizes, label = "C2C Inter-Node")
    plt.add_anchored_legend(ncol=2)
    plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
    plt.set_scale('log', 'log')
    plt.save_plot(mode + "_gpu_aware_vs_c2c.pdf")



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
    plt.save_plot(mode + "_gpu_multi_perproc.pdf")

    # Multiple Processes Per GPU
    plt.add_luke_options()
    for i in range(len(gpu_multi)):
        sizes = [s*gpu_multi[i].active_procs for s in gpu_multi[i].sizes]
        plt.line_plot(gpu_multi[i].times, sizes, label = "%d Active Procs"%gpu_multi[i].active_procs)
    plt.add_anchored_legend(ncol=2)
    plt.add_labels("Message Size Per GPU (Bytes)", "Time (Seconds)")
    plt.set_scale('log', 'log')
    plt.save_plot(mode + "_gpu_multi_pergpu.pdf")

    # Multiple Processes Per GPU
    plt.add_luke_options()
    for i in range(len(gpu_multi_all)):
        plt.line_plot(gpu_multi_all[i].times, gpu_multi_all[i].sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
    plt.add_anchored_legend(ncol=2)
    plt.add_labels("Message Size Per Process (Bytes)", "Time (Seconds)")
    plt.set_scale('log', 'log')
    plt.save_plot(mode + "_gpu_multi_allgpusactive_perproc.pdf")


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
    plt.add_anchored_legend(ncol=2)
    plt.add_labels("Message Size Per GPU (Bytes)", "Time (Seconds)")
    plt.set_scale('log', 'log')
    plt.save_plot(mode + "_gpu_multi_allgpusactive_pergpu.pdf")

    # Speedup at max size and max PPG
    sub = 0
    max_ppg = max(map(lambda x: gpu_multi_all[x].active_procs, range(len(gpu_multi_all))))
    while True:
        try:
            max_ppg_actual = max_ppg - sub
            if max_ppg_actual <= 1:
                break
            one_ppg_sizes = None
            one_ppg_times = None
            max_ppg_sizes = None
            max_ppg_times = None
            for i in range(len(gpu_multi_all)):
                if gpu_multi_all[i].active_procs == 1:
                    one_ppg_sizes = [s*gpu_multi_all[i].active_procs for s in gpu_multi_all[i].sizes]
                    one_ppg_times = gpu_multi_all[i].times
                elif gpu_multi_all[i].active_procs == max_ppg_actual:
                    max_ppg_sizes = [s*gpu_multi_all[i].active_procs for s in gpu_multi_all[i].sizes]
                    max_ppg_times = gpu_multi_all[i].times
            intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
            max_size = max(intersect)
            print(mode + "_gpu_multi_allgpusactive_pergpu: speedup at " + str(max_size) + " for active_procs " + str(max_ppg_actual) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
            break
        except:
            sub += 1

    # Multiple Processes Per GPU
    plt.add_luke_options()
    for i in range(len(gpu_multi_all)):
        sizes = [s*gpu_multi_all[i].active_procs*curr_gpn for s in gpu_multi_all[i].sizes]
        plt.line_plot(gpu_multi_all[i].times, sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
    plt.add_anchored_legend(ncol=2)
    plt.add_labels("Message Size Per GPU (Bytes)", "Time (Seconds)")
    plt.set_scale('log', 'log')
    plt.save_plot(mode + "_gpu_multi_allgpusactive_pernode.pdf")


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
    plt.save_plot(mode + "_gpu_bw_allgpusactive_pergpu.pdf")

    # Multiple Processes Per GPU -- Bandwidth
    plt.add_luke_options()
    for i in range(len(gpu_multi_all)):
        sizes = [s * gpu_multi_all[i].active_procs *curr_gpn for s in gpu_multi_all[i].sizes]
        bw = [gpu_multi_all[i].sizes[j] * gpu_multi_all[i].active_procs * curr_gpn / gpu_multi_all[i].times[j] for j in range(len(gpu_multi_all[i].sizes))]
        plt.line_plot(bw, sizes, label = "%d Active Procs"%gpu_multi_all[i].active_procs)
    plt.add_anchored_legend(ncol=2)
    plt.add_labels("Message Size Per Node (Bytes)", "Bandwidth (Bytes Per Second)")
    plt.set_scale('log', 'log')
    plt.save_plot(mode + "_bw_allgpusactive_pernode.pdf")