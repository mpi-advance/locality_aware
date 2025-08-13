import numpy as np
import glob

computer = "dane"
ppn = 112
procs_per_leader = [4, 8, 16]
n_leaders = [ppn / x for x in procs_per_leader]

import matplotlib
matplotlib.use("Qt5Agg")
import pyfancyplot as plt

class algorithm_times:
    n_leaders = 0
    pairwise = ""
    nonblocking = ""
    internal_gather = ""
    internal_scatter = ""
    internal_pairwise_intranode = ""    
    internal_pairwise_internode = ""
    internal_nonblocking_intranode = ""    
    internal_nonblocking_internode = ""

    def __init__(self, n_leaders):
        self.n_leaders = n_leaders
        self.pairwise = list()
        self.nonblocking = list()
        self.internal_gather = list()
        self.internal_scatter = list()
        self.internal_pairwise_intranode = list()
        self.internal_pairwise_internode = list()
        self.internal_nonblocking_intranode = list()
        self.internal_nonblocking_internode = list()

    def add_size(self):
        self.pairwise.append(np.inf)
        self.nonblocking.append(np.inf)
        self.internal_gather.append(np.inf)
        self.internal_scatter.append(np.inf)
        self.internal_pairwise_intranode.append(np.inf)
        self.internal_pairwise_internode.append(np.inf)
        self.internal_nonblocking_intranode.append(np.inf)
        self.internal_nonblocking_internode.append(np.inf)

    def add_time(self, name, idx, time):
        name = name.lower()

        time_list = ""
        if "internal" in name or "interal" in name:
            if "gather" in name:
                time_list = self.internal_gather
            elif "scatter" in name:
                time_list = self.internal_scatter
            elif "pairwise" in name:
                if "intra" in name:
                    time_list = self.internal_pairwise_intranode
                else:
                    time_list = self.internal_pairwise_internode
            elif "nonblocking" in name:
                if "intra" in name:
                    time_list = self.internal_nonblocking_intranode
                else:
                    time_list = self.internal_nonblocking_internode
        elif "pairwise" in name:
            time_list = self.pairwise
        elif "nonblocking" in name:
            time_list = self.nonblocking
        else:
            print("No name!", name)

        if (time < time_list[idx]):
            time_list[idx] = time
                

 
class alltoalls:
    pmpi = ""
    hierarchical = ""
    multileader = ""
    node_aware = ""
    locality_aware = ""
    multileader_node = ""
    sizes = ""

    def __init__(self):
        self.sizes = list()
        self.pmpi = list()
        self.hierarchical = algorithm_times(1)
        self.node_aware = algorithm_times(np.inf)

        self.multileader = list()
        self.locality_aware = list()
        self.multileader_node = list()
        for nl in n_leaders:
            self.multileader.append(algorithm_times(nl))
            self.locality_aware.append(algorithm_times(nl))
            self.multileader_node.append(algorithm_times(nl))

    def add_size(self, size):
        if (self.sizes.count(size)):
            return

        self.sizes.append(size)
        self.pmpi.append(np.inf)
        self.hierarchical.add_size()
        self.node_aware.add_size()
        for i in range(len(n_leaders)):
            self.multileader[i].add_size()
            self.locality_aware[i].add_size()
            self.multileader_node[i].add_size()

    def add_time(self, name, size, time):
        idx = self.sizes.index(size)
        if "PMPI" in name:
            if (time < self.pmpi[idx]):
                self.pmpi[idx] = time
        elif "N Leaders" in line or "N Groups" in line:
            print(line)
            leaders = (int)((name.rsplit(')')[0]).rsplit(' ')[-1])
            leader_idx = n_leaders.index(leaders)
            if "Multileader Node-Aware" in line:
                self.multileader_node[leader_idx].add_time(name, idx, time)
            elif "Hierarchical" in line:
                print(line)
                self.multileader[leader_idx].add_time(name, idx, time)
            elif "Locality-Aware" in line:
                self.locality_aware[leader_idx].add_time(name, idx, time)
        elif "Hierarchical" in line:
            self.hierarchical.add_time(name, idx, time)
        elif "Node-Aware" in line:
            self.node_aware.add_time(name, idx, time)

nodes = [2, 4, 8, 16, 32]
timings = []
size = 0
for n in nodes:
    timings.append(alltoalls())
    print("../benchmark_tests/%s/alltoalls_n%d*.out"%(computer, n))
    for f in glob.glob("../benchmark_tests/%s/alltoalls_n%d*.out"%(computer, n)):
        print(f)
        file = open(f)
        for line in file:
            if "Size" in line:
                size = (int)((line.rsplit('\n')[0]).rsplit(' ')[1])
                print(size)
                timings[-1].add_size(size)
                #elif "PMPI" in line or "Hierarchical" in line or "Node-Aware" in line or "Locality-Aware" in line:
            elif ":" in line:
                #print(line)
                name = line
                time = (float)((line.rsplit('\n')[0]).rsplit(' ')[-1])
                #print(line)
                timings[-1].add_time(name, size, time)
        file.close()


print(timings[-1].sizes[0:-1])
print(timings[-1].pmpi[0:-1])
print(timings[-1].hierarchical.internal_gather[0:-1])
print(timings[-1].hierarchical.internal_scatter[0:-1])
print(timings[-1].hierarchical.internal_pairwise_internode[0:-1])
print(timings[-1].hierarchical.internal_nonblocking_internode[0:-1])


# Hierarchical Timings [-1] (largest node count), scaling sizes
plt.add_luke_options()
data = [timings[-1].hierarchical.internal_gather[0:-1], timings[-1].hierarchical.internal_scatter[0:-1], timings[-1].hierarchical.internal_pairwise_internode[0:-1], timings[-1].hierarchical.internal_nonblocking_internode[0:-1]]
labels = ["MPI_Gather", "MPI_Scatter", "Alltoall (Pairwise)", "Alltoall (Nonblocking)"]
plt.add_labels("Per-Message Size (Bytes)", "Time (Seconds)")
plt.barplot(timings[-1].sizes[0:-1], data, labels)
plt.set_scale('linear', 'log')
plt.set_xticks(np.arange(len(timings[-1].sizes[0:-1])), [4*t for t in timings[-1].sizes[0:-1]])
plt.save_plot("hierarchical_sizes.pdf")


# Hierarchical Timings, largest size, scaling node counts
plt.add_luke_options()
gather = [t.hierarchical.internal_gather[-2] for t in timings]
scatter = [t.hierarchical.internal_scatter[-2] for t in timings]
inter_pairwise = [t.hierarchical.internal_pairwise_internode[-2] for t in timings]
data = [gather, scatter, inter_pairwise]
labels = ["MPI_Gather", "MPI_Scatter", "Alltoall"]
plt.add_labels("Number of Nodes", "Time (Seconds)")
plt.stacked_barplot(nodes, data, labels)
plt.save_plot("hierarchical_nodes.pdf")



# Multileader Timings, largest size, scaling node counts
plt.add_luke_options()
gather = [timings[-1].hierarchical.internal_gather[-2]]
scatter = [timings[-1].hierarchical.internal_scatter[-2]]
inter_pairwise = [timings[-1].hierarchical.internal_pairwise_internode[-2]]

for i in range(len(procs_per_leader)-1, -1, -1):
    print(i, timings[-1].multileader[i].internal_gather)
    gather.append(timings[-1].multileader[i].internal_gather[-2])
    scatter.append(timings[-1].multileader[i].internal_scatter[-2])
    inter_pairwise.append(timings[-1].multileader[i].internal_pairwise_internode[-2])

labels = ["MPI_Gather", "MPI_Scatter", "Alltoall"]
plt.barplot([0, 1, 2, 3], [gather, scatter, inter_pairwise], labels)
    
plt.add_labels("Leader Size", "Time (Seconds)")
plt.set_xticks([0,1,2,3], ["Hierarchical", "16 PPL", "8 PPL", "4 PPL"])
plt.save_plot("multileader_nodes.pdf")

# Node-Aware Timings [-1] (largest node count), scaling sizes
plt.add_luke_options()
data = [timings[-1].node_aware.internal_pairwise_intranode[0:-1], timings[-1].node_aware.internal_pairwise_internode[0:-1], timings[-1].node_aware.internal_nonblocking_intranode[0:-1], timings[-1].node_aware.internal_nonblocking_internode[0:-1]]
labels = ["Intra-Node (Pairwise) ", "Inter-Node (Pairwise)", "Intra-Node (Nonblocking) ", "Inter-Node (Nonblocking)"]
plt.add_labels("Per-Message Size (Bytes)", "Time (Seconds)")
plt.barplot(timings[-1].sizes[0:-1], data, labels)
plt.set_scale('linear', 'log')
plt.set_xticks(np.arange(len(timings[-1].sizes[0:-1])), [4*t for t in timings[-1].sizes[0:-1]])
plt.save_plot("node_aware_sizes.pdf")


# Node-Aware Timings, largest size, scaling node counts
plt.add_luke_options()
intra_alltoall = [t.node_aware.internal_pairwise_intranode[-2] for t in timings]
inter_alltoall = [t.node_aware.internal_pairwise_internode[-2] for t in timings]
data = [intra_alltoall, inter_alltoall]
labels = ["Intra-Node Alltoall", "Inter-Node Alltoall"]
plt.add_labels("Number of Nodes", "Time (Seconds)")
plt.stacked_barplot(nodes, data, labels)
plt.save_plot("node_aware_nodes.pdf")


# Multileader Timings, largest size, scaling node counts
plt.add_luke_options()
intra_alltoall = [timings[-1].node_aware.internal_pairwise_intranode[-2]]
inter_alltoall = [timings[-1].node_aware.internal_pairwise_internode[-2]]
for i in range(len(procs_per_leader)-1, -1, -1):
    print(i, timings[-1].locality_aware[i].internal_pairwise_intranode)
    intra_alltoall.append(timings[-1].locality_aware[i].internal_pairwise_intranode[-2])
    inter_alltoall.append(timings[-1].locality_aware[i].internal_pairwise_internode[-2])

labels = ["Intra-Node Alltoall", "Inter-Node Alltoall"]
plt.barplot([0, 1, 2, 3], [intra_alltoall, inter_alltoall], labels)
    
plt.add_labels("Group Size", "Time (Seconds)")
plt.set_xticks([0,1,2,3], ["Node-Aware", "16 PPG", "8 PPG", "4 PPG"])
plt.save_plot("locality_aware_nodes.pdf")


