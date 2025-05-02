import numpy as np
import glob

computer = "dane"
leaders = [4, 10, 20]

import pyfancyplot as plt
import matplotlib
matplotlib.use("qtagg")

class multileader:
    ml = ""
    loc = ""
    ml_loc = ""
    n_leaders = 0

    def __init__(self, n_leaders):
        self.n_leaders = n_leaders
        self.ml = list()
        self.loc = list()
        self.ml_loc = list()

    def add_size(self):
        self.ml.append(np.inf)
        self.loc.append(np.inf)
        self.ml_loc.append(np.inf)

    def add_time(self, name, idx, time):
        if "Multileader Locality" in name:
            if (time < self.ml_loc[idx]):
                self.ml_loc[idx] = time
        elif "Multileader" in name:
            if (time < self.ml[idx]):
                self.ml[idx] = time
        elif "Locality" in name:
            if (time < self.loc[idx]):
                self.loc[idx] = time


class implementation:
    std = ""
    hierarchical = ""
    node_aware = ""
    leader = ""

    def __init__(self):
        self.std = list()
        self.hierarchical = list()
        self.node_aware = list()
        self.leader = list()
        for l in leaders:
            self.leader.append(multileader(l))

    def add_size(self):
        self.std.append(np.inf)
        self.hierarchical.append(np.inf)
        self.node_aware.append(np.inf)
        for l in self.leader:
            l.add_size()

    def add_time(self, name, idx, time):
        if "Hierarchical" in name:
            if (time < self.hierarchical[idx]):
                self.hierarchical[idx] = time
        elif "Node Aware" in name:
            if (time < self.node_aware[idx]):
                self.node_aware[idx] = time
        elif "leaders" in name:
            n_leaders = (int)(name.rsplit(' ')[-2])
            leader_idx = leaders.index(n_leaders)
            self.leader[leader_idx].add_time(name, idx, time)
        else:
            if (time < self.std[idx]):
                self.std[idx] = time


 
class alltoalls:
    pmpi = ""
    pairwise = ""
    nonblocking = ""
    sizes = ""

    def __init__(self):
        self.sizes = list()
        self.pmpi = list()
        self.pairwise = implementation()
        self.nonblocking = implementation()

    def add_size(self, size):
        if (self.sizes.count(size)):
            return
        self.sizes.append(size)
        self.pmpi.append(np.inf)
        self.pairwise.add_size()
        self.nonblocking.add_size()

    def add_time(self, name, size, time):
        idx = self.sizes.index(size)
        if "PMPI" in name:
            if (time < self.pmpi[idx]):
                self.pmpi[idx] = time
        elif "Pairwise" in name:
            self.pairwise.add_time(name, idx, time)
        elif "Nonblocking" in name or "NonBlocking" in line:
            self.nonblocking.add_time(name, idx, time)




nodes = [2, 4, 8, 16, 32]
timings = []
size = 0
for n in nodes:
    timings.append(alltoalls())
    for f in glob.glob("../runscripts/%s_results/alltoall_N%d*.out"%(computer, n)):
        file = open(f)
        
        for line in file:
            if "Size" in line:
                size = (int)((line.rsplit('\n')[0]).rsplit(' ')[1])
                timings[-1].add_size(size)
            elif "PMPI" in line or "Pairwise" in line or "Nonblocking" in line or "NonBlocking" in line:
                name = line.rsplit(':')[0]
                time = (float)((line.rsplit('\n')[0]).rsplit(' ')[-1])
                timings[-1].add_time(name, size, time)
        file.close()


print(timings[-1].nonblocking.std)
print(timings[-1].pairwise.leader[-1].ml_loc)


for i in range(len(nodes)):
    plt.add_luke_options()
    plt.line_plot(timings[i].pmpi, timings[i].sizes, color='black', label='PMPI')

    plt.line_plot(timings[i].pairwise.std, timings[i].sizes, color='red', label='Pairwise')
    plt.line_plot(timings[i].nonblocking.std, timings[i].sizes, color='red', tickmark='--', label="Nonblocking")

    plt.line_plot(timings[i].pairwise.hierarchical, timings[i].sizes, color='blue', label='Hierarchical')
    plt.line_plot(timings[i].nonblocking.hierarchical, timings[i].sizes, color='blue', tickmark='--')

    plt.line_plot(timings[i].pairwise.node_aware, timings[i].sizes, color='green', label='Node-Aware')
    plt.line_plot(timings[i].nonblocking.node_aware, timings[i].sizes, color='green', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[0].ml, timings[i].sizes, color='orange', label='Multileader')
    plt.line_plot(timings[i].nonblocking.leader[0].ml, timings[i].sizes, color='orange', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[0].loc, timings[i].sizes, color='brown', label='Locality-Aware')
    plt.line_plot(timings[i].nonblocking.leader[0].loc, timings[i].sizes, color='brown', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[0].ml_loc, timings[i].sizes, color='purple', label='Multileader + Locality')
    plt.line_plot(timings[i].nonblocking.leader[0].ml_loc, timings[i].sizes, color='purple', tickmark='--')

    plt.add_anchored_legend()
    plt.set_scale('log', 'log')
    plt.add_labels("Msg Size", "Timing (seconds)")
    plt.save_plot("%s/sizes_n%d.pdf"%(computer,nodes[i]))
    plt.plt.clf()


    plt.add_luke_options()
    plt.line_plot(timings[i].pmpi, timings[i].sizes, color='black', label='PMPI')

    plt.line_plot(timings[i].pairwise.leader[0].ml, timings[i].sizes, color='purple', label="4 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[0].ml, timings[i].sizes, color='purple', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[1].ml, timings[i].sizes, color='purple', label="10 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[1].ml, timings[i].sizes, color='purple', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[2].ml, timings[i].sizes, color='purple', label="20 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[2].ml, timings[i].sizes, color='purple', tickmark='--')

    plt.add_anchored_legend()
    plt.set_scale('log', 'log')
    plt.add_labels("Msg Size", "Timing (seconds)")
    plt.save_plot("%s/multileader_n%d.pdf"%(computer,nodes[i]))
    plt.plt.clf()


    plt.add_luke_options()
    plt.line_plot(timings[i].pmpi, timings[i].sizes, color='black', label='PMPI')

    plt.line_plot(timings[i].pairwise.leader[0].loc, timings[i].sizes, color='purple', label="4 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[0].loc, timings[i].sizes, color='purple', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[1].loc, timings[i].sizes, color='purple', label="10 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[1].loc, timings[i].sizes, color='purple', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[2].loc, timings[i].sizes, color='purple', label="20 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[2].loc, timings[i].sizes, color='purple', tickmark='--')

    plt.add_anchored_legend()
    plt.set_scale('log', 'log')
    plt.add_labels("Msg Size", "Timing (seconds)")
    plt.save_plot("%s/locality_n%d.pdf"%(computer,nodes[i]))
    plt.plt.clf()


    plt.add_luke_options()
    plt.line_plot(timings[i].pmpi, timings[i].sizes, color='black', label='PMPI')

    plt.line_plot(timings[i].pairwise.leader[0].ml_loc, timings[i].sizes, color='red', label="4 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[0].ml_loc, timings[i].sizes, color='red', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[1].ml_loc, timings[i].sizes, color='purple', label="10 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[1].ml_loc, timings[i].sizes, color='purple', tickmark='--')

    plt.line_plot(timings[i].pairwise.leader[2].ml_loc, timings[i].sizes, color='green', label="20 Leaders Per Node")
    plt.line_plot(timings[i].nonblocking.leader[2].ml_loc, timings[i].sizes, color='green', tickmark='--')

    plt.line_plot(timings[i].pairwise.node_aware, timings[i].sizes, color='blue', label="Node-Aware")
    plt.line_plot(timings[i].nonblocking.node_aware, timings[i].sizes, color='blue', tickmark='--')

    plt.add_anchored_legend()
    plt.set_scale('log', 'log')
    plt.add_labels("Msg Size", "Timing (seconds)")
    plt.save_plot("%s/multileader_loc_n%d.pdf"%(computer,nodes[i]))
    plt.plt.clf()




## Plot Size 1 Across Nodes
plt.add_luke_options()
plt.line_plot([timings[i].pmpi[0] for i in range(len(timings))], nodes, color='black', label='PMPI')

plt.line_plot([timings[i].pairwise.std[0] for i in range(len(timings))], nodes, color='red', label='Pairwise')
plt.line_plot([timings[i].nonblocking.std[0] for i in range(len(timings))], nodes, color='red', tickmark='--', label='Nonblocking')

plt.line_plot([timings[i].pairwise.hierarchical[0] for i in range(len(timings))], nodes, color='blue', label='Hierarchical')
plt.line_plot([timings[i].nonblocking.hierarchical[0] for i in range(len(timings))], nodes, color='blue', tickmark='--')

plt.line_plot([timings[i].pairwise.node_aware[0] for i in range(len(timings))], nodes, color='green', label='Node-Aware')
plt.line_plot([timings[i].nonblocking.node_aware[0] for i in range(len(timings))], nodes, color='green', tickmark='--')

plt.line_plot([timings[i].pairwise.leader[-1].ml[0] for i in range(len(timings))], nodes, color='orange', label='Multileader')
plt.line_plot([timings[i].nonblocking.leader[-1].ml[0] for i in range(len(timings))], nodes, color='orange', tickmark='--')

plt.line_plot([timings[i].pairwise.leader[-1].loc[0] for i in range(len(timings))], nodes, color='brown', label='Locality-Aware')
plt.line_plot([timings[i].nonblocking.leader[-1].loc[0] for i in range(len(timings))], nodes, color='brown', tickmark='--')

plt.line_plot([timings[i].pairwise.leader[-1].ml_loc[0] for i in range(len(timings))], nodes, color='purple', label='Multileader + Locality')
plt.line_plot([timings[i].nonblocking.leader[-1].ml_loc[0] for i in range(len(timings))], nodes, color='purple', tickmark='--')

plt.add_anchored_legend()
ax = plt.get_ax()
ax.set_xscale('log', base=2)
ax.set_yscale('log')
plt.add_labels("Nodes", "Timing (seconds)")
plt.save_plot("%s/scaling_size1.pdf"%computer)
plt.plt.clf()


## Plot Largest Size Across Nodes
size = timings[-1].sizes[-2]
idx = len(timings[-1].sizes) - 2
print("Size:", timings[-1].sizes[idx])

plt.add_luke_options()
plt.line_plot([timings[i].pmpi[idx] for i in range(len(timings))], nodes, color='black', label='PMPI')

plt.line_plot([timings[i].pairwise.std[idx] for i in range(len(timings))], nodes, color='red', label='Pairwise')
plt.line_plot([timings[i].nonblocking.std[idx] for i in range(len(timings))], nodes, color='red', tickmark='--', label='Nonblocking')

plt.line_plot([timings[i].pairwise.hierarchical[idx] for i in range(len(timings))], nodes, color='blue', label='Hierarchical')
plt.line_plot([timings[i].nonblocking.hierarchical[idx] for i in range(len(timings))], nodes, color='blue', tickmark='--')

plt.line_plot([timings[i].pairwise.node_aware[idx] for i in range(len(timings))], nodes, color='green', label='Node-Aware')
plt.line_plot([timings[i].nonblocking.node_aware[idx] for i in range(len(timings))], nodes, color='green', tickmark='--')

plt.line_plot([timings[i].pairwise.leader[-1].ml[idx] for i in range(len(timings))], nodes, color='orange', label='Multileader')
plt.line_plot([timings[i].nonblocking.leader[-1].ml[idx] for i in range(len(timings))], nodes, color='orange', tickmark='--')

plt.line_plot([timings[i].pairwise.leader[-1].loc[idx] for i in range(len(timings))], nodes, color='brown', label='Locality-Aware')
plt.line_plot([timings[i].nonblocking.leader[-1].loc[idx] for i in range(len(timings))], nodes, color='brown', tickmark='--')

plt.line_plot([timings[i].pairwise.leader[-1].ml_loc[idx] for i in range(len(timings))], nodes, color='purple', label='Multileader + Locality')
plt.line_plot([timings[i].nonblocking.leader[-1].ml_loc[idx] for i in range(len(timings))], nodes, color='purple', tickmark='--')

plt.add_anchored_legend()
ax = plt.get_ax()
ax.set_xscale('log', base=2)
ax.set_yscale('log')
plt.add_labels("Nodes", "Timing (seconds)")
plt.save_plot("%s/scaling_size%d.pdf"%(computer,size))
plt.plt.clf()

