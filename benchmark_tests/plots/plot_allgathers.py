import numpy as np

import profile
import glob

import matplotlib
from matplotlib import pyplot as plt
from pyfancyplot import plot

class Test():
    procs = 0
    ppn = 0
    mpi = ""
    standard = ""
    locality = ""
    hierarchical = ""
    multilane = ""
    sizes = ""

    def __init__(self, procs, ppn):
        self.procs = procs
        self.ppn = ppn
        self.mpi = list()
        self.standard = list()
        self.locality  = list()
        self.hierarchical = list()
        self.multilane = list()
        self.sizes = list()

    def list_append(self, size):
        self.sizes.append(size)
        self.mpi.append(np.inf)
        self.standard.append(np.inf)
        self.locality.append(np.inf)
        self.hierarchical.append(np.inf)
        self.multilane.append(np.inf)

    def add_time(self, i, line):
        time = (float)((line.rsplit('\n')[0]).rsplit(' ')[-1])
        if "MPI" in line:
            if time < self.mpi[i]:
                self.mpi[i] = time
        elif "mult_hier" in line: 
            if time < self.multilane[i]:
                self.multilane[i] = time
        elif "hier" in line:
            if time < self.hierarchical[i]:
                self.hierarchical[i] = time
        elif "loc" in line:
            if time < self.locality[i]:
                self.locality[i] = time
        else:
            if time < self.standard[i]:
                self.standard[i] = time


tests = list()
folder = "../%s"%profile.computer
for i in range(len(profile.procs)):
    tests.append(Test(profile.procs[i], profile.ppn[i]))
    for file in glob.glob("%s/%s_n%d_ppn%d.*.out"%(folder, profile.filename, profile.procs[i], profile.ppn[i])):
        idx = -1
        f = open(file)
        for line in f:
            if "Testing Size" in line:
                if idx + 1 == len(tests[-1].sizes):
                    tests[-1].list_append((int)((line.rsplit('\n')[0]).rsplit(' ')[-1]))
                idx = idx + 1
            elif "Time" in line:
                tests[-1].add_time(idx, line)
        f.close()


# Plot Size = 1
plot.add_luke_options()
standard = [t.standard[1] for t in tests]
hierarchical = [t.hierarchical[1] for t in tests]
multilane = [t.multilane[1] for t in tests]
locality = [t.locality[1] for t in tests]
mpi = [t.mpi[1] for t in tests]
plot.barplot(np.arange(len(tests)), [standard, hierarchical, multilane, locality], ["Bruck", "Hierarchical", "Multi-lane", "Locality"])

width = 0.4
ax = plot.get_ax()
xticks = ax.get_xticks()

for i in range(len(xticks)):
    plt.plot([xticks[i] - width, xticks[i] + width], [mpi[i], mpi[i]], '--', color='black')
plot.add_labels("", "Measured Time (Seconds)")
labels = list()
for i in range(len(profile.ppn)):
    ppn = profile.ppn[i]
    nodes = profile.procs[i] / ppn
    labels.append("%d PPN\n%d Nodes"%(ppn, nodes))

plot.set_xticklabels(labels, rotation='horizontal')
plot.save_plot("%s_%s.pdf"%(profile.computer, profile.filename))
