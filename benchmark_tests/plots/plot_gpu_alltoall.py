import numpy as np

import prof
import glob

import matplotlib
from matplotlib import pyplot as plt
from pyfancyplot import plot

std = 1

matplotlib.use("TkAgg")

class Test():
    procs = 0
    mpi = ""
    thread_mpi = ""
    ga_pe = ""
    ga_nb = ""
    ctc_pe = ""
    ctc_nb = ""
    thread_pe = ""
    thread_nb = ""
    sizes = ""

    def __init__(self, procs):
        self.procs = procs
        self.mpi = list()
        self.thread_mpi = list()
        self.ga_pe = list()
        self.ga_nb = list()
        self.ctc_pe = list()
        self.ctc_nb = list()
        self.thread_pe = list()
        self.thread_nb = list()
        self.sizes = list()


    def list_append(self, size):
        self.sizes.append(size)
        self.mpi.append(np.inf)
        self.thread_mpi.append(np.inf)
        self.ga_pe.append(np.inf)
        self.ga_nb.append(np.inf)
        self.ctc_pe.append(np.inf)
        self.ctc_nb.append(np.inf)
        self.thread_pe.append(np.inf)
        self.thread_nb.append(np.inf)


    def add_time(self, i, line, thread):
        time = (float)((line.rsplit('\n')[0]).rsplit(' ')[-1])
        if not thread:
            if "PMPI" in line:
                if time < self.mpi[i]:
                    self.mpi[i] = time
            elif "GPU-Aware" in line:
                if "Pairwise Exchange" in line:
                    if time < self.ga_pe[i]:
                        self.ga_pe[i] = time
                elif "Nonblocking" in line:
                    if time < self.ga_nb[i]:
                        self.ga_nb[i] = time
            elif "Copy-to-CPU" in line:
                if "Pairwise Exchange" in line:
                    if time < self.ctc_pe[i]:
                        self.ctc_pe[i] = time
                elif "Nonblocking" in line:
                    if time < self.ctc_nb[i]:
                        self.ctc_nb[i] = time
        else:
            if "PMPI" in line:
                if time < self.thread_mpi[i]:
                    self.thread_mpi[i] = time
            elif "Pairwise Exchange" in line:
                if time < self.thread_pe[i]:
                    self.thread_pe[i] = time
            elif "Nonblocking" in line: 
                if time < self.thread_nb[i]:
                    self.thread_nb[i] = time


tests = list()
folder = "../%s/gpu_alltoall_malloc"%prof.computer
if std:
    folder = "../%s/gpu_alltoall_std"%prof.computer

for i in range(len(prof.procs)):
    tests.append(Test(prof.procs[i]))
    for file in glob.glob("%s/%s%d.*.out"%(folder, prof.filename, prof.procs[i])):
        idx = -1
        thread = 0
        f = open(file)
        for line in f:
            if "Testing Size" in line:
                if not thread:
                    if idx + 1 == len(tests[-1].sizes):
                        tests[-1].list_append((int)((line.rsplit('\n')[0]).rsplit(' ')[-1]))
                idx = idx + 1
            elif "Time" in line:
                tests[-1].add_time(idx, line, thread)
            elif "./gpu_alltoall" in line:
                thread = 1
                idx = -1
        f.close()


# Plot Largest Proc Count
plot.add_luke_options()
sizes = tests[-1].sizes
mpi = tests[-1].mpi
thread_mpi = tests[-1].thread_mpi
ga_pe = tests[-1].ga_pe
ga_nb = tests[-1].ga_nb
ctc_pe = tests[-1].ctc_pe
ctc_nb = tests[-1].ctc_nb
thread_pe = tests[-1].thread_pe
thread_nb = tests[-1].thread_nb

if std:
    plot.line_plot(mpi, sizes, '-', color='black')
plot.line_plot(ga_pe, sizes, '-', color='red', label = "GPUDirect Pairwise")
plot.line_plot(ctc_pe, sizes, linestyle='dotted', color='red', label = "Copy2CPU Pairwise")
plot.line_plot(thread_pe, sizes, '--', color='red', label = "Thread Pairwise")
#if std:
#    plot.line_plot(thread_mpi, sizes, '--',  color='black', label = "Thread PMPI")
plot.line_plot(ga_nb, sizes, '-', color='blue', label = "GPUDirect Nonblocking")
plot.line_plot(ctc_nb, sizes, linestyle='dotted', color='blue', label = "Copy2CPU Nonblocking")
plot.line_plot(thread_nb, sizes, '--', color='blue', label = "Thread Nonblocking")
plot.add_anchored_legend()
plot.add_labels("Alltoall Count", "Measured Time (Seconds)")
if std:
    plt.savefig("gpu_alltoall32_std.pdf", bbox_inches = "tight")
else:
    plt.savefig("gpu_alltoall32_malloc.pdf", bbox_inches = "tight")
plt.clf()


plot.add_luke_options()
sizes = prof.procs
mpi = [t.mpi[-1] for t in tests]
thread_mpi = [t.thread_mpi[-1] for t in tests]
ga_pe = [t.ga_pe[-1] for t in tests]
ga_nb = [t.ga_nb[-1] for t in tests]
ctc_pe = [t.ctc_pe[-1] for t in tests]
ctc_nb = [t.ctc_nb[-1] for t in tests]
thread_pe = [t.thread_pe[-1] for t in tests]
thread_nb = [t.thread_nb[-1] for t in tests]
ax = plot.get_ax()
ax.set_xscale('log', base=2)
if std:
    plot.line_plot(mpi, sizes, '-', color='black')
#    plot.line_plot(thread_mpi, sizes, '--',  color='black', label = "Thread PMPI")
plot.line_plot(ga_pe, sizes, '-', color='red', label = "GPUDirect Pairwise")
plot.line_plot(ctc_pe, sizes, linestyle='dotted', color='red', label = "Copy2CPU Pairwise")
plot.line_plot(thread_pe, sizes, '--', color='red', label = "Thread Pairwise")
plot.line_plot(ga_nb, sizes, '-', color='blue', label = "GPUDirect Nonblocking")
plot.line_plot(ctc_nb, sizes, linestyle='dotted', color='blue', label = "Copy2CPU Nonblocking")
plot.line_plot(thread_nb, sizes, '--', color='blue', label = "Thread Nonblocking")
plot.add_anchored_legend()
plot.add_labels("Node Count", "Measured Time (Seconds)")
if std:
    plt.savefig("gpu_alltoall_scale_std.pdf", bbox_inches = "tight")
else:
    plt.savefig("gpu_alltoall_scale_malloc.pdf", bbox_inches = "tight")
plt.clf()
