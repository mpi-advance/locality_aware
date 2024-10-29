file_in = "../dane/microbench.out"
folder_out = "dane"

import os
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

file = open(file_in)

class Timings:
    sizes = ""
    times = ""

    def __init__(self):
        self.sizes = list()
        self.times = list()

    def add_timing(self, size, time):
        self.sizes.append(size)
        self.times.append(time)

class PingPong:
    on_socket = ""
    off_socket = ""
    off_node = ""

    def __init__(self):
        self.on_socket = Timings()
        self.off_socket = Timings()
        self.off_node = Timings()

class MultiProcTimings:
    active_procs = ""
    multi_proc_timings = ""

    def __init__(self):
        self.active_procs = list()
        self.multi_proc_timings = list()

    def add_active_procs(self, n_active):
        self.active_procs.append(n_active)
        self.multi_proc_timings.append(Timings())

    def add_timing(self, size, time):
        self.multi_proc_timings[-1].add_timing(size, time)

class MultiProcPingPong:
    on_socket =  ""
    off_socket = ""
    on_node = ""
    on_socket_all = ""
    off_node_all = ""

    def __init__(self):
        self.on_socket = MultiProcTimings()
        self.off_socket = MultiProcTimings()
        self.off_node = MultiProcTimings()
        self.on_socket_all = MultiProcTimings()
        self.off_node_all = MultiProcTimings()

standard = PingPong()
multiproc = MultiProcPingPong()
multimsg = PingPong()
matching = PingPong()

group = ""
test = ""

## Collect Standard Data
for line in file:
    if "MultiProc" in line:
        group = multiproc
    elif "Multi" in line:
        group = multimsg
    elif "Matching" in line:
        group = matching
    elif "Ping-Pong" in line:
        group = standard

    if "On-Socket" in line:
        if "All Sockets Active" in line:
            test = group.on_socket_all
        else:
            test = group.on_socket
    elif "Off-Socket" in line:
        test = group.off_socket
    elif "Off-Node" in line:
        if "Even Sockets" in line:
            test = group.off_node_all
        else:
            test = group.off_node

    if "Active Procs" in line:
        active = (int)((line.rsplit('\n')[0]).rsplit(' ')[-1])
        test.add_active_procs(active)

    if "Size" in line:
        data = (line.rsplit('\n')[0]).rsplit(':')
        size = (int)(data[0].rsplit(' ')[-1])
        time = (float)(data[1].rsplit(' ')[-1])
        test.add_timing(size, time)

import pyfancyplot as plot
import matplotlib.backends


## Plot Standard On-Socket vs Off-Socket vs Off-Node
plot.add_luke_options()
plot.line_plot(standard.on_socket.times, x_data = standard.on_socket.sizes, tickmark = '.-', color = 'blue', label = "On-Socket")
plot.line_plot(standard.off_socket.times, x_data = standard.off_socket.sizes, tickmark = '.-', color = 'red', label = "Off-Socket")
plot.line_plot(standard.off_node.times, x_data = standard.off_node.sizes, tickmark = '.-', color = 'black', label = "Off-Node")
plot.add_anchored_legend()
plot.set_scale('log', 'log')
plot.add_labels("Message Size (Bytes)", "Measured Time (Seconds)");
plot.save_plot("%s/standard.pdf"%folder_out)


## Plot On-Socket PPN
plot.add_luke_options()
plot.set_palette(n_colors=len(multiproc.on_socket.active_procs))

for i in range(len(multiproc.on_socket.active_procs)):
    n_proc = multiproc.on_socket.active_procs[i]
    data = multiproc.on_socket.multi_proc_timings[i]
    plot.line_plot(data.times, x_data = data.sizes, tickmark = ".-", label = "%d Active"%(n_proc))
plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.add_labels("Message Size (Bytes)", "Measured Time (Seconds)");
plot.save_plot("%s/multiproc-on-socket.pdf"%folder_out)

## Plot Off-Socket PPN
plot.add_luke_options()
plot.set_palette(n_colors=len(multiproc.off_socket.active_procs))

for i in range(len(multiproc.off_socket.active_procs)):
    n_proc = multiproc.off_socket.active_procs[i]
    data = multiproc.off_socket.multi_proc_timings[i]
    plot.line_plot(data.times, x_data = data.sizes, tickmark = ".-", label = "%d Active"%(n_proc))
plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.add_labels("Message Size (Bytes)", "Measured Time (Seconds)");
plot.save_plot("%s/multiproc-off-socket.pdf"%folder_out)

## Plot Off-Node PPN
plot.add_luke_options()
plot.set_palette(n_colors=len(multiproc.off_node.active_procs))

for i in range(len(multiproc.off_node.active_procs)):
    n_proc = multiproc.off_node.active_procs[i]
    data = multiproc.off_node.multi_proc_timings[i]
    plot.line_plot(data.times, x_data = data.sizes, tickmark = ".-", label = "%d Active"%(n_proc))
plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.add_labels("Message Size (Bytes)", "Measured Time (Seconds)");
plot.save_plot("%s/multiproc-off-node.pdf"%folder_out)


## Plot On-Socket PPN, All Active 
plot.add_luke_options()
plot.set_palette(n_colors=len(multiproc.on_socket_all.active_procs))

for i in range(len(multiproc.on_socket_all.active_procs)):
    n_proc = multiproc.on_socket_all.active_procs[i]
    data = multiproc.on_socket_all.multi_proc_timings[i]
    plot.line_plot(data.times, x_data = data.sizes, tickmark = ".-", label = "%d Active"%(n_proc))
plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.save_plot("%s/multiproc-on-all-sockets.pdf"%folder_out)

## Plot Off-Node PPN, Even Sockets 
plot.add_luke_options()
plot.set_palette(n_colors=len(multiproc.off_node_all.active_procs))

for i in range(len(multiproc.off_node_all.active_procs)):
    n_proc = multiproc.off_node_all.active_procs[i]
    data = multiproc.off_node_all.multi_proc_timings[i]
    data_orig = multiproc.off_node.multi_proc_timings[i+1]
    ctr = plot.color_ctr
    plot.line_plot(data.times, x_data = data.sizes, tickmark = ".-", label = "%d Active"%(n_proc*2))
    plot.color_ctr = ctr
    plot.line_plot(data_orig.times, x_data = data.sizes, tickmark = ":")

plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.save_plot("%s/multiproc-off-node-even.pdf"%folder_out)

# Plot Multiple Messages
plot.add_luke_options()
plot.line_plot(multimsg.on_socket.times, x_data = multimsg.on_socket.sizes, tickmark=".-", label = "On-Socket")
plot.line_plot(multimsg.off_socket.times, x_data = multimsg.off_socket.sizes, tickmark=".-", label = "Off-Socket")
plot.line_plot(multimsg.off_node.times, x_data = multimsg.off_node.sizes, tickmark=".-", label = "Off-Node")
plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.save_plot("%s/multimsg.pdf"%folder_out)


# Plot Queue Searches
plot.add_luke_options()

ctr = plot.color_ctr
plot.line_plot(multimsg.on_socket.times, x_data = multimsg.on_socket.sizes, tickmark=":", label = "On-Socket")
plot.color_ctr = ctr
plot.line_plot(matching.on_socket.times, x_data = matching.on_socket.sizes, tickmark="-")

ctr = plot.color_ctr
plot.line_plot(multimsg.off_socket.times, x_data = multimsg.off_socket.sizes, tickmark=":", label = "Off-Socket")
plot.color_ctr = ctr
plot.line_plot(matching.off_socket.times, x_data = matching.off_socket.sizes, tickmark="-")

ctr = plot.color_ctr
plot.line_plot(multimsg.off_node.times, x_data = multimsg.off_node.sizes, tickmark=":", label = "Off-Node")
plot.color_ctr = ctr
plot.line_plot(matching.off_node.times, x_data = matching.off_node.sizes, tickmark="-")

plot.add_anchored_legend(ncols=3)
plot.set_scale('log', 'log')
plot.add_labels("Number of Messages (Each 1 Byte)", "Measured Time (Seconds)");
plot.save_plot("%s/matching.pdf"%folder_out)


