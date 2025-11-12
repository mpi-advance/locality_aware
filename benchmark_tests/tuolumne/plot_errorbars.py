import math
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter

import pyfancyplot as pfp

def find_procspn_gpuspn(l):
    match = re.search(r'(\d+)\s+Processes Per GPU, (\d+)\s+GPUs Per Node', l)
    if match:
        gpus_per_node = int(match.group(2))
        procs_per_node = int(match.group(1)) * gpus_per_node
    return procs_per_node, gpus_per_node

def find_size(l):
    match = re.search(r'Size\s+(\d+)', l)
    if match:
        size = int(match.group(1))
    return size

def find_timings(l):
    matches = re.findall(r'(?:\d+:\s+)?([^:\n]+):\s+([-+]?\d*\.\d+e[-+]\d+)', l)
    out = []
    for k, v in matches:
        out.append((k, float(v)))
    return out

def find_nnodes(fn):
    match = re.search(r'(?<=n)(\d+)', fn)
    if match:
        node_count = int(match.group(1))
    return node_count

def update(times, curr_size, local_times, size, procs_per_gpu, func=min):
    if 'SKIP_PPG' in os.environ.keys() and procs_per_gpu > int(os.environ['SKIP_PPG']):
        return times, size, []
    times[procs_per_gpu][curr_size] = func(local_times)
    return times, size, []

def find_besttimes(lines):
    times = {1: {}}
    for l in lines:
        if "Processes Per GPU" in l and "GPUs Per Node" in l:
            procs_per_node, gpus_per_node = find_procspn_gpuspn(l)
            procs_per_gpu = int(procs_per_node / gpus_per_node)
            if procs_per_gpu not in times.keys():
                if 'SKIP_PPG' in os.environ.keys() and procs_per_gpu > int(os.environ['SKIP_PPG']):
                    continue
                times[procs_per_gpu] = {}
                
    curr_size = 0
    procs_per_gpu = 1  
    local_times = []
    for l in lines:
        if "Processes Per GPU" in l and "GPUs Per Node" in l:
            times, curr_size, local_times = update(times, curr_size, local_times, 0, procs_per_gpu, func=lambda x: x)
            procs_per_node, gpus_per_node = find_procspn_gpuspn(l)
            procs_per_gpu = int(procs_per_node / gpus_per_node)
        elif "Size " in l:
            size = find_size(l)
            if size != curr_size and curr_size != 0:
                times, curr_size, local_times = update(times, curr_size, local_times, size, procs_per_gpu, func=lambda x: x)
            if curr_size == 0:
                curr_size = size
        for (k, v) in find_timings(l):
            local_times.append((k, v))
            
    times, curr_size, local_times = update(times, curr_size, local_times, curr_size, procs_per_gpu, func=lambda x: x)
    return times

# assume sizes pushed in
def combine_times(li_times):
    out = {}
    for times in li_times:
        for procs_per_gpu in times.keys():
            if procs_per_gpu not in out.keys():
                out[procs_per_gpu] = {}
            for k in times[procs_per_gpu].keys():
                if k not in out[procs_per_gpu].keys():
                    out[procs_per_gpu][k] = {}
                for (size, time) in times[procs_per_gpu][k]:
                    if size not in out[procs_per_gpu][k].keys():
                        out[procs_per_gpu][k][size] = []
                    out[procs_per_gpu][k][size].append(time)
    return out

def reduce_combined_times(combined_times, func):
    out = {}
    for procs_per_gpu in combined_times.keys():
        out[procs_per_gpu] = {}
        for k in combined_times[procs_per_gpu].keys():
            out[procs_per_gpu][k] = []
            for size in combined_times[procs_per_gpu][k].keys():
                out[procs_per_gpu][k].append((size, func(combined_times[procs_per_gpu][k][size])))
    return out

def get_min_max_error_diffs(li_local_times):    
    return (np.mean(li_local_times) - np.min(li_local_times), np.max(li_local_times) - np.mean(li_local_times))

def min_max_error_diff_reduced_times_to_np(times_error_diffs):
    out = {}
    for procs_per_gpu in times_error_diffs.keys():
        out[procs_per_gpu] = {}
        for k in times_error_diffs[procs_per_gpu].keys():
            out[procs_per_gpu][k] = np.zeros((3, len(times_error_diffs[procs_per_gpu][k])))
            for ind, (size, (min_time_diff, max_time_diff)) in enumerate(times_error_diffs[procs_per_gpu][k]):
                (out[procs_per_gpu][k])[0, ind] = size
                (out[procs_per_gpu][k])[1, ind] = min_time_diff
                (out[procs_per_gpu][k])[2, ind] = max_time_diff
    return out

def min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(array_2d_ref, lp_avg_ref, array_2d_new, lp_avg_new):
    new_2d_vals = np.copy(array_2d_new)
    new_2d_vals[1, :] = np.array([y[1] for y in lp_avg_new]) - array_2d_new[1, :]
    new_2d_vals[2, :] = np.array([y[1] for y in lp_avg_new]) + array_2d_new[2, :]
    
    ref_2d_vals = np.copy(array_2d_ref)
    ref_2d_vals[1, :] = np.array([y[1] for y in lp_avg_ref]) - array_2d_ref[1, :]
    ref_2d_vals[2, :] = np.array([y[1] for y in lp_avg_ref]) + array_2d_ref[2, :]
    
    mask = np.isin(ref_2d_vals[0, :], new_2d_vals[0, :])
    ref_2d_vals = ref_2d_vals[:, mask]
    
    speedups = np.copy(ref_2d_vals)
    speedups[1, :] = ref_2d_vals[1, :] / new_2d_vals[2, :]  # worst speedup is old min / new max
    speedups[2, :] = ref_2d_vals[2, :] / new_2d_vals[1, :]  # best speedup is old max / new min
    
    avg_speedups = (np.array([y[1] for y in lp_avg_ref]))[mask] / np.array([y[1] for y in lp_avg_new])
    
    speedup_diffs = np.copy(speedups)
    speedup_diffs[1, :] = avg_speedups - speedups[1, :]
    speedup_diffs[2, :] = speedups[2, :] - avg_speedups
    
    return speedup_diffs, speedup_diffs[0, :], avg_speedups

# assume sizes pushed in
def find_max_size_in_all_li_times(li_times):
    def extract_sizes(times):
        all_sizes = None
        for subdict in times.values():
            for inner_list in subdict.values():
                current = {pair[0] for pair in inner_list}
                if all_sizes is None:
                    all_sizes = current
                else:
                    all_sizes = all_sizes.intersection(current)
        return all_sizes if all_sizes is not None else set()

    common_sizes = None
    for times in li_times:
        sizes = extract_sizes(times)
        if common_sizes is None:
            common_sizes = sizes
        else:
            common_sizes = common_sizes.intersection(sizes)

    return max(common_sizes) if common_sizes else None

def num_node_combined_times_pairs_push_num_nodes_in_at_size(node_comb_times_pairs, target_size):
    out = {}
    for num_nodes, combined_times in node_comb_times_pairs:
        for procs_per_gpu in combined_times.keys():
            if procs_per_gpu not in out.keys():
                out[procs_per_gpu] = {}
            for k in combined_times[procs_per_gpu].keys():
                if k not in out[procs_per_gpu].keys():
                    out[procs_per_gpu][k] = {}
                for size in combined_times[procs_per_gpu][k].keys():
                    if size == target_size:
                        local_times = combined_times[procs_per_gpu][k][size]
                        out[procs_per_gpu][k][num_nodes] = local_times
    return out

def filter_sizes(times, min_size):
    out = {}
    for procs_per_gpu in times.keys():
        out[procs_per_gpu] = {}
        for size in times[procs_per_gpu].keys():
            if size >= min_size:
                out[procs_per_gpu][size] = times[procs_per_gpu][size]
    return out

def push_sizes_in(times):
    out = {}
    for procs_per_gpu in times.keys():
        out[procs_per_gpu] = {}
        for size in times[procs_per_gpu].keys():
            for (k, v) in times[procs_per_gpu][size]:
                if k not in out[procs_per_gpu].keys():
                    out[procs_per_gpu][k] = []
                out[procs_per_gpu][k].append((size, v))
    return out

def is_size_in(lp, s):
    return s in [x[0] for x in lp]

if __name__ == "__main__":
    if "NO_TITLE" in os.environ.keys() and int(os.environ["NO_TITLE"]) == 1:
        plt.title = lambda *args, **kwargs: None
    
    fn_in = sys.argv[1]
    nodes_input = sys.argv[2]
    prefix = "allreduce_plus_copy"
    suffix = ".out"
    if len(sys.argv) >= 4:
        prefix = sys.argv[3]
    if len(sys.argv) >= 5:
        suffix = sys.argv[4]
    
    if nodes_input != "all":    
        for n_nodes in list(map(int, nodes_input.split(",")[1:])):
            node_substring = nodes_input.split(",")[0] + str(n_nodes)
            li_lines = []
            for fn in os.listdir(fn_in):
                if node_substring in fn and fn.startswith(prefix) and fn.endswith(suffix) and find_nnodes(node_substring) == find_nnodes(fn):
                    with open(os.path.join(fn_in, fn), 'r') as f:
                        lines = f.readlines()
                        li_lines.append(lines)

            fn_out = os.path.join(fn_in, prefix + "_" + node_substring + "_node_run_errorbar.pdf")

            li_locality_times = []
            for lines in li_lines:
                locality_times = find_besttimes(lines)
                
                locality_times = filter_sizes(locality_times, 256)
                
                locality_times = push_sizes_in(locality_times)
                
                li_locality_times.append(locality_times)
            
            locality_times_combined = combine_times(li_locality_times)
            
            locality_times_avg = reduce_combined_times(locality_times_combined, np.mean)
            
            locality_times_error_diffs = reduce_combined_times(locality_times_combined, get_min_max_error_diffs)
            
            locality_times_error_diffs_np = min_max_error_diff_reduced_times_to_np(locality_times_error_diffs)
            
            pdf = PdfPages(fn_out)
            
            plt.figure()
            plt.errorbar([x[0] for x in locality_times_avg[1]["STD"]], [y[1] for y in locality_times_avg[1]["STD"]], yerr=(locality_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG")
            for ppg in locality_times_avg.keys():
                if ppg == 1:
                    continue
                plt.errorbar([x[0] for x in locality_times_avg[ppg]["STD MPS COPY"]], [y[1] for y in locality_times_avg[ppg]["STD MPS COPY"]], yerr=(locality_times_error_diffs_np[ppg]["STD MPS COPY"])[1:, :], label=str(ppg) + " PPG")
            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3)       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            plt.plot([x[0] for x in locality_times_avg[1]["STD"]], np.array([y[1] for y in locality_times_avg[1]["STD"]]) / np.array([y[1] for y in locality_times_avg[1]["STD"]]), label="1 PPG")
            for ppg in locality_times_avg.keys():
                if ppg == 1:
                    continue
                local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(locality_times_error_diffs_np[1]["STD"], 
                                                                                                            locality_times_avg[1]["STD"], 
                                                                                                            locality_times_error_diffs_np[ppg]["STD MPS COPY"], 
                                                                                                            locality_times_avg[ppg]["STD MPS COPY"])
                plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label=str(ppg) + " PPG")
            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
                
            pdf.close()
    else:
        dict_li_lines = {}
        for fn in os.listdir(fn_in):
            if fn.startswith(prefix) and fn.endswith(suffix):
                num_nodes = find_nnodes(fn)
                if num_nodes > 0:
                    with open(os.path.join(fn_in, fn), 'r') as f:
                        lines = f.readlines()
                        if num_nodes not in dict_li_lines.keys():
                            dict_li_lines[num_nodes] = []
                        dict_li_lines[num_nodes].append(lines)
        dict_li_lines = dict(sorted(dict_li_lines.items()))
        
        fn_out = os.path.join(fn_in, prefix + "_all_nodes_run_errorbar.pdf")

        dict_li_locality_times = {}
        for num_nodes in dict_li_lines.keys():
            if num_nodes not in dict_li_locality_times.keys():
                dict_li_locality_times[num_nodes] = []
            for lines in dict_li_lines[num_nodes]:
                locality_times = find_besttimes(lines)
                
                locality_times = push_sizes_in(locality_times)
                
                dict_li_locality_times[num_nodes].append(locality_times)
                
        target_size = find_max_size_in_all_li_times(sum(list(dict_li_locality_times.values()), []))
        target_power = int(np.log2(target_size))
        print("Target size: 2^" + str(target_power))
        
        locality_num_node_combined_times_pairs = []
        for num_nodes in dict_li_lines.keys():
            locality_times_combined = combine_times(dict_li_locality_times[num_nodes])
            
            locality_num_node_combined_times_pairs.append((num_nodes, locality_times_combined))
        
        # push num nodes in
        locality_times_combined = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_num_node_combined_times_pairs, target_size)
        
        locality_times_avg = reduce_combined_times(locality_times_combined, np.mean)
        
        locality_times_error_diffs = reduce_combined_times(locality_times_combined, get_min_max_error_diffs)
        
        locality_times_error_diffs_np = min_max_error_diff_reduced_times_to_np(locality_times_error_diffs)
               
        pdf = PdfPages(fn_out)
                
        plt.figure()
        plt.errorbar([x[0] for x in locality_times_avg[1]["STD"]], [y[1] for y in locality_times_avg[1]["STD"]], yerr=(locality_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG")
        for ppg in locality_times_avg.keys():
            if ppg == 1:
                continue
            plt.errorbar([x[0] for x in locality_times_avg[ppg]["STD MPS COPY"]], [y[1] for y in locality_times_avg[ppg]["STD MPS COPY"]], yerr=(locality_times_error_diffs_np[ppg]["STD MPS COPY"])[1:, :], label=str(ppg) + " PPG")
        plt.title("Allreduce\nAll nodes at $2^{" + str(target_power) + "}$ floats")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(ticks=sorted(set(dict_li_locality_times.keys()) | {10**i for i in range(math.ceil(math.log10(min(dict_li_locality_times))), math.floor(math.log10(max(dict_li_locality_times)))+1)}))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")
        # plt.legend()
        pfp.add_anchored_legend(ncol=3)
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()
        plt.plot([x[0] for x in locality_times_avg[1]["STD"]], np.array([y[1] for y in locality_times_avg[1]["STD"]]) / np.array([y[1] for y in locality_times_avg[1]["STD"]]), label="1 PPG")
        for ppg in locality_times_avg.keys():
            if ppg == 1:
                continue
            local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(locality_times_error_diffs_np[1]["STD"], 
                                                                                                        locality_times_avg[1]["STD"], 
                                                                                                        locality_times_error_diffs_np[ppg]["STD MPS COPY"], 
                                                                                                        locality_times_avg[ppg]["STD MPS COPY"])
            print(f"speedup:  std ppg {ppg} by std")
            print(local_x)
            print(local_avg)

            plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label=str(ppg) + " PPG")
        plt.title("Allreduce\nAll nodes at $2^{" + str(target_power) + "}$ floats")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(ticks=sorted(set(dict_li_locality_times.keys()) | {10**i for i in range(math.ceil(math.log10(min(dict_li_locality_times))), math.floor(math.log10(max(dict_li_locality_times)))+1)}))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.ylabel("Speedup")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        pdf.close()