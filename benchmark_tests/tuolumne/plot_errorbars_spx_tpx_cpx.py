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
    nodes_input = sys.argv[2]  # _n,2,4,8,... or all,_n
    prefix = "allreduce_plus_copy"
    prefix_tpx = prefix + "_tpx"
    prefix_cpx = prefix + "_cpx"
    suffix = ".out"
    
    if not nodes_input.startswith("all"):    
        cut_size = 256
        for n_nodes in list(map(int, nodes_input.split(",")[1:])):
            node_substring = nodes_input.split(",")[0] + str(n_nodes)
            li_lines = []
            li_tpx_lines = []
            li_cpx_lines = []
            for fn in os.listdir(fn_in):
                if node_substring in fn and fn.endswith(suffix) and find_nnodes(node_substring) == find_nnodes(fn):
                    which_li_lines = None
                    if fn.startswith(prefix + node_substring):
                        which_li_lines = li_lines
                    elif fn.startswith(prefix_tpx + node_substring):
                        which_li_lines = li_tpx_lines
                    elif fn.startswith(prefix_cpx + node_substring):
                        which_li_lines = li_cpx_lines
                    else:
                        continue
                    with open(os.path.join(fn_in, fn), 'r') as f:
                        lines = f.readlines()
                        which_li_lines.append(lines)

            fn_out = os.path.join(fn_in, prefix + node_substring + "_node_run_errorbar_spx_tpx_cpx.pdf")

            li_locality_times = []
            li_locality_tpx_times = []
            li_locality_cpx_times = []
            for lines in li_lines:
                locality_times = find_besttimes(lines)
                
                locality_times = filter_sizes(locality_times, cut_size)
                
                locality_times = push_sizes_in(locality_times)
                
                li_locality_times.append(locality_times)
            for lines in li_tpx_lines:
                locality_times = find_besttimes(lines)
                
                locality_times = filter_sizes(locality_times, cut_size)
                
                locality_times = push_sizes_in(locality_times)
                
                li_locality_tpx_times.append(locality_times)
            for lines in li_cpx_lines:
                locality_times = find_besttimes(lines)
                
                locality_times = filter_sizes(locality_times, cut_size)
                
                locality_times = push_sizes_in(locality_times)
                
                li_locality_cpx_times.append(locality_times)
            
            locality_times_combined = combine_times(li_locality_times)
            locality_tpx_times_combined = combine_times(li_locality_tpx_times)
            locality_cpx_times_combined = combine_times(li_locality_cpx_times)
            
            locality_times_avg = reduce_combined_times(locality_times_combined, np.mean)
            locality_tpx_times_avg = reduce_combined_times(locality_tpx_times_combined, np.mean)
            locality_cpx_times_avg = reduce_combined_times(locality_cpx_times_combined, np.mean)
            
            locality_times_error_diffs = reduce_combined_times(locality_times_combined, get_min_max_error_diffs)
            locality_tpx_times_error_diffs = reduce_combined_times(locality_tpx_times_combined, get_min_max_error_diffs)
            locality_cpx_times_error_diffs = reduce_combined_times(locality_cpx_times_combined, get_min_max_error_diffs)
            
            locality_times_error_diffs_np = min_max_error_diff_reduced_times_to_np(locality_times_error_diffs)
            locality_tpx_times_error_diffs_np = min_max_error_diff_reduced_times_to_np(locality_tpx_times_error_diffs)
            locality_cpx_times_error_diffs_np = min_max_error_diff_reduced_times_to_np(locality_cpx_times_error_diffs)
            
            pdf = PdfPages(fn_out)
            
            plt.figure()
            plt.errorbar([x[0] for x in locality_times_avg[1]["STD"]], [y[1] for y in locality_times_avg[1]["STD"]], yerr=(locality_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[1]["STD"]], [y[1] for y in locality_tpx_times_avg[1]["STD"]], yerr=(locality_tpx_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG, TPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[1]["STD"]], [y[1] for y in locality_cpx_times_avg[1]["STD"]], yerr=(locality_cpx_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG, CPX")
            plt.errorbar([x[0] for x in locality_times_avg[2]["STD MPS COPY"]], [y[1] for y in locality_times_avg[2]["STD MPS COPY"]], yerr=(locality_times_error_diffs_np[2]["STD MPS COPY"])[1:, :], label="2 PPG")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[2]["STD MPS COPY"]], [y[1] for y in locality_tpx_times_avg[2]["STD MPS COPY"]], yerr=(locality_tpx_times_error_diffs_np[2]["STD MPS COPY"])[1:, :], label="2 PPG, TPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[2]["STD MPS COPY"]], [y[1] for y in locality_cpx_times_avg[2]["STD MPS COPY"]], yerr=(locality_cpx_times_error_diffs_np[2]["STD MPS COPY"])[1:, :], label="2 PPG, CPX")
            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            plt.plot([x[0] for x in locality_times_avg[1]["STD"]], np.array([y[1] for y in locality_times_avg[1]["STD"]]) / np.array([y[1] for y in locality_times_avg[1]["STD"]]), label="1 PPG")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_tpx_times_error_diffs_np[1]["STD"],
                    locality_tpx_times_avg[1]["STD"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="1 PPG, TPX")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_cpx_times_error_diffs_np[1]["STD"],
                    locality_cpx_times_avg[1]["STD"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="1 PPG, CPX")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_times_error_diffs_np[2]["STD MPS COPY"],
                    locality_times_avg[2]["STD MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="2 PPG")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_tpx_times_error_diffs_np[2]["STD MPS COPY"],
                    locality_tpx_times_avg[2]["STD MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="2 PPG, TPX")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_cpx_times_error_diffs_np[2]["STD MPS COPY"],
                    locality_cpx_times_avg[2]["STD MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="2 PPG, CPX")

            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            plt.errorbar([x[0] for x in locality_times_avg[1]["MIKELANE"]], [y[1] for y in locality_times_avg[1]["MIKELANE"]], yerr=(locality_times_error_diffs_np[1]["MIKELANE"])[1:, :], label="Lane, 1 PPG")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[1]["MIKELANE"]], [y[1] for y in locality_tpx_times_avg[1]["MIKELANE"]], yerr=(locality_tpx_times_error_diffs_np[1]["MIKELANE"])[1:, :], label="Lane, 1 PPG, TPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[1]["MIKELANE"]], [y[1] for y in locality_cpx_times_avg[1]["MIKELANE"]], yerr=(locality_cpx_times_error_diffs_np[1]["MIKELANE"])[1:, :], label="Lane, 1 PPG, CPX")
            plt.errorbar([x[0] for x in locality_times_avg[2]["MIKELANE MPS COPY"]], [y[1] for y in locality_times_avg[2]["MIKELANE MPS COPY"]], yerr=(locality_times_error_diffs_np[2]["MIKELANE MPS COPY"])[1:, :], label="Lane, 2 PPG")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[2]["MIKELANE MPS COPY"]], [y[1] for y in locality_tpx_times_avg[2]["MIKELANE MPS COPY"]], yerr=(locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"])[1:, :], label="Lane, 2 PPG, TPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[2]["MIKELANE MPS COPY"]], [y[1] for y in locality_cpx_times_avg[2]["MIKELANE MPS COPY"]], yerr=(locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"])[1:, :], label="Lane, 2 PPG, CPX")
            plt.title("Lane Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            plt.plot([x[0] for x in locality_times_avg[1]["MIKELANE"]], np.array([y[1] for y in locality_times_avg[1]["MIKELANE"]]) / np.array([y[1] for y in locality_times_avg[1]["MIKELANE"]]), label="Lane, 1 PPG")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["MIKELANE"],
                    locality_times_avg[1]["MIKELANE"],
                    locality_tpx_times_error_diffs_np[1]["MIKELANE"],
                    locality_tpx_times_avg[1]["MIKELANE"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 1 PPG, TPX")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["MIKELANE"],
                    locality_times_avg[1]["MIKELANE"],
                    locality_cpx_times_error_diffs_np[1]["MIKELANE"],
                    locality_cpx_times_avg[1]["MIKELANE"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 1 PPG, CPX")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["MIKELANE"],
                    locality_times_avg[1]["MIKELANE"],
                    locality_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                    locality_times_avg[2]["MIKELANE MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 2 PPG")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["MIKELANE"],
                    locality_times_avg[1]["MIKELANE"],
                    locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                    locality_tpx_times_avg[2]["MIKELANE MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 2 PPG, TPX")

            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["MIKELANE"],
                    locality_times_avg[1]["MIKELANE"],
                    locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                    locality_cpx_times_avg[2]["MIKELANE MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 2 PPG, CPX")

            plt.title("Lane Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            plt.errorbar([x[0] for x in locality_times_avg[1]["STD"]],
                        [y[1] for y in locality_times_avg[1]["STD"]],
                        yerr=(locality_times_error_diffs_np[1]["STD"])[1:, :],
                        label="1 PPG, SPX")

            plt.errorbar([x[0] for x in locality_times_avg[2]["STD MPS COPY"]],
                        [y[1] for y in locality_times_avg[2]["STD MPS COPY"]],
                        yerr=(locality_times_error_diffs_np[2]["STD MPS COPY"])[1:, :],
                        label="2 PPG, SPX")

            plt.errorbar([x[0] for x in locality_times_avg[1]["MIKELANE"]],
                        [y[1] for y in locality_times_avg[1]["MIKELANE"]],
                        yerr=(locality_times_error_diffs_np[1]["MIKELANE"])[1:, :],
                        label="Lane, 1 PPG, SPX")

            plt.errorbar([x[0] for x in locality_times_avg[2]["MIKELANE MPS COPY"]],
                        [y[1] for y in locality_times_avg[2]["MIKELANE MPS COPY"]],
                        yerr=(locality_times_error_diffs_np[2]["MIKELANE MPS COPY"])[1:, :],
                        label="Lane, 2 PPG, SPX")

            plt.title("Lane+Std Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A SPX")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
            plt.tight_layout()
            pdf.savefig(plt.gcf())

            plt.figure()

            # Baseline: 1-PPG STD SPX → normalized
            plt.plot([x[0] for x in locality_times_avg[1]["STD"]],
                    np.array([y[1] for y in locality_times_avg[1]["STD"]]) /
                    np.array([y[1] for y in locality_times_avg[1]["STD"]]),
                    label="1 PPG, SPX")

            # 2-PPG STD SPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_times_error_diffs_np[2]["STD MPS COPY"],
                    locality_times_avg[2]["STD MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="2 PPG, SPX")

            # Lane, 1-PPG SPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_times_error_diffs_np[1]["MIKELANE"],
                    locality_times_avg[1]["MIKELANE"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 1 PPG, SPX")

            # Lane, 2-PPG SPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_times_error_diffs_np[1]["STD"],
                    locality_times_avg[1]["STD"],
                    locality_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                    locality_times_avg[2]["MIKELANE MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 2 PPG, SPX")

            plt.title("Lane+Std Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A SPX")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
                
            plt.figure()
            plt.errorbar([x[0] for x in locality_tpx_times_avg[1]["STD"]], [y[1] for y in locality_tpx_times_avg[1]["STD"]], yerr=(locality_tpx_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG, TPX")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[2]["STD MPS COPY"]], [y[1] for y in locality_tpx_times_avg[2]["STD MPS COPY"]], yerr=(locality_tpx_times_error_diffs_np[2]["STD MPS COPY"])[1:, :], label="2 PPG, TPX")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[1]["MIKELANE"]], [y[1] for y in locality_tpx_times_avg[1]["MIKELANE"]], yerr=(locality_tpx_times_error_diffs_np[1]["MIKELANE"])[1:, :], label="Lane, 1 PPG, TPX")
            plt.errorbar([x[0] for x in locality_tpx_times_avg[2]["MIKELANE MPS COPY"]], [y[1] for y in locality_tpx_times_avg[2]["MIKELANE MPS COPY"]], yerr=(locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"])[1:, :], label="Lane, 2 PPG, TPX")
            plt.title("Lane+Std Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A TPX")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()

            # Baseline: 1-PPG STD TPX → normalized to 1
            plt.plot([x[0] for x in locality_tpx_times_avg[1]["STD"]],
                    np.array([y[1] for y in locality_tpx_times_avg[1]["STD"]]) /
                    np.array([y[1] for y in locality_tpx_times_avg[1]["STD"]]),
                    label="1 PPG, TPX")

            # 2-PPG STD TPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_tpx_times_error_diffs_np[1]["STD"],
                    locality_tpx_times_avg[1]["STD"],
                    locality_tpx_times_error_diffs_np[2]["STD MPS COPY"],
                    locality_tpx_times_avg[2]["STD MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="2 PPG, TPX")

            # Lane, 1-PPG TPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_tpx_times_error_diffs_np[1]["STD"],
                    locality_tpx_times_avg[1]["STD"],
                    locality_tpx_times_error_diffs_np[1]["MIKELANE"],
                    locality_tpx_times_avg[1]["MIKELANE"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 1 PPG, TPX")

            # Lane, 2-PPG TPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_tpx_times_error_diffs_np[1]["STD"],
                    locality_tpx_times_avg[1]["STD"],
                    locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                    locality_tpx_times_avg[2]["MIKELANE MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 2 PPG, TPX")

            plt.title("Lane+Std Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A TPX")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            plt.errorbar([x[0] for x in locality_cpx_times_avg[1]["STD"]], [y[1] for y in locality_cpx_times_avg[1]["STD"]], yerr=(locality_cpx_times_error_diffs_np[1]["STD"])[1:, :], label="1 PPG, CPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[2]["STD MPS COPY"]], [y[1] for y in locality_cpx_times_avg[2]["STD MPS COPY"]], yerr=(locality_cpx_times_error_diffs_np[2]["STD MPS COPY"])[1:, :], label="2 PPG, CPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[1]["MIKELANE"]], [y[1] for y in locality_cpx_times_avg[1]["MIKELANE"]], yerr=(locality_cpx_times_error_diffs_np[1]["MIKELANE"])[1:, :], label="Lane, 1 PPG, CPX")
            plt.errorbar([x[0] for x in locality_cpx_times_avg[2]["MIKELANE MPS COPY"]], [y[1] for y in locality_cpx_times_avg[2]["MIKELANE MPS COPY"]], yerr=(locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"])[1:, :], label="Lane, 2 PPG, CPX")
            plt.title("Lane+Std Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A CPX")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()

            # Baseline: 1-PPG STD CPX → normalized to 1
            plt.plot([x[0] for x in locality_cpx_times_avg[1]["STD"]],
                    np.array([y[1] for y in locality_cpx_times_avg[1]["STD"]]) /
                    np.array([y[1] for y in locality_cpx_times_avg[1]["STD"]]),
                    label="1 PPG, CPX")

            # 2-PPG STD CPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_cpx_times_error_diffs_np[1]["STD"],
                    locality_cpx_times_avg[1]["STD"],
                    locality_cpx_times_error_diffs_np[2]["STD MPS COPY"],
                    locality_cpx_times_avg[2]["STD MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="2 PPG, CPX")

            # Lane, 1-PPG CPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_cpx_times_error_diffs_np[1]["STD"],
                    locality_cpx_times_avg[1]["STD"],
                    locality_cpx_times_error_diffs_np[1]["MIKELANE"],
                    locality_cpx_times_avg[1]["MIKELANE"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 1 PPG, CPX")

            # Lane, 2-PPG CPX
            local_speedup_error_diffs, local_x, local_avg = \
                min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                    locality_cpx_times_error_diffs_np[1]["STD"],
                    locality_cpx_times_avg[1]["STD"],
                    locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                    locality_cpx_times_avg[2]["MIKELANE MPS COPY"]
                )

            plt.errorbar(local_x, local_avg,
                        yerr=local_speedup_error_diffs[1:, :],
                        label="Lane, 2 PPG, CPX")
            
            plt.title("Lane+Std Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A CPX")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            PX_COLORS = {
                "SPX": "green",
                "TPX": "blue",
                "CPX": "red",
            }
            
            PX_LANE_COLORS = {
                "SPX": "green",
                "TPX": "blue",
                "CPX": "red",
            }
            
            plt.figure()
            # ---- SPX ----
            plt.errorbar(
                [x[0] for x in locality_times_avg[1]["STD"]],
                [y[1] for y in locality_times_avg[1]["STD"]],
                yerr=locality_times_error_diffs_np[1]["STD"][1:, :],
                color=PX_COLORS["SPX"], linestyle="-", label="1 PPG, SPX"
            )

            plt.errorbar(
                [x[0] for x in locality_times_avg[2]["STD MPS COPY"]],
                [y[1] for y in locality_times_avg[2]["STD MPS COPY"]],
                yerr=locality_times_error_diffs_np[2]["STD MPS COPY"][1:, :],
                color=PX_COLORS["SPX"], linestyle="--", label="2 PPG, SPX"
            )

            # ---- TPX ----
            plt.errorbar(
                [x[0] for x in locality_tpx_times_avg[1]["STD"]],
                [y[1] for y in locality_tpx_times_avg[1]["STD"]],
                yerr=locality_tpx_times_error_diffs_np[1]["STD"][1:, :],
                color=PX_COLORS["TPX"], linestyle="-", label="1 PPG, TPX"
            )

            plt.errorbar(
                [x[0] for x in locality_tpx_times_avg[2]["STD MPS COPY"]],
                [y[1] for y in locality_tpx_times_avg[2]["STD MPS COPY"]],
                yerr=locality_tpx_times_error_diffs_np[2]["STD MPS COPY"][1:, :],
                color=PX_COLORS["TPX"], linestyle="--", label="2 PPG, TPX"
            )

            # ---- CPX ----
            plt.errorbar(
                [x[0] for x in locality_cpx_times_avg[1]["STD"]],
                [y[1] for y in locality_cpx_times_avg[1]["STD"]],
                yerr=locality_cpx_times_error_diffs_np[1]["STD"][1:, :],
                color=PX_COLORS["CPX"], linestyle="-", label="1 PPG, CPX"
            )

            plt.errorbar(
                [x[0] for x in locality_cpx_times_avg[2]["STD MPS COPY"]],
                [y[1] for y in locality_cpx_times_avg[2]["STD MPS COPY"]],
                yerr=locality_cpx_times_error_diffs_np[2]["STD MPS COPY"][1:, :],
                color=PX_COLORS["CPX"], linestyle="--", label="2 PPG, CPX"
            )            
            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            # ---- SPX speedup vs SPX 1-PPG ----
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_times_error_diffs_np[1]["STD"],
                locality_times_avg[1]["STD"],
                locality_times_error_diffs_np[2]["STD MPS COPY"],
                locality_times_avg[2]["STD MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_COLORS["SPX"], linestyle="--", label="2 PPG, SPX")

            # ---- TPX speedup vs TPX 1-PPG ----
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_tpx_times_error_diffs_np[1]["STD"],
                locality_tpx_times_avg[1]["STD"],
                locality_tpx_times_error_diffs_np[2]["STD MPS COPY"],
                locality_tpx_times_avg[2]["STD MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_COLORS["TPX"], linestyle="--", label="2 PPG, TPX")

            # ---- CPX speedup vs CPX 1-PPG ----
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_cpx_times_error_diffs_np[1]["STD"],
                locality_cpx_times_avg[1]["STD"],
                locality_cpx_times_error_diffs_np[2]["STD MPS COPY"],
                locality_cpx_times_avg[2]["STD MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_COLORS["CPX"], linestyle="--", label="2 PPG, CPX")
            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.ylim((0., 1.5))
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            # SPX
            plt.errorbar([x[0] for x in locality_times_avg[1]["MIKELANE"]],
                        [y[1] for y in locality_times_avg[1]["MIKELANE"]],
                        yerr=locality_times_error_diffs_np[1]["MIKELANE"][1:,:],
                        color=PX_LANE_COLORS["SPX"], linestyle="-", label="Lane, 1 PPG, SPX")

            plt.errorbar([x[0] for x in locality_times_avg[2]["MIKELANE MPS COPY"]],
                        [y[1] for y in locality_times_avg[2]["MIKELANE MPS COPY"]],
                        yerr=locality_times_error_diffs_np[2]["MIKELANE MPS COPY"][1:,:],
                        color=PX_LANE_COLORS["SPX"], linestyle="--", label="Lane, 2 PPG, SPX")

            # TPX
            plt.errorbar([x[0] for x in locality_tpx_times_avg[1]["MIKELANE"]],
                        [y[1] for y in locality_tpx_times_avg[1]["MIKELANE"]],
                        yerr=locality_tpx_times_error_diffs_np[1]["MIKELANE"][1:,:],
                        color=PX_LANE_COLORS["TPX"], linestyle="-", label="Lane, 1 PPG, TPX")

            plt.errorbar([x[0] for x in locality_tpx_times_avg[2]["MIKELANE MPS COPY"]],
                        [y[1] for y in locality_tpx_times_avg[2]["MIKELANE MPS COPY"]],
                        yerr=locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"][1:,:],
                        color=PX_LANE_COLORS["TPX"], linestyle="--", label="Lane, 2 PPG, TPX")

            # CPX
            plt.errorbar([x[0] for x in locality_cpx_times_avg[1]["MIKELANE"]],
                        [y[1] for y in locality_cpx_times_avg[1]["MIKELANE"]],
                        yerr=locality_cpx_times_error_diffs_np[1]["MIKELANE"][1:,:],
                        color=PX_LANE_COLORS["CPX"], linestyle="-", label="Lane, 1 PPG, CPX")

            plt.errorbar([x[0] for x in locality_cpx_times_avg[2]["MIKELANE MPS COPY"]],
                        [y[1] for y in locality_cpx_times_avg[2]["MIKELANE MPS COPY"]],
                        yerr=locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"][1:,:],
                        color=PX_LANE_COLORS["CPX"], linestyle="--", label="Lane, 2 PPG, CPX")
            plt.title("Lane Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Time (Seconds)")
            plt.yscale("log")
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            # SPX
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_times_error_diffs_np[1]["MIKELANE"],
                locality_times_avg[1]["MIKELANE"],
                locality_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                locality_times_avg[2]["MIKELANE MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_LANE_COLORS["SPX"], linestyle="--", label="Lane, 2 PPG, SPX")

            # TPX
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_tpx_times_error_diffs_np[1]["MIKELANE"],
                locality_tpx_times_avg[1]["MIKELANE"],
                locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                locality_tpx_times_avg[2]["MIKELANE MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_LANE_COLORS["TPX"], linestyle="--", label="Lane, 2 PPG, TPX")

            # CPX
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_cpx_times_error_diffs_np[1]["MIKELANE"],
                locality_cpx_times_avg[1]["MIKELANE"],
                locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                locality_cpx_times_avg[2]["MIKELANE MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_LANE_COLORS["CPX"], linestyle="--", label="Lane, 2 PPG, CPX")
            plt.title("Lane Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.ylim((0., 3.5))
            plt.legend()
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            
            plt.figure()
            # ---- SPX speedup vs SPX 1-PPG ----
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_times_error_diffs_np[1]["STD"],
                locality_times_avg[1]["STD"],
                locality_times_error_diffs_np[2]["STD MPS COPY"],
                locality_times_avg[2]["STD MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_COLORS["SPX"], linestyle="-", label="2 PPG, SPX")
            
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_times_error_diffs_np[1]["STD"],
                locality_times_avg[1]["STD"],
                locality_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                locality_times_avg[2]["MIKELANE MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_LANE_COLORS["SPX"], linestyle="--", label="Lane, 2 PPG, SPX")

            # ---- TPX speedup vs TPX 1-PPG ----
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_tpx_times_error_diffs_np[1]["STD"],
                locality_tpx_times_avg[1]["STD"],
                locality_tpx_times_error_diffs_np[2]["STD MPS COPY"],
                locality_tpx_times_avg[2]["STD MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_COLORS["TPX"], linestyle="-", label="2 PPG, TPX")
            
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_tpx_times_error_diffs_np[1]["STD"],
                locality_tpx_times_avg[1]["STD"],
                locality_tpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                locality_tpx_times_avg[2]["MIKELANE MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_LANE_COLORS["TPX"], linestyle="--", label="Lane, 2 PPG, TPX")

            # ---- CPX speedup vs CPX 1-PPG ----
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_cpx_times_error_diffs_np[1]["STD"],
                locality_cpx_times_avg[1]["STD"],
                locality_cpx_times_error_diffs_np[2]["STD MPS COPY"],
                locality_cpx_times_avg[2]["STD MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_COLORS["CPX"], linestyle="-", label="2 PPG, CPX")
            e,x,avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_cpx_times_error_diffs_np[1]["STD"],
                locality_cpx_times_avg[1]["STD"],
                locality_cpx_times_error_diffs_np[2]["MIKELANE MPS COPY"],
                locality_cpx_times_avg[2]["MIKELANE MPS COPY"]
            )
            plt.errorbar(x, avg, yerr=e[1:,:],
                        color=PX_LANE_COLORS["CPX"], linestyle="--", label="Lane, 2 PPG, CPX")            
            plt.title("Allreduce\n" + str(find_nnodes(node_substring)) + " nodes, MI300A modes")
            plt.xlabel("Num Floats")
            plt.xscale("log")
            plt.ylabel("Speedup")
            plt.ylim((0., 3.5))
            # plt.legend()
            pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
            plt.tight_layout()
            pdf.savefig(plt.gcf())
                
            pdf.close()
    else:
        prefix_node_suffix = nodes_input.split(",")[1]
        
        dict_li_lines = {}
        dict_li_tpx_lines = {}
        dict_li_cpx_lines = {}
        for fn in os.listdir(fn_in):
            if fn.startswith(prefix) and fn.endswith(suffix):
                num_nodes = find_nnodes(fn)
            else:
                continue
                
            which_dict = None
            if fn.startswith(prefix + prefix_node_suffix + str(num_nodes)):
                which_dict = dict_li_lines
            elif fn.startswith(prefix_tpx + prefix_node_suffix + str(num_nodes)):
                which_dict = dict_li_tpx_lines
            elif fn.startswith(prefix_cpx + prefix_node_suffix + str(num_nodes)):
                which_dict = dict_li_cpx_lines
            else:
                continue

            if num_nodes > 0:
                with open(os.path.join(fn_in, fn), 'r') as f:
                    lines = f.readlines()
                    if num_nodes not in which_dict.keys():
                        which_dict[num_nodes] = []
                    which_dict[num_nodes].append(lines)
        dict_li_lines = dict(sorted(dict_li_lines.items()))
        dict_li_tpx_lines = dict(sorted(dict_li_tpx_lines.items()))
        dict_li_cpx_lines = dict(sorted(dict_li_cpx_lines.items()))
        
        fn_out = os.path.join(fn_in, prefix + "_all_nodes_run_errorbar_spx_tpx_cpx.pdf")

        dict_li_locality_times = {}
        dict_li_tpx_locality_times = {}
        dict_li_cpx_locality_times = {}

        for num_nodes in dict_li_lines.keys():

            if num_nodes not in dict_li_locality_times.keys():
                dict_li_locality_times[num_nodes] = []
            if num_nodes not in dict_li_tpx_locality_times.keys():
                dict_li_tpx_locality_times[num_nodes] = []
            if num_nodes not in dict_li_cpx_locality_times.keys():
                dict_li_cpx_locality_times[num_nodes] = []

            for lines in dict_li_lines[num_nodes]:
                locality_times = find_besttimes(lines)
                locality_times = push_sizes_in(locality_times)
                dict_li_locality_times[num_nodes].append(locality_times)

            for lines in dict_li_tpx_lines.get(num_nodes, []):
                locality_times = find_besttimes(lines)
                locality_times = push_sizes_in(locality_times)
                dict_li_tpx_locality_times[num_nodes].append(locality_times)

            for lines in dict_li_cpx_lines.get(num_nodes, []):
                locality_times = find_besttimes(lines)
                locality_times = push_sizes_in(locality_times)
                dict_li_cpx_locality_times[num_nodes].append(locality_times)
                
        # Combined SPX + TPX + CPX for global target size
        target_size_all = find_max_size_in_all_li_times(
            sum(list(dict_li_locality_times.values()), []) +
            sum(list(dict_li_tpx_locality_times.values()), []) +
            sum(list(dict_li_cpx_locality_times.values()), [])
        )
        target_power_all = int(np.log2(target_size_all))
        print("Global target size (SPX+TPX+CPX): 2^" + str(target_power_all))

        # SPX only
        target_size_spx = find_max_size_in_all_li_times(
            sum(list(dict_li_locality_times.values()), [])
        )
        target_power_spx = int(np.log2(target_size_spx))
        print("SPX target size: 2^" + str(target_power_spx))

        # TPX only
        target_size_tpx = find_max_size_in_all_li_times(
            sum(list(dict_li_tpx_locality_times.values()), [])
        )
        target_power_tpx = int(np.log2(target_size_tpx))
        print("TPX target size: 2^" + str(target_power_tpx))

        # CPX only
        target_size_cpx = find_max_size_in_all_li_times(
            sum(list(dict_li_cpx_locality_times.values()), [])
        )
        target_power_cpx = int(np.log2(target_size_cpx))
        print("CPX target size: 2^" + str(target_power_cpx))
        
        # TPX + CPX
        target_size_tpx_cpx = find_max_size_in_all_li_times(
            sum(list(dict_li_tpx_locality_times.values()), []) +
            sum(list(dict_li_cpx_locality_times.values()), [])
        )
        target_power_tpx_cpx = int(np.log2(target_size_tpx_cpx))
        print("TPX+CPX target size: 2^" + str(target_power_tpx_cpx))
        
        # Combined times pairs for all three modes
        locality_num_node_combined_times_pairs = []
        locality_tpx_num_node_combined_times_pairs = []
        locality_cpx_num_node_combined_times_pairs = []

        for num_nodes in dict_li_locality_times.keys():
            # SPX
            combined_times = combine_times(dict_li_locality_times[num_nodes])
            locality_num_node_combined_times_pairs.append((num_nodes, combined_times))
            
            # TPX
            combined_tpx_times = combine_times(dict_li_tpx_locality_times[num_nodes])
            locality_tpx_num_node_combined_times_pairs.append((num_nodes, combined_tpx_times))
            
            # CPX
            combined_cpx_times = combine_times(dict_li_cpx_locality_times[num_nodes])
            locality_cpx_num_node_combined_times_pairs.append((num_nodes, combined_cpx_times))
        
        # push num nodes in
        # ALL modes
        locality_all_spx_combined     = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_num_node_combined_times_pairs, target_size_all)
        locality_all_tpx_combined     = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_tpx_num_node_combined_times_pairs, target_size_all)
        locality_all_cpx_combined     = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_cpx_num_node_combined_times_pairs, target_size_all)

        # SPX only
        locality_spx_spx_combined     = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_num_node_combined_times_pairs, target_size_spx)

        # TPX only
        locality_tpx_tpx_combined     = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_tpx_num_node_combined_times_pairs, target_size_tpx)

        # CPX only
        locality_cpx_cpx_combined     = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_cpx_num_node_combined_times_pairs, target_size_cpx)

        # TPX + CPX
        locality_tpx_cpx_tpx_combined = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_tpx_num_node_combined_times_pairs, target_size_tpx_cpx)
        locality_tpx_cpx_cpx_combined = num_node_combined_times_pairs_push_num_nodes_in_at_size(locality_cpx_num_node_combined_times_pairs, target_size_tpx_cpx)
        
        # ALL modes
        locality_all_spx_avg       = reduce_combined_times(locality_all_spx_combined, np.mean)
        locality_all_spx_error     = reduce_combined_times(locality_all_spx_combined, get_min_max_error_diffs)
        locality_all_spx_error_np  = min_max_error_diff_reduced_times_to_np(locality_all_spx_error)

        locality_all_tpx_avg       = reduce_combined_times(locality_all_tpx_combined, np.mean)
        locality_all_tpx_error     = reduce_combined_times(locality_all_tpx_combined, get_min_max_error_diffs)
        locality_all_tpx_error_np  = min_max_error_diff_reduced_times_to_np(locality_all_tpx_error)

        locality_all_cpx_avg       = reduce_combined_times(locality_all_cpx_combined, np.mean)
        locality_all_cpx_error     = reduce_combined_times(locality_all_cpx_combined, get_min_max_error_diffs)
        locality_all_cpx_error_np  = min_max_error_diff_reduced_times_to_np(locality_all_cpx_error)

        # SPX only
        locality_spx_spx_avg       = reduce_combined_times(locality_spx_spx_combined, np.mean)
        locality_spx_spx_error     = reduce_combined_times(locality_spx_spx_combined, get_min_max_error_diffs)
        locality_spx_spx_error_np  = min_max_error_diff_reduced_times_to_np(locality_spx_spx_error)

        # TPX only
        locality_tpx_tpx_avg       = reduce_combined_times(locality_tpx_tpx_combined, np.mean)
        locality_tpx_tpx_error     = reduce_combined_times(locality_tpx_tpx_combined, get_min_max_error_diffs)
        locality_tpx_tpx_error_np  = min_max_error_diff_reduced_times_to_np(locality_tpx_tpx_error)

        # CPX only
        locality_cpx_cpx_avg       = reduce_combined_times(locality_cpx_cpx_combined, np.mean)
        locality_cpx_cpx_error     = reduce_combined_times(locality_cpx_cpx_combined, get_min_max_error_diffs)
        locality_cpx_cpx_error_np  = min_max_error_diff_reduced_times_to_np(locality_cpx_cpx_error)

        # TPX + CPX
        locality_tpx_cpx_tpx_avg   = reduce_combined_times(locality_tpx_cpx_tpx_combined, np.mean)
        locality_tpx_cpx_tpx_error = reduce_combined_times(locality_tpx_cpx_tpx_combined, get_min_max_error_diffs)
        locality_tpx_cpx_tpx_error_np = min_max_error_diff_reduced_times_to_np(locality_tpx_cpx_tpx_error)

        locality_tpx_cpx_cpx_avg   = reduce_combined_times(locality_tpx_cpx_cpx_combined, np.mean)
        locality_tpx_cpx_cpx_error = reduce_combined_times(locality_tpx_cpx_cpx_combined, get_min_max_error_diffs)
        locality_tpx_cpx_cpx_error_np = min_max_error_diff_reduced_times_to_np(locality_tpx_cpx_cpx_error)
               
        pdf = PdfPages(fn_out)
                
        # For ALL modes
        plt.figure()

        # 1 PPG
        plt.errorbar(
            [x[0] for x in locality_all_spx_avg[1]["STD"]],
            [y[1] for y in locality_all_spx_avg[1]["STD"]],
            yerr=(locality_all_spx_error_np[1]["STD"])[1:, :],
            label="1 PPG, SPX"
        )
        plt.errorbar(
            [x[0] for x in locality_all_tpx_avg[1]["STD"]],
            [y[1] for y in locality_all_tpx_avg[1]["STD"]],
            yerr=(locality_all_tpx_error_np[1]["STD"])[1:, :],
            label="1 PPG, TPX"
        )
        plt.errorbar(
            [x[0] for x in locality_all_cpx_avg[1]["STD"]],
            [y[1] for y in locality_all_cpx_avg[1]["STD"]],
            yerr=(locality_all_cpx_error_np[1]["STD"])[1:, :],
            label="1 PPG, CPX"
        )

        # 2 PPG
        plt.errorbar(
            [x[0] for x in locality_all_spx_avg[2]["STD MPS COPY"]],
            [y[1] for y in locality_all_spx_avg[2]["STD MPS COPY"]],
            yerr=(locality_all_spx_error_np[2]["STD MPS COPY"])[1:, :],
            label="2 PPG, SPX"
        )
        plt.errorbar(
            [x[0] for x in locality_all_tpx_avg[2]["STD MPS COPY"]],
            [y[1] for y in locality_all_tpx_avg[2]["STD MPS COPY"]],
            yerr=(locality_all_tpx_error_np[2]["STD MPS COPY"])[1:, :],
            label="2 PPG, TPX"
        )
        plt.errorbar(
            [x[0] for x in locality_all_cpx_avg[2]["STD MPS COPY"]],
            [y[1] for y in locality_all_cpx_avg[2]["STD MPS COPY"]],
            yerr=(locality_all_cpx_error_np[2]["STD MPS COPY"])[1:, :],
            label="2 PPG, CPX"
        )

        # Styling
        plt.title(
            f"Allreduce\nAll nodes at $2^{{{target_power_all}}}$ floats"
        )
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(math.ceil(math.log10(min(dict_li_locality_times))),
                                    math.floor(math.log10(max(dict_li_locality_times)))+1)}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")

        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # Baseline: 1 PPG, SPX (normalized)
        plt.plot(
            [x[0] for x in locality_all_spx_avg[1]["STD"]],
            np.array([y[1] for y in locality_all_spx_avg[1]["STD"]]) /
            np.array([y[1] for y in locality_all_spx_avg[1]["STD"]]),
            label="1 PPG, SPX"
        )

        # 1 PPG, TPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_tpx_error_np[1]["STD"],
            locality_all_tpx_avg[1]["STD"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="1 PPG, TPX")

        # 1 PPG, CPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_cpx_error_np[1]["STD"],
            locality_all_cpx_avg[1]["STD"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="1 PPG, CPX")

        # 2 PPG, SPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_spx_error_np[2]["STD MPS COPY"],
            locality_all_spx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="2 PPG, SPX")

        # 2 PPG, TPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_tpx_error_np[2]["STD MPS COPY"],
            locality_all_tpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="2 PPG, TPX")

        # 2 PPG, CPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_cpx_error_np[2]["STD MPS COPY"],
            locality_all_cpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="2 PPG, CPX")

        # Styling
        plt.title(f"Allreduce\nAll nodes at $2^{{{target_power_all}}}$ floats, MI300A modes")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.ylabel("Speedup")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(math.ceil(math.log10(min(dict_li_locality_times))),
                                    math.floor(math.log10(max(dict_li_locality_times)))+1)}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)

        plt.legend()
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # 1 PPG
        plt.errorbar([x[0] for x in locality_all_spx_avg[1]["MIKELANE"]],
                    [y[1] for y in locality_all_spx_avg[1]["MIKELANE"]],
                    yerr=(locality_all_spx_error_np[1]["MIKELANE"])[1:, :],
                    label="Lane, 1 PPG, SPX")

        plt.errorbar([x[0] for x in locality_all_tpx_avg[1]["MIKELANE"]],
                    [y[1] for y in locality_all_tpx_avg[1]["MIKELANE"]],
                    yerr=(locality_all_tpx_error_np[1]["MIKELANE"])[1:, :],
                    label="Lane, 1 PPG, TPX")

        plt.errorbar([x[0] for x in locality_all_cpx_avg[1]["MIKELANE"]],
                    [y[1] for y in locality_all_cpx_avg[1]["MIKELANE"]],
                    yerr=(locality_all_cpx_error_np[1]["MIKELANE"])[1:, :],
                    label="Lane, 1 PPG, CPX")

        # 2 PPG
        plt.errorbar([x[0] for x in locality_all_spx_avg[2]["MIKELANE MPS COPY"]],
                    [y[1] for y in locality_all_spx_avg[2]["MIKELANE MPS COPY"]],
                    yerr=(locality_all_spx_error_np[2]["MIKELANE MPS COPY"])[1:, :],
                    label="Lane, 2 PPG, SPX")

        plt.errorbar([x[0] for x in locality_all_tpx_avg[2]["MIKELANE MPS COPY"]],
                    [y[1] for y in locality_all_tpx_avg[2]["MIKELANE MPS COPY"]],
                    yerr=(locality_all_tpx_error_np[2]["MIKELANE MPS COPY"])[1:, :],
                    label="Lane, 2 PPG, TPX")

        plt.errorbar([x[0] for x in locality_all_cpx_avg[2]["MIKELANE MPS COPY"]],
                    [y[1] for y in locality_all_cpx_avg[2]["MIKELANE MPS COPY"]],
                    yerr=(locality_all_cpx_error_np[2]["MIKELANE MPS COPY"])[1:, :],
                    label="Lane, 2 PPG, CPX")

        # Styling
        plt.title(f"Lane Allreduce\nAll nodes at $2^{{{target_power_all}}}$ floats")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(math.ceil(math.log10(min(dict_li_locality_times))),
                                    math.floor(math.log10(max(dict_li_locality_times)))+1)}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")

        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())

        plt.figure()

        # Baseline for speedup (1 PPG, MIKE Lane)
        baseline_x = [x[0] for x in locality_all_spx_avg[1]["MIKELANE"]]
        baseline_y = np.array([y[1] for y in locality_all_spx_avg[1]["MIKELANE"]])

        # Speedup = time_1 / time_current
        plt.plot(baseline_x, baseline_y / baseline_y, label="Lane, 1 PPG, SPX")  # baseline

        # 1 PPG, TPX
        local_speedup_error_diffs, local_x, local_avg = \
            min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_all_spx_error_np[1]["MIKELANE"],
                locality_all_spx_avg[1]["MIKELANE"],
                locality_all_tpx_error_np[1]["MIKELANE"],
                locality_all_tpx_avg[1]["MIKELANE"]
            )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 1 PPG, TPX")

        # 1 PPG, CPX
        local_speedup_error_diffs, local_x, local_avg = \
            min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_all_spx_error_np[1]["MIKELANE"],
                locality_all_spx_avg[1]["MIKELANE"],
                locality_all_cpx_error_np[1]["MIKELANE"],
                locality_all_cpx_avg[1]["MIKELANE"]
            )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 1 PPG, CPX")

        # 2 PPG, SPX
        local_speedup_error_diffs, local_x, local_avg = \
            min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_all_spx_error_np[1]["MIKELANE"],
                locality_all_spx_avg[1]["MIKELANE"],
                locality_all_spx_error_np[2]["MIKELANE MPS COPY"],
                locality_all_spx_avg[2]["MIKELANE MPS COPY"]
            )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 2 PPG, SPX")

        # 2 PPG, TPX
        local_speedup_error_diffs, local_x, local_avg = \
            min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_all_spx_error_np[1]["MIKELANE"],
                locality_all_spx_avg[1]["MIKELANE"],
                locality_all_tpx_error_np[2]["MIKELANE MPS COPY"],
                locality_all_tpx_avg[2]["MIKELANE MPS COPY"]
            )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 2 PPG, TPX")

        # 2 PPG, CPX
        local_speedup_error_diffs, local_x, local_avg = \
            min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                locality_all_spx_error_np[1]["MIKELANE"],
                locality_all_spx_avg[1]["MIKELANE"],
                locality_all_cpx_error_np[2]["MIKELANE MPS COPY"],
                locality_all_cpx_avg[2]["MIKELANE MPS COPY"]
            )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 2 PPG, CPX")

        # Styling
        plt.title(f"Lane Allreduce\nAll nodes at $2^{{{target_power_all}}}$ floats")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(math.ceil(math.log10(min(dict_li_locality_times))),
                                    math.floor(math.log10(max(dict_li_locality_times)))+1)}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")
        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())

        plt.figure()

        # 1 PPG, SPX
        plt.errorbar(
            [x[0] for x in locality_spx_spx_avg[1]["STD"]],
            [y[1] for y in locality_spx_spx_avg[1]["STD"]],
            yerr=(locality_spx_spx_error_np[1]["STD"])[1:, :],
            label="1 PPG, SPX"
        )

        # 2 PPG, SPX
        plt.errorbar(
            [x[0] for x in locality_spx_spx_avg[2]["STD MPS COPY"]],
            [y[1] for y in locality_spx_spx_avg[2]["STD MPS COPY"]],
            yerr=(locality_spx_spx_error_np[2]["STD MPS COPY"])[1:, :],
            label="2 PPG, SPX"
        )

        # Lane 1 PPG, SPX
        plt.errorbar(
            [x[0] for x in locality_spx_spx_avg[1]["MIKELANE"]],
            [y[1] for y in locality_spx_spx_avg[1]["MIKELANE"]],
            yerr=(locality_spx_spx_error_np[1]["MIKELANE"])[1:, :],
            label="Lane, 1 PPG, SPX"
        )

        # Lane 2 PPG, SPX
        plt.errorbar(
            [x[0] for x in locality_spx_spx_avg[2]["MIKELANE MPS COPY"]],
            [y[1] for y in locality_spx_spx_avg[2]["MIKELANE MPS COPY"]],
            yerr=(locality_spx_spx_error_np[2]["MIKELANE MPS COPY"])[1:, :],
            label="Lane, 2 PPG, SPX"
        )

        # Styling
        plt.title(f"Lane+Std Allreduce\nAll nodes at $2^{{{target_power_spx}}}$ floats, MI300A SPX")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")

        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # Baseline: 1-PPG STD SPX → normalized
        plt.plot(
            [x[0] for x in locality_spx_spx_avg[1]["STD"]],
            np.array([y[1] for y in locality_spx_spx_avg[1]["STD"]]) /
            np.array([y[1] for y in locality_spx_spx_avg[1]["STD"]]),
            label="1 PPG, SPX"
        )

        # 2-PPG STD SPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_spx_spx_error_np[1]["STD"],
            locality_spx_spx_avg[1]["STD"],
            locality_spx_spx_error_np[2]["STD MPS COPY"],
            locality_spx_spx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="2 PPG, SPX")

        # Lane, 1-PPG SPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_spx_spx_error_np[1]["STD"],
            locality_spx_spx_avg[1]["STD"],
            locality_spx_spx_error_np[1]["MIKELANE"],
            locality_spx_spx_avg[1]["MIKELANE"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 1 PPG, SPX")

        # Lane, 2-PPG SPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_spx_spx_error_np[1]["STD"],
            locality_spx_spx_avg[1]["STD"],
            locality_spx_spx_error_np[2]["MIKELANE MPS COPY"],
            locality_spx_spx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 2 PPG, SPX")

        # Styling
        plt.title(f"Lane+Std Allreduce\nAll nodes at $2^{{{target_power_spx}}}$ floats, MI300A SPX")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")

        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
  
        plt.figure()

        # 1-PPG STD TPX
        plt.errorbar(
            [x[0] for x in locality_tpx_tpx_avg[1]["STD"]],
            [y[1] for y in locality_tpx_tpx_avg[1]["STD"]],
            yerr=(locality_tpx_tpx_error_np[1]["STD"])[1:, :],
            label="1 PPG, TPX"
        )

        # 2-PPG STD TPX
        plt.errorbar(
            [x[0] for x in locality_tpx_tpx_avg[2]["STD MPS COPY"]],
            [y[1] for y in locality_tpx_tpx_avg[2]["STD MPS COPY"]],
            yerr=(locality_tpx_tpx_error_np[2]["STD MPS COPY"])[1:, :],
            label="2 PPG, TPX"
        )

        # Lane, 1-PPG TPX
        plt.errorbar(
            [x[0] for x in locality_tpx_tpx_avg[1]["MIKELANE"]],
            [y[1] for y in locality_tpx_tpx_avg[1]["MIKELANE"]],
            yerr=(locality_tpx_tpx_error_np[1]["MIKELANE"])[1:, :],
            label="Lane, 1 PPG, TPX"
        )

        # Lane, 2-PPG TPX
        plt.errorbar(
            [x[0] for x in locality_tpx_tpx_avg[2]["MIKELANE MPS COPY"]],
            [y[1] for y in locality_tpx_tpx_avg[2]["MIKELANE MPS COPY"]],
            yerr=(locality_tpx_tpx_error_np[2]["MIKELANE MPS COPY"])[1:, :],
            label="Lane, 2 PPG, TPX"
        )

        # Styling
        plt.title(f"Lane+Std Allreduce\nAll nodes at $2^{{{target_power_tpx}}}$ floats, MI300A TPX")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")

        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # Baseline: 1-PPG STD TPX → normalized to 1
        plt.plot(
            [x[0] for x in locality_tpx_tpx_avg[1]["STD"]],
            np.array([y[1] for y in locality_tpx_tpx_avg[1]["STD"]]) /
            np.array([y[1] for y in locality_tpx_tpx_avg[1]["STD"]]),
            label="1 PPG, TPX"
        )

        # 2-PPG STD TPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_tpx_tpx_error_np[1]["STD"],
            locality_tpx_tpx_avg[1]["STD"],
            locality_tpx_tpx_error_np[2]["STD MPS COPY"],
            locality_tpx_tpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="2 PPG, TPX")

        # Lane, 1-PPG TPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_tpx_tpx_error_np[1]["STD"],
            locality_tpx_tpx_avg[1]["STD"],
            locality_tpx_tpx_error_np[1]["MIKELANE"],
            locality_tpx_tpx_avg[1]["MIKELANE"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 1 PPG, TPX")

        # Lane, 2-PPG TPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_tpx_tpx_error_np[1]["STD"],
            locality_tpx_tpx_avg[1]["STD"],
            locality_tpx_tpx_error_np[2]["MIKELANE MPS COPY"],
            locality_tpx_tpx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 2 PPG, TPX")

        plt.title(f"Lane+Std Allreduce\nAll nodes at $2^{{{target_power_tpx}}}$ floats, MI300A TPX")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(plt.gcf())


        # ========================
        # CPX Time Plot
        # ========================
        plt.figure()

        # 1-PPG CPX
        plt.errorbar(
            [x[0] for x in locality_cpx_cpx_avg[1]["STD"]],
            [y[1] for y in locality_cpx_cpx_avg[1]["STD"]],
            yerr=(locality_cpx_cpx_error_np[1]["STD"])[1:, :],
            label="1 PPG, CPX"
        )

        # 2-PPG CPX
        plt.errorbar(
            [x[0] for x in locality_cpx_cpx_avg[2]["STD MPS COPY"]],
            [y[1] for y in locality_cpx_cpx_avg[2]["STD MPS COPY"]],
            yerr=(locality_cpx_cpx_error_np[2]["STD MPS COPY"])[1:, :],
            label="2 PPG, CPX"
        )

        # Lane, 1-PPG CPX
        plt.errorbar(
            [x[0] for x in locality_cpx_cpx_avg[1]["MIKELANE"]],
            [y[1] for y in locality_cpx_cpx_avg[1]["MIKELANE"]],
            yerr=(locality_cpx_cpx_error_np[1]["MIKELANE"])[1:, :],
            label="Lane, 1 PPG, CPX"
        )

        # Lane, 2-PPG CPX
        plt.errorbar(
            [x[0] for x in locality_cpx_cpx_avg[2]["MIKELANE MPS COPY"]],
            [y[1] for y in locality_cpx_cpx_avg[2]["MIKELANE MPS COPY"]],
            yerr=(locality_cpx_cpx_error_np[2]["MIKELANE MPS COPY"])[1:, :],
            label="Lane, 2 PPG, CPX"
        )

        plt.title(f"Lane+Std Allreduce\nAll nodes at $2^{{{target_power_cpx}}}$ floats, MI300A CPX")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")

        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())


        # ========================
        # CPX Speedup Plot
        # ========================
        plt.figure()

        # Baseline: 1-PPG STD CPX → normalized to 1
        plt.plot(
            [x[0] for x in locality_cpx_cpx_avg[1]["STD"]],
            np.array([y[1] for y in locality_cpx_cpx_avg[1]["STD"]]) /
            np.array([y[1] for y in locality_cpx_cpx_avg[1]["STD"]]),
            label="1 PPG, CPX"
        )

        # 2-PPG STD CPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_cpx_cpx_error_np[1]["STD"],
            locality_cpx_cpx_avg[1]["STD"],
            locality_cpx_cpx_error_np[2]["STD MPS COPY"],
            locality_cpx_cpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="2 PPG, CPX")

        # Lane, 1-PPG CPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_cpx_cpx_error_np[1]["STD"],
            locality_cpx_cpx_avg[1]["STD"],
            locality_cpx_cpx_error_np[1]["MIKELANE"],
            locality_cpx_cpx_avg[1]["MIKELANE"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 1 PPG, CPX")

        # Lane, 2-PPG CPX
        local_speedup_error_diffs, local_x, local_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_cpx_cpx_error_np[1]["STD"],
            locality_cpx_cpx_avg[1]["STD"],
            locality_cpx_cpx_error_np[2]["MIKELANE MPS COPY"],
            locality_cpx_cpx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(local_x, local_avg, yerr=local_speedup_error_diffs[1:, :], label="Lane, 2 PPG, CPX")

        plt.title(f"Lane+Std Allreduce\nAll nodes at $2^{{{target_power_cpx}}}$ floats, MI300A CPX")
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        PX_COLORS = {
            "SPX": "green",
            "TPX": "blue",
            "CPX": "red",
        }
        
        PX_LANE_COLORS = {
            "SPX": "green",
            "TPX": "blue",
            "CPX": "red",
        }
        
        plt.figure()
        # ---- SPX ----
        plt.errorbar([x[0] for x in locality_all_spx_avg[1]["STD"]],
                    [y[1] for y in locality_all_spx_avg[1]["STD"]],
                    yerr=locality_all_spx_error_np[1]["STD"][1:, :],
                    color=PX_COLORS["SPX"], linestyle="-", label="1 PPG, SPX")

        plt.errorbar([x[0] for x in locality_all_spx_avg[2]["STD MPS COPY"]],
                    [y[1] for y in locality_all_spx_avg[2]["STD MPS COPY"]],
                    yerr=locality_all_spx_error_np[2]["STD MPS COPY"][1:, :],
                    color=PX_COLORS["SPX"], linestyle="--", label="2 PPG, SPX")

        # ---- TPX ----
        plt.errorbar([x[0] for x in locality_all_tpx_avg[1]["STD"]],
                    [y[1] for y in locality_all_tpx_avg[1]["STD"]],
                    yerr=locality_all_tpx_error_np[1]["STD"][1:, :],
                    color=PX_COLORS["TPX"], linestyle="-", label="1 PPG, TPX")

        plt.errorbar([x[0] for x in locality_all_tpx_avg[2]["STD MPS COPY"]],
                    [y[1] for y in locality_all_tpx_avg[2]["STD MPS COPY"]],
                    yerr=locality_all_tpx_error_np[2]["STD MPS COPY"][1:, :],
                    color=PX_COLORS["TPX"], linestyle="--", label="2 PPG, TPX")

        # ---- CPX ----
        plt.errorbar([x[0] for x in locality_all_cpx_avg[1]["STD"]],
                    [y[1] for y in locality_all_cpx_avg[1]["STD"]],
                    yerr=locality_all_cpx_error_np[1]["STD"][1:, :],
                    color=PX_COLORS["CPX"], linestyle="-", label="1 PPG, CPX")

        plt.errorbar([x[0] for x in locality_all_cpx_avg[2]["STD MPS COPY"]],
                    [y[1] for y in locality_all_cpx_avg[2]["STD MPS COPY"]],
                    yerr=locality_all_cpx_error_np[2]["STD MPS COPY"][1:, :],
                    color=PX_COLORS["CPX"], linestyle="--", label="2 PPG, CPX")

        plt.xlabel("Nodes")
        plt.ylabel("Time (Seconds)")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.yscale("log")
        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # ---- SPX speedup vs SPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_spx_error_np[2]["STD MPS COPY"],
            locality_all_spx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_COLORS["SPX"], linestyle="--", label="2 PPG, SPX")


        # ---- TPX speedup vs TPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_tpx_error_np[1]["STD"],
            locality_all_tpx_avg[1]["STD"],
            locality_all_tpx_error_np[2]["STD MPS COPY"],
            locality_all_tpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_COLORS["TPX"], linestyle="--", label="2 PPG, TPX")


        # ---- CPX speedup vs CPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_cpx_error_np[1]["STD"],
            locality_all_cpx_avg[1]["STD"],
            locality_all_cpx_error_np[2]["STD MPS COPY"],
            locality_all_cpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_COLORS["CPX"], linestyle="--", label="2 PPG, CPX")


        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(plt.gcf())
    
        plt.figure()

        # ---- SPX ----
        plt.errorbar([x[0] for x in locality_all_spx_avg[1]["MIKELANE"]],
                    [y[1] for y in locality_all_spx_avg[1]["MIKELANE"]],
                    yerr=locality_all_spx_error_np[1]["MIKELANE"][1:, :],
                    color=PX_LANE_COLORS["SPX"], linestyle="-", label="Lane, 1 PPG, SPX")

        plt.errorbar([x[0] for x in locality_all_spx_avg[2]["MIKELANE MPS COPY"]],
                    [y[1] for y in locality_all_spx_avg[2]["MIKELANE MPS COPY"]],
                    yerr=locality_all_spx_error_np[2]["MIKELANE MPS COPY"][1:, :],
                    color=PX_LANE_COLORS["SPX"], linestyle="--", label="Lane, 2 PPG, SPX")


        # ---- TPX ----
        plt.errorbar([x[0] for x in locality_all_tpx_avg[1]["MIKELANE"]],
                    [y[1] for y in locality_all_tpx_avg[1]["MIKELANE"]],
                    yerr=locality_all_tpx_error_np[1]["MIKELANE"][1:, :],
                    color=PX_LANE_COLORS["TPX"], linestyle="-", label="Lane, 1 PPG, TPX")

        plt.errorbar([x[0] for x in locality_all_tpx_avg[2]["MIKELANE MPS COPY"]],
                    [y[1] for y in locality_all_tpx_avg[2]["MIKELANE MPS COPY"]],
                    yerr=locality_all_tpx_error_np[2]["MIKELANE MPS COPY"][1:, :],
                    color=PX_LANE_COLORS["TPX"], linestyle="--", label="Lane, 2 PPG, TPX")


        # ---- CPX ----
        plt.errorbar([x[0] for x in locality_all_cpx_avg[1]["MIKELANE"]],
                    [y[1] for y in locality_all_cpx_avg[1]["MIKELANE"]],
                    yerr=locality_all_cpx_error_np[1]["MIKELANE"][1:, :],
                    color=PX_LANE_COLORS["CPX"], linestyle="-", label="Lane, 1 PPG, CPX")

        plt.errorbar([x[0] for x in locality_all_cpx_avg[2]["MIKELANE MPS COPY"]],
                    [y[1] for y in locality_all_cpx_avg[2]["MIKELANE MPS COPY"]],
                    yerr=locality_all_cpx_error_np[2]["MIKELANE MPS COPY"][1:, :],
                    color=PX_LANE_COLORS["CPX"], linestyle="--", label="Lane, 2 PPG, CPX")


        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Time (Seconds)")
        plt.yscale("log")
        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # ---- SPX speedup vs SPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["MIKELANE"],
            locality_all_spx_avg[1]["MIKELANE"],
            locality_all_spx_error_np[2]["MIKELANE MPS COPY"],
            locality_all_spx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_LANE_COLORS["SPX"], linestyle="--", label="Lane, 2 PPG, SPX")

        # ---- TPX speedup vs TPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_tpx_error_np[1]["MIKELANE"],
            locality_all_tpx_avg[1]["MIKELANE"],
            locality_all_tpx_error_np[2]["MIKELANE MPS COPY"],
            locality_all_tpx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_LANE_COLORS["TPX"], linestyle="--", label="Lane, 2 PPG, TPX")

        # ---- CPX speedup vs CPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_cpx_error_np[1]["MIKELANE"],
            locality_all_cpx_avg[1]["MIKELANE"],
            locality_all_cpx_error_np[2]["MIKELANE MPS COPY"],
            locality_all_cpx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_LANE_COLORS["CPX"], linestyle="--", label="Lane, 2 PPG, CPX")

        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")
        plt.ylim((0.5, 3.))
        plt.legend()
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        plt.figure()

        # ---- SPX speedup vs SPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_spx_error_np[2]["STD MPS COPY"],
            locality_all_spx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_COLORS["SPX"], linestyle="-", label="2 PPG, SPX")

        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_spx_error_np[1]["STD"],
            locality_all_spx_avg[1]["STD"],
            locality_all_spx_error_np[2]["MIKELANE MPS COPY"],
            locality_all_spx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_LANE_COLORS["SPX"], linestyle="--", label="Lane, 2 PPG, SPX")

        # ---- TPX speedup vs TPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_tpx_error_np[1]["STD"],
            locality_all_tpx_avg[1]["STD"],
            locality_all_tpx_error_np[2]["STD MPS COPY"],
            locality_all_tpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_COLORS["TPX"], linestyle="-", label="2 PPG, TPX")

        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_tpx_error_np[1]["STD"],
            locality_all_tpx_avg[1]["STD"],
            locality_all_tpx_error_np[2]["MIKELANE MPS COPY"],
            locality_all_tpx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_LANE_COLORS["TPX"], linestyle="--", label="Lane, 2 PPG, TPX")

        # ---- CPX speedup vs CPX 1-PPG ----
        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_cpx_error_np[1]["STD"],
            locality_all_cpx_avg[1]["STD"],
            locality_all_cpx_error_np[2]["STD MPS COPY"],
            locality_all_cpx_avg[2]["STD MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_COLORS["CPX"], linestyle="-", label="2 PPG, CPX")

        e, x, avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
            locality_all_cpx_error_np[1]["STD"],
            locality_all_cpx_avg[1]["STD"],
            locality_all_cpx_error_np[2]["MIKELANE MPS COPY"],
            locality_all_cpx_avg[2]["MIKELANE MPS COPY"]
        )
        plt.errorbar(x, avg, yerr=e[1:, :],
                    color=PX_LANE_COLORS["CPX"], linestyle="--", label="Lane, 2 PPG, CPX")

        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(
            ticks=sorted(
                set(dict_li_locality_times.keys()) |
                {10**i for i in range(
                    math.ceil(math.log10(min(dict_li_locality_times))),
                    math.floor(math.log10(max(dict_li_locality_times)))+1
                )}
            )
        )
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")
        plt.tick_params(which="minor", left=True)
        plt.ylabel("Speedup")
        # plt.legend()
        pfp.add_anchored_legend(ncol=3, fontsize=16, anchor=(0, 1.1, 1, 0.102))       
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        
        pdf.close()