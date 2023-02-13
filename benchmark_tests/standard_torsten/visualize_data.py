import os
import typing
import matplotlib.pyplot as plt

machine_name = "Hopper"
matrix_directories = next(os.walk('.'))[1]

standard_dict_average = dict()
standard_dict_max = dict()
standard_dict_min = dict()
standard_dict_num_msg = dict()
standard_dict_msg_size = dict()

torsten_dict_average = dict()
torsten_dict_max = dict()
torsten_dict_min = dict()
torsten_dict_num_msg = dict()
torsten_dict_msg_size = dict()

rma_dict_average = dict()
rma_dict_max = dict()
rma_dict_min = dict()
rma_dict_num_msg = dict()
rma_dict_msg_size = dict()

# Takes dictionaries w/ data, prints output to a file and returns sorted data lists (average_list, max_list, min_list, num_msg_list, msg_size_list)
def print_data(fp : __file__, average_dict : dict, max_dict : dict, min_dict : dict, num_msg_dict : dict, msg_size_dict : dict) -> tuple[list, list, list, list, list]:
  (average_keys, max_keys, min_keys, num_msg_keys, msg_size_keys) = get_and_sort_keys(average_dict, max_dict, min_dict, num_msg_dict, msg_size_dict)

  average_data = []
  max_data = []
  min_data = []
  num_msg_data = []
  msg_size_data = []

  fp.write("AVERAGE DATA:\n")
  for k in average_keys:
    fp.write(f"{k},{average_dict.get(k)[1]/average_dict.get(k)[0]:.6f}\n")
    average_data.append(1000*(average_dict.get(k)[1]/average_dict.get(k)[0]))

  fp.write("MAX DATA:\n")
  for k in max_keys:
    fp.write(f"{k},{max_dict.get(k):.6f}\n")
    max_data.append(1000*(max_dict.get(k)))

  fp.write("MIN DATA:\n")
  for k in min_keys:
    fp.write(f"{k},{min_dict.get(k):.6f}\n")
    min_data.append(1000*(min_dict.get(k)))

  fp.write("NUM MESSAGE DATA:\n")
  for k in num_msg_keys:
    fp.write(f"{k},{num_msg_dict.get(k)}\n")
    num_msg_data.append(num_msg_dict.get(k))

  fp.write("MESSAGE SIZE DATA:\n")
  for k in msg_size_keys:
    fp.write(f"{k},{msg_size_dict.get(k)}\n")
    msg_size_data.append(msg_size_dict.get(k))
  return (average_data, max_data, min_data, num_msg_data, msg_size_data)

# Gets and sorts keys for all the various dictionaries, returns 5-tuple in order of arguments
def get_and_sort_keys(average_dict : dict, max_dict : dict, min_dict : dict, num_msg_dict : dict, msg_size_dict : dict) -> tuple(list, list, list, list ,list):
  average_dict_keys = list(average_dict.keys())
  average_dict_keys.sort()
  max_dict_keys = list(max_dict.keys())
  max_dict_keys.sort()
  min_dict_keys = list(min_dict.keys())
  min_dict_keys.sort()
  num_msg_dict_keys = list(num_msg_dict.keys())
  num_msg_dict_keys.sort()
  msg_size_dict_keys = list(msg_size_dict.keys())
  msg_size_dict_keys.sort()
  return (average_dict_keys, max_dict_keys, min_dict_keys, num_msg_dict_keys, msg_size_dict_keys)

# Parses the list with data and updates the dictonaries with said data
def parse_output_strings(average_dict : dict, max_dict : dict, min_dict : dict, num_message_dict : dict, msg_size_dict : dict,  out_strings : list):
  i = 0
  while i < len(out_strings):
    # Get number of procs / number of tests
    line_parts = out_strings[i].split(',')
    num_procs = int(line_parts[1].strip().split(' ')[0])
    num_tests = int(line_parts[2].strip().split(' ')[0])
    # Get num messages / message size
    i = i + 1
    line_parts = out_strings[i].split(',')

    num_messages = int(line_parts[0].strip().split(' ')[1])
    msg_size = int(line_parts[1].strip().split(' ')[1])
    num_message_dict.update({num_procs : num_messages})
    msg_size_dict.update({num_procs : msg_size})
    for _ in range(num_tests):
      i = i + 1
      line_parts = out_strings[i].split(',')
      time_taken = float(line_parts[0])
      # Update average dictionary 
      x = average_dict.get(num_procs)
      if x == None: 
        average_dict.update({num_procs:(1, time_taken)})
      else: 
        average_dict.update({num_procs:(x[0]+1,x[1]+time_taken)})
      # Update max dictionary 
      x = max_dict.get(num_procs)
      if x == None: 
        max_dict.update({num_procs : time_taken})
      else: 
        if (x > time_taken):
          max_dict.update({num_procs : time_taken })
        else: 
          max_dict.update({num_procs : x})
      # Update min dictionary
      x = min_dict.get(num_procs)
      if x == None: 
        min_dict.update({num_procs : time_taken})
      else: 
        if (x > time_taken): 
          max_dict.update({num_procs : x})
        else:
          max_dict.update({num_procs : time_taken})
    i = i + 1

def make_time_plot(key_list : list, data_list : list, title : str, out_file : str):
  plt.plot(key_list, data_list)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(title)
  plt.savefig(out_file)
  plt.clf()

def visualize_data(fp_1 : __file__, fp_2 : __file__, matrix : str, machine_name : str):
  (average_keys_standard, max_keys_standard, min_keys_standard, num_msg_keys_standard, msg_size_keys_standard) = get_and_sort_keys(standard_dict_average, standard_dict_max, standard_dict_min, standard_dict_num_msg, standard_dict_msg_size)
  (average_data_standard, max_data_standard, min_data_standard, num_msg_data_standard, msg_size_data_standard) = print_data(fp_1, standard_dict_average, standard_dict_max, standard_dict_min, standard_dict_num_msg, standard_dict_msg_size)

  (average_keys_torsten, max_keys_torsten, min_keys_torsten, num_msg_keys_torsten, msg_size_keys_torsten) = get_and_sort_keys(torsten_dict_average, torsten_dict_max, torsten_dict_min, torsten_dict_num_msg, torsten_dict_msg_size)
  (average_data_torsten, max_data_torsten, min_data_torsten, num_msg_data_torsten, msg_size_data_torsten) =  print_data(fp_2, torsten_dict_average, torsten_dict_max, torsten_dict_min, torsten_dict_num_msg, torsten_dict_msg_size)

  make_time_plot(average_keys_standard, average_data_standard, f"{matrix} average run time on {machine_name} (standard method)", f"./{matrix}/{matrix}_{machine_name}_standard_average_plot.png")
  make_time_plot(max_keys_standard, max_data_standard, f"{matrix} max run time on {machine_name} (standard method)", f"./{matrix}/{matrix}_{machine_name}_standard_max_plot.png")
  make_time_plot(min_keys_standard, min_data_standard, f"{matrix} min run time on {machine_name} (standard method)", f"./{matrix}/{matrix}_{machine_name}_standard_min_plot.png")

  plt.plot(num_msg_keys_standard, num_msg_data_standard)
  plt.xlabel("Number of Processes")
  plt.ylabel("Max # of Messages Sent")
  plt.title(f"{matrix} num messages on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_num_msg_plot.png")
  plt.clf()

  plt.plot(msg_size_keys_standard, msg_size_data_standard)
  plt.xlabel("Number of Processes")
  plt.ylabel("Max Message Size (Bytes)")
  plt.title(f"{matrix} message size on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_msg_size_plot.png")
  plt.clf()

  make_time_plot(average_keys_torsten, average_data_torsten, f"{matrix} average run time on {machine_name} (torsten's method)", f"./{matrix}/{matrix}_{machine_name}_torsten_average_plot.png")
  make_time_plot(max_keys_torsten, max_data_torsten, f"{matrix} max run time on {machine_name} (torsten's method)", f"./{matrix}/{matrix}_{machine_name}_torsten_max_plot.png")
  make_time_plot(min_keys_torsten, min_data_torsten, f"{matrix} min run time on {machine_name} (torsten's method)", f"./{matrix}/{matrix}_{machine_name}_torsten_min_plot.png")

  plt.plot(num_msg_keys_torsten, num_msg_data_torsten)
  plt.xlabel("Number of Processes")
  plt.ylabel("Max # of Messages Sent")
  plt.title(f"{matrix} num messages on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_num_msg_plot.png")
  plt.clf()

  plt.plot(msg_size_keys_torsten, msg_size_data_torsten)
  plt.xlabel("Number of Processes")
  plt.ylabel("Max Message Size (Bytes)")
  plt.title(f"{matrix} message size on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_msg_size_plot.png")
  plt.clf()

  plt.plot(average_keys_standard, average_data_standard, '-.', average_keys_torsten, average_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_average_plot.png")
  plt.clf()

  plt.plot(max_keys_standard, max_data_standard, '-.', max_keys_torsten, max_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_max_plot.png")
  plt.clf()

  plt.plot(min_keys_standard, min_data_standard, '-.', min_keys_torsten, min_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} min run time on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_min_plot.png")
  plt.clf()

  plt.plot(num_msg_keys_standard, num_msg_data_standard, '-.', num_msg_keys_torsten, num_msg_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Max # of Messages Sent")
  plt.title(f"{matrix} num messages on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_num_msg_plot.png")
  plt.clf()

  plt.plot(msg_size_keys_standard, msg_size_data_standard, '-.', msg_size_keys_torsten, msg_size_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Max Msg Size (Bytes)")
  plt.title(f"{matrix} message size on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_msg_size_plot.png")
  plt.clf()


for matrix in matrix_directories:
  out_strings_standard = []
  out_strings_torsten = []
  single_node_standard = open(f"./{matrix}/{matrix}_{machine_name}_Standard_one_node", 'r')
  many_node_standard = open(f"./{matrix}/{matrix}_{machine_name}_Standard_many_node",'r')
  single_node_torsten = open(f"./{matrix}/{matrix}_{machine_name}_Torsten_one_node",'r')
  many_node_torsten = open(f"./{matrix}/{matrix}_{machine_name}_Torsten_many_node",'r')

  # Clear lines in file which are not test data (i.e. filler output)
  for line in single_node_standard.read().splitlines():
    if (line.strip().split(',')[0] != "STANDARD") and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != 'MAX_MSG_COUNT'):
      continue
    out_strings_standard.append(line)
  
  for line in many_node_standard.read().splitlines():
    if (line.strip().split(',')[0] != "STANDARD") and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != "MAX_MSG_COUNT"):
      continue
    out_strings_standard.append(line)

  for line in single_node_torsten.read().splitlines():
    if(line.strip().split(',')[0] != "TORSTEN") and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != "MAX_MSG_COUNT"):
      continue
    out_strings_torsten.append(line)

  for line in many_node_torsten.read().splitlines():
    if(line.strip().split(',')[0] != "TORSTEN") and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != "MAX_MSG_COUNT"):
      continue
    out_strings_torsten.append(line)

  fp_1 = open(f"./{matrix}/{matrix}_{machine_name}_standard_data.txt","w")
  fp_2 = open(f"./{matrix}/{matrix}_{machine_name}_torsten_data.txt","w")

  # Add data to standard dictionaries
  parse_output_strings(standard_dict_average, standard_dict_max, standard_dict_min, standard_dict_num_msg, standard_dict_msg_size, out_strings_standard)
  # Add data to torsten dictionaries
  parse_output_strings(torsten_dict_average, torsten_dict_max, torsten_dict_min, torsten_dict_num_msg, torsten_dict_msg_size, out_strings_torsten)
 
  visualize_data(fp_1, fp_2, matrix, machine_name)
  fp_1.close() 
  fp_2.close()