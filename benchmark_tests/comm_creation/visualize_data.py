import os
import typing
import matplotlib.pyplot as plt

RUN_STANDARD = False
RUN_TORSTEN = False
RUN_RMA = True
RUN_RMA_DYANMIC = True

machine_name = "Wheeler"
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

rma_dynamic_dict_average = dict()
rma_dynamic_dict_max = dict()
rma_dynamic_dict_min = dict()
rma_dynamic_dict_num_msg = dict()
rma_dynamic_dict_msg_size = dict()

def create_dirs(matrix : str):
  if not os.path.exists(f"./{matrix}/parsed_data"):
    os.mkdir(f"./{matrix}/parsed_data")
    os.mkdir(f"./{matrix}/parsed_data/tables")
    os.mkdir(f"./{matrix}/parsed_data/plots")
    os.mkdir(f"./{matrix}/parsed_data/one_test_output")
    os.mkdir(f"./{matrix}/plots/average")
    os.mkdir(f"./{matrix}/plots/min")
    os.mkdir(f"./{matrix}/plots/max")


# Takes dictionaries w/ data, prints output to a file and returns sorted data lists (average_list, max_list, min_list, num_msg_list, msg_size_list)
def print_data(fp : __file__, average_dict : dict, max_dict : dict, min_dict : dict, num_msg_dict : dict, msg_size_dict : dict): 
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
def get_and_sort_keys(average_dict : dict, max_dict : dict, min_dict : dict, num_msg_dict : dict, msg_size_dict : dict):
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

def make_msg_num_plot(num_msg_keys : list, num_msg_data : list, matrix : str, machine_name : str, method : str):
  plt.plot(num_msg_keys, num_msg_data)
  plt.xlabel("Number of Processes")
  plt.ylabel("Max # of Messages Sent")
  plt.title(f"{matrix} num message on {machine_name} ({method})")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_{method}_num_msg_plot.png")
  plt.clf()

def make_msg_size_plot(msg_size_keys : list, msg_size_data : list, matrix: str, machine_name : str, method : str):
  plt.plot(msg_size_keys, msg_size_data)
  plt.xlabel("Number of Proceses")
  plt.ylabel("Max Message Size (Bytes)")
  plt.title(f"{matrix} message size on {machine_name} ({method})")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_{method}_msg_size_plot.png")
  plt.clf()

def visualize_data(fp_1 : __file__, fp_2 : __file__, fp_3 : __file__, fp_4 : __file__, matrix : str, machine_name : str):
  (average_keys_standard, max_keys_standard, min_keys_standard, num_msg_keys_standard, msg_size_keys_standard) = get_and_sort_keys(standard_dict_average, standard_dict_max, standard_dict_min, standard_dict_num_msg, standard_dict_msg_size)
  (average_data_standard, max_data_standard, min_data_standard, num_msg_data_standard, msg_size_data_standard) = print_data(fp_1, standard_dict_average, standard_dict_max, standard_dict_min, standard_dict_num_msg, standard_dict_msg_size)

  (average_keys_torsten, max_keys_torsten, min_keys_torsten, num_msg_keys_torsten, msg_size_keys_torsten) = get_and_sort_keys(torsten_dict_average, torsten_dict_max, torsten_dict_min, torsten_dict_num_msg, torsten_dict_msg_size)
  (average_data_torsten, max_data_torsten, min_data_torsten, num_msg_data_torsten, msg_size_data_torsten) =  print_data(fp_2, torsten_dict_average, torsten_dict_max, torsten_dict_min, torsten_dict_num_msg, torsten_dict_msg_size)

  (average_keys_rma, max_keys_rma, min_keys_rma, num_msg_keys_rma, msg_size_keys_rma) = get_and_sort_keys(rma_dict_average, rma_dict_max, rma_dict_min, rma_dict_num_msg, rma_dict_msg_size)
  (average_data_rma, max_data_rma, min_data_rma, num_msg_data_rma, msg_size_data_rma) = print_data(fp_3, rma_dict_average, rma_dict_max, rma_dict_min, rma_dict_num_msg, rma_dict_msg_size)

  (average_keys_rma_dynamic, max_keys_rma_dynamic, min_keys_rma_dynamic, num_msg_keys_rma_dynamic, msg_size_keys_rma_dynamic) = get_and_sort_keys(rma_dynamic_dict_average, rma_dynamic_dict_max, rma_dynamic_dict_min, rma_dynamic_dict_num_msg, rma_dynamic_dict_msg_size)
  (average_data_rma_dynamic, max_data_rma_dynamic, min_data_rma_dynamic, num_msg_data_rma_dynamic, msg_size_data_rma_dynamic) = print_data(fp_4, rma_dynamic_dict_average, rma_dynamic_dict_max, rma_dynamic_dict_min, rma_dynamic_dict_num_msg, rma_dynamic_dict_msg_size)


  
  make_time_plot(average_keys_standard, average_data_standard, f"{matrix} average run time on {machine_name} (standard method)", f"./{matrix}/plots/{matrix}_{machine_name}_standard_average_plot.png")
  make_time_plot(max_keys_standard, max_data_standard, f"{matrix} max run time on {machine_name} (standard method)", f"./{matrix}/plots/{matrix}_{machine_name}_standard_max_plot.png")
  make_time_plot(min_keys_standard, min_data_standard, f"{matrix} min run time on {machine_name} (standard method)", f"./{matrix}/plots/{matrix}_{machine_name}_standard_min_plot.png")
  make_msg_num_plot(num_msg_keys_standard, num_msg_data_standard, matrix, machine_name, "standard")
  make_msg_size_plot(msg_size_keys_standard, msg_size_data_standard, matrix, machine_name, "standard")

  make_time_plot(average_keys_torsten, average_data_torsten, f"{matrix} average run time on {machine_name} (torsten's method)", f"./{matrix}/plots/average/{matrix}_{machine_name}_torsten_average_plot.png")
  make_time_plot(max_keys_torsten, max_data_torsten, f"{matrix} max run time on {machine_name} (torsten's method)", f"./{matrix}/plots/max/{matrix}_{machine_name}_torsten_max_plot.png")
  make_time_plot(min_keys_torsten, min_data_torsten, f"{matrix} min run time on {machine_name} (torsten's method)", f"./{matrix}/plots/min/{matrix}_{machine_name}_torsten_min_plot.png")
  make_msg_num_plot(num_msg_keys_torsten, num_msg_data_torsten, matrix, machine_name, "torsten")
  make_msg_size_plot(msg_size_keys_torsten, msg_size_data_torsten, matrix, machine_name, "torsten")

  make_time_plot(average_keys_rma, average_data_rma, f"{matrix} average run time on {machine_name} (RMA)", f"./{matrix}/plots/average/{matrix}_{machine_name}_RMA_average_plot.png")
  make_time_plot(max_keys_rma, max_data_rma, f"{matrix} max run time on {machine_name} (RMA)", f"./{matrix}/plots/max/{matrix}_{machine_name}_RMA_max_plot.png")
  make_time_plot(min_keys_rma, min_data_rma, f"{matrix} min run time on {machine_name} (RMA)", f"./{matrix}/plots/min/{matrix}_{machine_name}_RMA_min_plot.png")
  make_msg_num_plot(num_msg_keys_rma, num_msg_data_rma, matrix, machine_name, "RMA")
  make_msg_size_plot(msg_size_keys_rma, msg_size_data_rma, matrix, machine_name, "RMA")

  make_time_plot(average_keys_rma_dynamic, average_data_rma_dynamic, f"{matrix} average run time on {machine_name} (RMA_DYNAMIC)", f"./{matrix}/plots/average/{matrix}_{machine_name}_RMA_DYNAMIC_average_plot.png")
  make_time_plot(max_keys_rma_dynamic, max_data_rma_dynamic, f"{matrix} max run time on {machine_name} (RMA_DYNAMIC)", f"./{matrix}/plots/max/{matrix}_{machine_name}_RMA_DYNAMIC_max_plot.png")
  make_time_plot(min_keys_rma_dynamic, min_data_rma_dynamic, f"{matrix} min run time on {machine_name} (RMA_DYNAMIC)", f"./{matrix}/plots/min/{matrix}_{machine_name}_RMA_DYNAMIC_min_plot.png")
  make_msg_num_plot(num_msg_keys_rma_dynamic, num_msg_data_rma_dynamic, matrix, machine_name, "RMA_DYNAMIC")
  make_msg_size_plot(msg_size_keys_rma_dynamic, msg_size_data_rma_dynamic, matrix, machine_name, "RMA_DYNAMIC")

  plt.plot(average_keys_standard, average_data_standard, average_keys_torsten, average_data_torsten, average_keys_rma, average_data_rma, average_keys_rma_dynamic, average_data_rma_dynamic)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (standard vs torsten vs RMA vs dynamic RMA)")
  plt.legend(["standard", "torsten", "RMA","dynamic RMA"])
  plt.savefig(f"./{matrix}/plots/average/{matrix}_{machine_name}_compare_average_plot.png")
  plt.clf()

  plt.plot(max_keys_standard, max_data_standard, max_keys_torsten, max_data_torsten, max_keys_rma, max_data_rma, max_keys_rma_dynamic, max_data_rma_dynamic)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (standard vs torsten vs RMA vs dynamic RMA)")
  plt.legend(["standard", "torsten", "RMA", "dynamic RMA"])
  plt.savefig(f"./{matrix}/plots/max/{matrix}_{machine_name}_compare_max_plot.png")
  plt.clf()

  plt.plot(min_keys_standard, min_data_standard, min_keys_torsten, min_data_torsten, min_keys_rma, min_data_rma, min_keys_rma_dynamic, min_data_rma_dynamic)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} min run time on {machine_name} (standard vs torsten vs RMA vs dynamic RMA)")
  plt.legend(["standard", "torsten", "RMA", "dynamic RMA"])
  plt.savefig(f"./{matrix}/plots/min/{matrix}_{machine_name}_compare_min_plot.png")
  plt.clf()


for matrix in matrix_directories:
  create_dirs(matrix) 
  out_strings_standard = []
  out_strings_torsten = []
  out_strings_rma = []
  out_strings_rma_dynamic = []
  single_node_standard = None
  many_node_standard = None
  single_node_torsten = None
  many_node_torsten = None
  single_node_rma = None
  many_node_rma = None
  single_node_rma_dynamic = None
  many_node_rma_dynamic = None
  try:
    single_node_standard = open(f"./{matrix}/{matrix}_{machine_name}_Standard_one_node", 'r')
  except:
   single_node_standard = None
  try:
    many_node_standard = open(f"./{matrix}/{matrix}_{machine_name}_Standard_many_node",'r')
  except:
   many_node_standard = None
  try:
    single_node_torsten = open(f"./{matrix}/{matrix}_{machine_name}_Torsten_one_node",'r')
  except:
    single_node_torsten = None
  try:
    many_node_torsten = open(f"./{matrix}/{matrix}_{machine_name}_Torsten_many_node",'r')
  except:
    many_node_torsten = None
  try:
    single_node_rma = open(f"./{matrix}/{matrix}_{machine_name}_RMA_one_node", 'r')
  except:
    single_node_rma = None
  try:
    many_node_rma = open(f"./{matrix}/{matrix}_{machine_name}_RMA_many_node", 'r')
  except:
    many_node_rma = None
  try: 
    single_node_rma_dynamic = open(f"./{matrix}/{matrix}_{machine_name}_RMA_DYNAMIC_one_node",'r')
  except:
    single_node_rma_dynamic = None
  try:
    many_node_rma_dynamic = open(f"./{matrix}/{matrix}_{machine_name}_RMA_DYNAMIC_many_node",'r')
  except:
    many_node_rma_dynamic = None
  
  file_strings_map = [(out_strings_standard,single_node_standard,"STANDARD"),(out_strings_standard,many_node_standard,"STANDARD"),
                      (out_strings_torsten,single_node_torsten,"TORSTEN"),(out_strings_torsten,many_node_torsten,"TORSTEN"),
                      (out_strings_rma,single_node_rma,"RMA"),(out_strings_rma,many_node_rma,"RMA"),
                      (out_strings_rma_dynamic,single_node_rma_dynamic,"RMA_DYNAMIC"),(out_strings_rma_dynamic,many_node_rma_dynamic,"RMA_DYNAMIC")]


  for (out_list,out_file,out_name) in file_strings_map:
    if out_file == None:
      continue
    print(out_file.name)
    for line in out_file.read().splitlines():
      if (line.strip().split(',')[0] != out_name) and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != 'MAX_MSG_COUNT'):
        continue
      out_list.append(line)

  fp_1 = open(f"./{matrix}/one_test_output/{matrix}_{machine_name}_standard_data.txt","w")
  fp_2 = open(f"./{matrix}/one_test_output/{matrix}_{machine_name}_torsten_data.txt","w")
  fp_3 = open(f"./{matrix}/one_test_output/{matrix}_{machine_name}_RMA_data.txt","w")
  fp_4 = open(f"./{matrix}/one_test_output/{matrix}_{machine_name}_RMA_DYNAMIC_data.txt","w")

  # Add data to standard dictionaries
  parse_output_strings(standard_dict_average, standard_dict_max, standard_dict_min, standard_dict_num_msg, standard_dict_msg_size, out_strings_standard)
  # Add data to torsten dictionaries
  parse_output_strings(torsten_dict_average, torsten_dict_max, torsten_dict_min, torsten_dict_num_msg, torsten_dict_msg_size, out_strings_torsten)
  # Add data to RMA dictionaries
  parse_output_strings(rma_dict_average, rma_dict_max, rma_dict_min, rma_dict_num_msg, rma_dict_msg_size, out_strings_rma);
  # Add data to RMA_DYNAMIC dictionaries
  parse_output_strings(rma_dynamic_dict_average, rma_dynamic_dict_max, rma_dynamic_dict_min, rma_dynamic_dict_num_msg, rma_dynamic_dict_msg_size, out_strings_rma_dynamic)
 
  visualize_data(fp_1, fp_2, fp_3, fp_4, matrix, machine_name)
  fp_1.close() 
  fp_2.close()
  fp_3.close()
  fp_4.close()