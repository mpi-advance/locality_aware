import os
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

def visualize_data(fp_1 : __file__, fp_2 : __file__, matrix : str, machine_name : str):
  average_keys_standard = list(standard_dict_average)
  average_keys_standard.sort()
  average_data_standard = []

  average_keys_torsten = list(torsten_dict_average)
  average_keys_torsten.sort()
  average_data_torsten = []

  max_keys_standard = list(standard_dict_max)
  max_keys_standard.sort()
  max_data_standard = []

  max_keys_torsten = list(torsten_dict_max)
  max_keys_torsten.sort()
  max_data_torsten = []

  min_keys_standard = list(standard_dict_min)
  min_keys_standard.sort()
  min_data_standard = []

  min_keys_torsten = list(torsten_dict_min)
  min_keys_torsten.sort()
  min_data_torsten = []

  num_msg_keys_standard = list(standard_dict_num_msg)
  num_msg_keys_standard.sort()
  num_msg_data_standard = []

  num_msg_keys_torsten = list(torsten_dict_num_msg)
  num_msg_keys_torsten.sort()
  num_msg_data_torsten = []

  msg_size_keys_standard = list(standard_dict_msg_size)
  msg_size_keys_standard.sort()
  msg_size_data_standard = []

  msg_size_keys_torsten = list(torsten_dict_msg_size)
  msg_size_keys_torsten.sort()
  msg_size_data_torsten = []

  fp_1.write("AVERAGE DATA:\n")
  for k in average_keys_standard:
    fp_1.write(f"{k},{standard_dict_average.get(k)[1]/standard_dict_average.get(k)[0]:.6f}\n")
    average_data_standard.append(1000*(standard_dict_average.get(k)[1]/standard_dict_average.get(k)[0]))
  
  fp_1.write("MAX DATA:\n")
  for k in max_keys_standard:
    fp_1.write(f"{k},{standard_dict_max.get(k):.6f}\n")
    max_data_standard.append(1000*(standard_dict_max.get(k)))

  fp_1.write("MIN DATA:\n")
  for k in min_keys_standard:
    fp_1.write(f"{k},{standard_dict_min.get(k):.6f}\n")
    min_data_standard.append(1000*(standard_dict_min.get(k)))
  
  fp_1.write("NUM MESSAGE DATA:\n")
  for k in num_msg_keys_standard:
    fp_1.write(f"{k},{standard_dict_num_msg.get(k)}\n")
    num_msg_data_standard.append(standard_dict_num_msg.get(k))
  
  fp_1.write("MESSAGE SIZE DATA:\n")
  for k in msg_size_keys_standard:
    fp_1.write(f"{k},{standard_dict_msg_size.get(k)}\n")
    msg_size_data_standard.append(standard_dict_msg_size.get(k))
  

  fp_2.write("AVERAGE DATA:\n")
  for k in average_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_average.get(k)[1]/torsten_dict_average.get(k)[0]:.6f}\n")
    average_data_torsten.append(1000*(torsten_dict_average.get(k)[1]/torsten_dict_average.get(k)[0]))

  fp_2.write("MAX DATA:\n")
  for k in max_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_max.get(k):.6f}\n")
    max_data_torsten.append(1000*(torsten_dict_max.get(k)))
  
  fp_2.write("MIN DATA:\n")
  for k in min_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_min.get(k):.6f}\n")
    min_data_torsten.append(1000*(torsten_dict_min.get(k)))
  
  fp_2.write("NUM MESSAGE DATA:\n")
  for k in num_msg_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_num_msg.get(k)}\n")
    num_msg_data_torsten.append(torsten_dict_num_msg.get(k))
  
  fp_2.write("MESSAGE SIZE DATA:\n")
  for k in msg_size_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_msg_size.get(k)}\n")
    msg_size_data_torsten.append(torsten_dict_msg_size.get(k))


  plt.plot(average_keys_standard, average_data_standard)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_average_plot.png")
  plt.clf()

  plt.plot(max_keys_standard, max_data_standard)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_max_plot.png")
  plt.clf()

  plt.plot(min_keys_standard, min_data_standard)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} min run time on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_min_plot.png")
  plt.clf()

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

  plt.plot(average_keys_torsten, average_data_torsten)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_average_plot.png")
  plt.clf()

  plt.plot(max_keys_torsten, max_data_torsten)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_max_plot.png")
  plt.clf()

  plt.plot(min_keys_torsten, min_data_torsten)
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} min run time on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_min_plot.png")
  plt.clf()
  
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