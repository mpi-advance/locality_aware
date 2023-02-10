import os
import matplotlib.pyplot as plt

machine_name = "Hopper"
matrix_directories = next(os.walk('.'))[1]

standard_dict_average = dict()
standard_dict_max = dict()

torsten_dict_average = dict()
torsten_dict_max = dict()

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

  fp_1.write("AVERAGE DATA:\n")
  for k in average_keys_standard:
    fp_1.write(f"{k},{standard_dict_average.get(k)[1]/standard_dict_average.get(k)[0]:.6f}\n")
    average_data_standard.append(1000*(standard_dict_average.get(k)[1]/standard_dict_average.get(k)[0]))
  
  fp_1.write("MAX DATA:\n")
  for k in max_keys_standard:
    fp_1.write(f"{k},{standard_dict_max.get(k):.6f}")
    max_data_standard.append(1000*(standard_dict_max.get(k)))
  
  fp_2.write("AVERAGE DATA:\n")
  for k in average_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_average.get(k)[1]/torsten_dict_average.get(k)[0]:.6f}\n")
    average_data_torsten.append(1000*(torsten_dict_average.get(k)[1]/torsten_dict_average.get(k)[0]))

  fp_2.write("MAX DATA:\n")
  for k in max_keys_torsten:
    fp_2.write(f"{k},{torsten_dict_max.get(k):.6f}\n")
    max_data_torsten.append(1000*(torsten_dict_max.get(k)))
  

  plt.plot(average_keys_standard, average_data_standard, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_average_plot.png")
  plt.clf()

  plt.plot(max_keys_standard, max_data_standard, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (standard method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_standard_max_plot.png")
  plt.clf()

  plt.plot(average_keys_torsten, average_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_average_plot.png")
  plt.clf()

  plt.plot(max_keys_torsten, max_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (torsten's method)")
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_torsten_max_plot.png")
  plt.clf()
  

  plt.plot(average_keys_standard, average_data_standard, '-.', average_keys_torsten, average_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} average run time on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_average_plot.png")
  plt.clf()

  plt.plot(max_keys_standard, max_data_standard, max_keys_torsten, max_data_torsten, '-.')
  plt.xlabel("Number of Processes")
  plt.ylabel("Time Taken (ms)")
  plt.title(f"{matrix} max run time on {machine_name} (standard vs torsten)")
  plt.legend(["standard", "torsten"])
  plt.savefig(f"./{matrix}/{matrix}_{machine_name}_compare_max_plot.png")
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
    if (line.strip().split(',')[0] != "STANDARD") and (not line.replace('.','',1).isdigit()):
      continue
    out_strings_standard.append(line)
  
  for line in many_node_standard.read().splitlines():
    if (line.strip().split(',')[0] != "STANDARD") and (not line.replace('.','',1).isdigit()):
      continue
    out_strings_standard.append(line)

  for line in single_node_torsten.read().splitlines():
    if(line.strip().split(',')[0] != "TORSTEN") and (not line.replace('.','',1).isdigit()):
      continue
    out_strings_torsten.append(line)

  for line in many_node_torsten.read().splitlines():
    if(line.strip().split(',')[0] != "TORSTEN") and (not line.replace('.','',1).isdigit()):
      continue
    out_strings_torsten.append(line)

  fp_1 = open(f"./{matrix}/{matrix}_{machine_name}_standard_data.txt","w")
  fp_2 = open(f"./{matrix}/{matrix}_{machine_name}_torsten_data.txt","w")

  # Add data to standard dictionaries
  i = 0
  while i < len(out_strings_standard):
    line_parts = out_strings_standard[i].split(',')
    num_procs = int(line_parts[1].strip().split(' ')[0])
    num_tests = int(line_parts[2].strip().split(' ')[0])
    for j in range(num_tests):
      i = i+1
      line_parts = out_strings_standard[i].split(',')
      time_taken = float(line_parts[0])
      # Update average dictionary
      x = standard_dict_average.get(num_procs)
      if x == None:
        standard_dict_average.update({num_procs:(1,time_taken)})
      else:
        standard_dict_average.update({num_procs:(x[0]+1,x[1]+time_taken)})
      # Update max dictionary
      x = standard_dict_max.get(num_procs)
      if x == None:
        standard_dict_max.update({num_procs: time_taken})
      else:
        if (x > time_taken): 
          standard_dict_max.update({num_procs : x})
        else:
          standard_dict_max.update({num_procs : time_taken})
    i = i + 1

  # Add data to torsten dictionaries
  i = 0
  while i < len(out_strings_torsten):
    line_parts = out_strings_torsten[i].split(',')
    num_procs = int(line_parts[1].strip().split(' ')[0])
    num_tests = int(line_parts[2].strip().split(' ')[0])
    for j in range(num_tests):
      i = i+1
      line_parts = out_strings_torsten[i].split(',')
      time_taken = float(line_parts[0])
      # Update average dictionary
      x = torsten_dict_average.get(num_procs)
      if x == None:
        torsten_dict_average.update({num_procs:(1,time_taken)})
      else:
        torsten_dict_average.update({num_procs:(x[0]+1,x[1]+time_taken)})
      # Update max dictionary
      x = torsten_dict_max.get(num_procs)
      if x == None:
        torsten_dict_max.update({num_procs: time_taken})
      else:
        if (x > time_taken): 
          torsten_dict_max.update({num_procs : x})
        else:
          torsten_dict_max.update({num_procs : time_taken})
    i = i + 1
 
  visualize_data(fp_1, fp_2, matrix, machine_name)
  fp_1.close() 
  fp_2.close()