import os
import re

machine_name = "Hopper"
matrix_directories = next(os.walk('.'))[1]

for matrix in matrix_directories:
  standard_dict = dict()
  out_strings = []
  single_node = open(f"./{matrix}/{matrix}_{machine_name}_Standard_one_node", 'r')

  for line in single_node.read().splitlines():
    if (line.strip().split(',')[0] != "STANDARD") and (not line.replace('.','',1).isdigit()):
      continue
    out_strings.append(line)
  i = 0
  for i in range(len(out_strings)):
    line_parts = out_strings[i].split(',')
    num_procs = int(line_parts[1].strip().split(' ')[0])
    num_tests = int(line_parts[2].strip().split(' ')[0])
    print(f"num procs {num_procs}, num_tests {num_tests}")
    for j in range(num_tests):
      i = i+1
      time_taken = float(line_parts[i].strip().split()[0])
      x = standard_dict.get(num_procs)
      if x == None:
        standard_dict.update([num_procs,(1,time_taken)])
      else:
        standard_dict.update([num_procs,(x[0]+1,x[1]+time_taken)])
  fp = open(f"./{matrix}/{matrix}_{machine_name}_data.txt","w")
  for k in standard_dict.keys():
    fp.write(f"{k},{standard_dict.get(k)[1]/standard_dict.get(k)[0]}")
  fp.close()




