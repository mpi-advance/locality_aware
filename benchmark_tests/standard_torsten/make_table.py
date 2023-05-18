import os

matrix_directories = next(os.walk('.'))[1]
machine_name = "Wheeler"

for matrix in matrix_directories:
  standard_file = open(f"./{matrix}/{matrix}_{machine_name}_Standard_varied_runs","r")
  output_lines = []
  
  ## Open File, clear out useless lines 
  for line in standard_file.read().splitlines():
    line.strip().split(',')[0] != "STANDARD"
    if (line.strip().split(',')[0] != "STANDARD") and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != 'MAX_MSG_COUNT'):
      continue
    output_lines.append(line)
  
  standard_out = open(f"./{matrix}/{matrix}_{machine_name}_Standard_table.txt","w")
  data_out = []
  
  # Data format: 
  # Array has elements [(num_tests, average_run_time)], where index represents 
  i = 0 
  while i < len(output_lines):
    print(line)
    num_procs = int(line.strip().split(',')[1].strip().split(' ')[0])
    num_tests = int(line.strip().split(',')[2].strip().split(' ')[0])
    count = 0
    i = i + 2
    for j in range(num_tests):
      count += float(output_lines[i])
      i += 1
    data_out[num_procs-2].append((num_tests, "%.4f" % (count / num_tests)))
  
  print(data_out)
    
