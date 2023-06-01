import os
import typing
import matplotlib.pyplot as plt

matrix_directories = next(os.walk('.'))[1]
machine_name = "Wheeler"
algos = ['Standard', 'Torsten', 'RMA', 'RMA_DYNAMIC']
standard_data = []
torsten_data = []
RMA_data = []
RMA_dynamic_data = []

data_map = {
  'Standard': standard_data,
  'Torsten': torsten_data,
  'RMA': RMA_data,
  'RMA_DYNAMIC': RMA_dynamic_data
}

def create_dirs(matrix : str):
  os.mkdir()
  if not os.path.exists(f"./{matrix}/parsed_data"):
    os.mkdir(f"./{matrix}/parsed_data")
    os.mkdir(f"./{matrix}/parsed_data/tables")
    os.mkdir(f"./{matrix}/parsed_data/plots")
    os.mkdir(f"./{matrix}/parsed_data/one_test_output")
    os.mkdir(f"./{matrix}/plots/average")
    os.mkdir(f"./{matrix}/plots/min")
    os.mkdir(f"./{matrix}/plots/max")


for matrix in matrix_directories:
  for algo in algos:
    standard_file = open(f"./{matrix}/{matrix}_{machine_name}_{algo}_varied_runs","r")
    output_lines = []
  
    ## Open File, clear out useless lines 
    for line in standard_file.read().splitlines():
      line.strip().split(',')[0] != algo.upper()
      if (line.strip().split(',')[0] != algo.upper()) and (not line.replace('.','',1).isdigit()) and (line.strip().split(',')[0].split(' ')[0] != 'MAX_MSG_COUNT'):
        continue
      output_lines.append(line)
  
    # create_dirs(matrix)
    standard_out = open(f"./{matrix}/tables/{matrix}_{machine_name}_{algo}_table.txt","w")
    data_out = data_map[algo]
  
    # Parse information about average runtime, write to array
    i = 0 
    prev_num_procs = 0
    while i < len(output_lines):
      num_procs = int(output_lines[i].strip().split(',')[1].strip().split(' ')[0])
      num_tests = int(output_lines[i].split(',')[2].strip().split(' ')[0])
      if num_procs != prev_num_procs: 
        data_out.append([num_procs])
      prev_num_procs = num_procs
      count = 0
      i = i + 2
      for j in range(num_tests):
        count += float(output_lines[i])
        i += 1
      data_out[-1].append((num_tests, ((count / float(num_tests)))))

    # Print out table data
    for x in data_out:
      standard_out.write(f"Number of Processes: {x[0]}\n")
      standard_out.write(f"Format (number_of_test, average_runtime)\n")
      for (n,f) in x[1:]:
        standard_out.write(f"({n},{f})\n")
      standard_out.write("\n")

  num_runs = len(data_map['Standard'])
  for x in range(num_runs):
    standard_data    = data_map['Standard'][x]
    torsten_data     = data_map['Torsten'][x]
    RMA_data         = data_map['RMA'][x]
    RMA_dynamic_data = data_map['RMA_DYNAMIC'][x]
    
    num_procs = standard_data[0]
    standard_data = standard_data[1:]
    torsten_data  = torsten_data[1:]
    RMA_data = RMA_data[1:]
    RMA_dynamic_data = RMA_dynamic_data[1:]

    plt.title(f"{matrix} ({num_procs} procs) (Wheeler)")
    plt.xlabel("Number of runs")
    plt.ylabel("Average running time")
    plt.plot([x[0] for x in standard_data], [x[1] for x in standard_data])
    plt.plot([x[0] for x in torsten_data], [x[1] for x in torsten_data])
    plt.plot([x[0] for x in RMA_dynamic_data], [x[1] for x in RMA_dynamic_data])
    #plt.plot(RMA_data)
    plt.legend(["Standard", "Torsten", "RMA_DYNAMIC"])
    plt.savefig(f"./test_fig_{num_procs}")
    plt.clf()




