import os

matrix_names = ["D_10","bcsstk01","ch-5-b1","dwt-162","gams10a","gams10am","impcol_c","odepa400","oscil_dcop_11","tumorAntiAngiogenesis_4","west0132","ww_36_pmec_36","3elt","abb313","M40PI_n1","M80PI_n1"]
f_path = "../../../benchmark_tests/comm_creation"

def Create_Varied_Runs_Test(m_name : str, algo : str, out_name : str, test_range : int):
  fp = open(f"{m_name}_{algo}_HOPPER_VARIED_ONE_NODE.sh", "w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output {f_path}/{m_name}/data/output/{m_name}_Hopper_{out_name}_varied_runs\n")
  fp.write(f"#SBATCH --error {f_path}/{m_name}/data/error/{m_name}_Hopper_{out_name}_varied_runs_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=1\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for k in range(2,33):
    for i in range(2, test_range):
      fp.write(f"srun --partition=general --nodes=1 --ntasks={k} --time=00:30:00 ../../../build_hopper/benchmarks/comm_creators ../../../test_data/{m_name}.pm {i} {m_name} {algo}\n")
  fp.close()
  fp = open(f"{m_name}_{algo}_HOPPER_VARIED_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output {f_path}/{m_name}/data/output/{m_name}_Hopper_{out_name}_varied_runs\n")
  fp.write(f"#SBATCH --error {f_path}/{m_name}/data/error/{m_name}_Hopper_{out_name}_varied_runs_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=64\n")
  fp.write("#SBATCH --nodes=2\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for k in range(33,65):
    for i in range(2, test_range):
      fp.write(f"srun --partition=general --nodes=2 --ntasks={k} --time=00:30:00 ../../../build_hopper/benchmarks/comm_creators ../../../test_data/{m_name}.pm {i} {m_name} {algo}\n")
  fp.close()



# Creates batch files which run the tests using just one node
def Create_One_Node_Test(m_name : str, algo : str, out_name : str, num_tests : int):
  fp = open(f"{m_name}_{algo}_HOPPER_ONE_NODE.sh", "w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output {f_path}/{m_name}/data/output/{m_name}_Hopper_{out_name}_one_node\n")
  fp.write(f"#SBATCH --error {f_path}/{m_name}/data/error/{m_name}_Hopper_{out_name}_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=1\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for k in range(2,33):
    fp.write(f"srun --partition=general --nodes=1 --ntasks={k} --time=00:30:00 ../../../build_hopper/benchmarks/comm_creators ../../../test_data/{m_name}.pm 1 {m_name} {algo}\n")
  fp.close()

# Creates batch files which run the tests using two nodes
def Create_Many_Node_Test(m_name : str, algo : str, out_name : str, num_tests : int):
  fp = open(f"{m_name}_{algo}_HOPPER_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output {f_path}/{m_name}/data/output/{m_name}_Hopper_{out_name}_many_node\n")
  fp.write(f"#SBATCH --error {f_path}/{m_name}/data/error/{m_name}_Hopper_{out_name}_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=64\n")
  fp.write("#SBATCH --nodes=2\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for k in range(33,65):
    fp.write(f"srun --partition=general --nodes=2 --ntasks={k} --time=00:30:00 ../../../build_hopper/benchmarks/comm_creators ../../../test_data/{m_name}.pm 1 {m_name} {algo}\n")
  fp.close()


for (i, m_name) in enumerate(matrix_names):
  if(not os.path.exists(f"{f_path}/{m_name}/")):
    os.mkdir(f"{f_path}/{m_name}/")
    os.mkdir(f"{f_path}/{m_name}/parsed_data")
    os.mkdir(f"{f_path}/{m_name}/parsed_data/tables")
    os.mkdir(f"{f_path}/{m_name}/parsed_data/plots")
    os.mkdir(f"{f_path}/{m_name}/parsed_data/one_test_output")
    os.mkdir(f"{f_path}/{m_name}/plots")
    os.mkdir(f"{f_path}/{m_name}/plots/average")
    os.mkdir(f"{f_path}/{m_name}/plots/min")
    os.mkdir(f"{f_path}/{m_name}/plots/max")
    os.mkdir(f"{f_path}/{m_name}/data")
    os.mkdir(f"{f_path}/{m_name}/data/output")
    os.mkdir(f"{f_path}/{m_name}/data/error")



  # CREATE ONE NODE TEST CASES
  num_tests = 1
  Create_One_Node_Test(m_name, "STANDARD", "Standard", num_tests)
  Create_One_Node_Test(m_name, "TORSTEN", "Torsten", num_tests)
  Create_One_Node_Test(m_name, "RMA", "RMA", num_tests)

  # CREATE MANY NODE TEST CASES
  Create_Many_Node_Test(m_name, "STANDARD", "Standard", num_tests)
  Create_Many_Node_Test(m_name, "TORSTEN", "Torsten", num_tests)
  Create_Many_Node_Test(m_name, "RMA", "RMA", num_tests)

  Create_Varied_Runs_Test(m_name, "STANDARD", "Standard", 50)
  Create_Varied_Runs_Test(m_name, "TORSTEN", "Torsten", 50)
  Create_Varied_Runs_Test(m_name, "RMA", "RMA", 50)
  Create_Varied_Runs_Test(m_name, "RMA_DYNAMIC", "RMA_DYNAMIC", 50)
