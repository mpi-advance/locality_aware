import os

matrix_names = ["D_10","bcsstk01","ch-5-b1","dwt-162","gams10a","gams10am","impcol_c","odepa400","oscil_dcop_11","tumorAntiAngiogenesis_4","west0132","ww_36_pmec_36","3elt","abb313","M40PI_n1","M80PI_n1"]

# Creates batch files which run the tests using just one node
def Create_One_Node_Test(m_name : str, algo : str, out_name : str):
  fp = open(f"{m_name}_{algo}_HOPPER_ONE_NODE.sh", "w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_{out_name}_one_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_{out_name}_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=1\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for k in range(2,33):
    fp.write(f"srun --partition=general --nodes=1 --ntasks={k} --time=00:30:00 ../../../build_hopper/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 10 {m_name} {algo}\n")
  fp.close()

# Creates batch files which run the tests using two nodes
def Create_Many_Node_Test(m_name : str, algo : str, out_name : str):
  fp = open(f"{m_name}_{algo}_HOPPER_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_{out_name}_many_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_{out_name}_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=64\n")
  fp.write("#SBATCH --nodes=2\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for k in range(33,65):
    fp.write(f"srun --partition=general --nodes=2 --ntasks={k} --time=00:30:00 ../../../build_hopper/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 10 {m_name} {algo}\n")
  fp.close()


for (i, m_name) in enumerate(matrix_names):
  if(not os.path.exists(f"../../../benchmark_tests/standard_torsten/{m_name}/")):
    os.mkdir(f"../../../benchmark_tests/standard_torsten/{m_name}/")

  # CREATE ONE NODE TEST CASES
  Create_One_Node_Test(m_name, "STANDARD", "Standard")
  Create_One_Node_Test(m_name, "TORSTEN", "Torsten")
  Create_One_Node_Test(m_name, "RMA", "RMA")
  Create_One_Node_Test(m_name, "RMA_DYNAMIC", "RMA_DYNAMIC")

  # CREATE MANY NODE TEST CASES
  Create_Many_Node_Test(m_name, "STANDARD", "Standard")
  Create_Many_Node_Test(m_name, "TORSTEN", "Torsten")
  Create_Many_Node_Test(m_name, "RMA", "RMA")
