import os

matrix_names = ["D_10","bcsstk01","ch-5-b1","dwt-162","gams10a","gams10am","impcol_c","odepa400","oscil_dcop_11","tumorAntiAngiogenesis_4","west0132","ww_36_pmec_36"]

# Create batch files for one node
def Create_One_Node_Test(m_name : str, algo : str, out_name : str):
  fp = open(f"{m_name}_{algo}_WHEELER_ONE_NODE.sh", "w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_{out_name}_one_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_{out_name}_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append\n")
  fp.write("#SBATCH --partition normal\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=4\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for i in range(2,9): 
    fp.write(f"srun --partition=normal --nodes=1 --ntasks={i} --time=24:00:00 ../../../build_wheeler/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} {algo}\n")
  for j in range(2,5):
    for i in range((j-1)*8+1, j*8+1):
      fp.write(f"srun --partition=normal --nodes={j} --ntasks={i} --time=24:00:00 ../../../build_wheeler/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} {algo}\n")
  fp.close()

# Create batch files for many nodes
def Create_Many_Node_Test(m_name : str, algo : str, out_name : str):
  fp = open(f"{m_name}_{algo}_WHEELER_MANY_NODE.sh", "w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_{out_name}_many_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_{out_name}_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append\n")
  fp.write("#SBATCH --partition normal\n")
  fp.write("#SBATCH --ntasks=128\n")
  fp.write("#SBATCH --nodes=16\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for j in range(8,17):
    for i in range((j-1)*8+1,j*8+1):
      fp.write(f"srun --partition=normal --nodes={j} --ntasks={i} --time=01:00:00 ../../../build_wheeler/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} {algo}\n")
  fp.close()


for (i, m_name) in enumerate(matrix_names):
  if(not os.path.exists(f"../../../benchmark_tests/standard_torsten/{m_name}/")):
    os.mkdir(f"../../../benchmark_tests/standard_torsten/{m_name}/")

  # CREATE ONE NODE TEST CASES
  Create_One_Node_Test(m_name, "STANDARD", "Standard")
  Create_One_Node_Test(m_name, "TORSTEN", "Torsten")
  Create_One_Node_Test(m_name, "RMA", "RMA")

  # CREATE MANY NODE TEST CASES
  Create_Many_Node_Test(m_name, "STANDARD", "Standard")
  Create_Many_Node_Test(m_name, "TORSTEN", "Torsten")
  Create_Many_Node_Test(m_name, "RMA", "RMA")
