import os

matrix_names = ["bcsstk01","ch5-5-b1","D_10","dwt_162","gams10a","gams10am","impcol_c","odepa400","oscil_dcop_11","tumorAntiAngiogenesis_4","west0132","ww_36_pmec_36","3elt","abb313","M40PI_n1","M80PI_n1"]

for (i, m_name) in enumerate(matrix_names):
  if not (os.path.exists(f"../../../benchmark_tests/standard_torsten/{m_name}/")):
    os.mkdir(f"../../../benchmark_tests/standard_torsten/{m_name}/")

  # CREATE ONE NODE STANDARD TEST CASES
  fp = open(f"{matrix_names[i]}_STANDARD_WHEELER_ONE_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Standard_one_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Standard_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition normal\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=4\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for i in range(2,9):
    fp.write(f"srun --partition=normal --nodes=1 --ntasks={i} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} STANDARD\n")
  for j in range(2,5):
    for i in range((j-1)*8+1,j*8+1):
      fp.write(f"srun --partition=normal --nodes={j} --ntasks={i} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} STANDARD\n")
  fp.close()

  # CREATE ONE NODE TORSTEN TEST CASES
  fp = open(f"{matrix_names[i]}_TORSTEN_WHEELER_ONE_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Torsten_one_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Torsten_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition normal\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=4\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for i in range(2,9):
    fp.write(f"srun --partition=normal --nodes=1 --ntasks={i} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} TORSTEN\n")
  for j in range(2,5):
    for i in range((j-1)*8+1,j*8+1):
      fp.write(f"srun --partition=normal --nodes={j} --ntasks={i} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} TORSTEN\n")
  fp.close()

  # CREATE MANY NODE STANDARD TEST CASES
  fp = open(f"{matrix_names[i]}_STANDARD_WHEELER_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Standard_many_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Standard_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition normal\n")
  fp.write("#SBATCH --ntasks=128\n")
  fp.write("#SBATCH --nodes=16\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for j in range(8,17):
    for i in range((j-1)*8+1,j*8+1):
      fp.write(f"srun --partition=normal --nodes={j} --ntasks={i} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} STANDARD\n")
  fp.close()

  # CREATE MANY NODE TORSTEN TEST CASES
  fp = open(f"{matrix_names[i]}_TORSTEN_WHEELER_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Torsten_many_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Wheeler_Torsten_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition normal\n")
  fp.write("#SBATCH --ntasks=128\n")
  fp.write("#SBATCH --nodes=16\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  for j in range(8,17):
    for i in range((j-1)*8+1,j*8+1):
      fp.write(f"srun --partition=normal --nodes={j} --ntasks={i} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} TORSTEN\n")
  fp.close()