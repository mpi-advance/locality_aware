import os

matrix_names = ["D_10","bcsstk01","ch-5-b1","dwt-162","gams10a","gams10am","impcol_c","odepa400","oscil_dcop_11","tumorAntiAngiogenesis_4","west0132","ww_36_pmec_36","3elt","abb313","M40PI_n1","M80PI_n1"]

for (i, m_name) in enumerate(matrix_names):
  if(not os.path.exists(f"../../../benchmark_tests/standard_torsten/{m_name}/")):
    os.mkdir(f"../../../benchmark_tests/standard_torsten/{m_name}/")

  # CREATE ONE NODE STANDARD TEST CASES
  fp = open(f"{matrix_names[i]}_STANDARD_HOPPER_ONE_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Standard_one_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Standard_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=1\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  
  for k in range(2,33):
    fp.write(f"srun --partition=general --nodes=1 --ntasks={k} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} STANDARD\n")
  fp.close()

  # CREATE ONE NODE TORSTEN TEST CASES
  fp = open(f"{matrix_names[i]}_TORSTEN_HOPPER_ONE_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Torsten_one_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Torsten_one_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=32\n")
  fp.write("#SBATCH --nodes=1\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")
  
  for k in range(2,33):
    fp.write(f"srun --partition=general --nodes=1 --ntasks={k} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} TORSTEN\n")
  fp.close()

  # CREATE MANY NODE STANDARD TEST CASES
  fp = open(f"{matrix_names[i]}_STANDARD_HOPPER_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Standard_many_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Standard_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=64\n")
  fp.write("#SBATCH --nodes=2\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")

  for k in range(48,65):
    fp.write(f"srun --partition=general --nodes=2 --ntasks={k} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} STANDARD\n")
  fp.close()

  # CREATE MANY NODE TORSTEN TEST CASES
  fp = open(f"{matrix_names[i]}_TORSTEN_HOPPER_MANY_NODE.sh","w")
  fp.write("#!/usr/bin/bash\n")
  fp.write(f"#SBATCH --output ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Torsten_many_node\n")
  fp.write(f"#SBATCH --error ../../../benchmark_tests/standard_torsten/{m_name}/{m_name}_Hopper_Torsten_many_node_err\n")
  fp.write(f"#SBATCH --open-mode=append")
  fp.write("#SBATCH --partition general\n")
  fp.write("#SBATCH --ntasks=64\n")
  fp.write("#SBATCH --nodes=2\n\n")
  fp.write("#SBATCH --mail-type=BEGIN,FAIL,END\n")
  fp.write("#SBATCH --mail-user=ageyko@unm.edu\n\n")
  fp.write("module load openmpi\n\n")

  for k in range(48,65):
    fp.write(f"srun --partition=general --nodes=2 --ntasks={k} --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/{m_name}.pm 1 {m_name} TORSTEN\n")
  fp.close()
