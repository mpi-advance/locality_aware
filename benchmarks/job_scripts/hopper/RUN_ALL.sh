for i in *NODE*
do
  echo $i
  sbatch $i
done
