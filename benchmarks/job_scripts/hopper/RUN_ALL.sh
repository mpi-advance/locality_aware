for i in *MANY_NODE*
do
  echo $i
  sbatch $i
done
