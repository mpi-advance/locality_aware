for i in *RMA*
do
  echo $i
  sbatch $i
done
