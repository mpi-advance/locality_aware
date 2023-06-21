for i in *
do 
  echo $i
  sbatch $i
done
