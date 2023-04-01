for i in *
do
  cat ./${i}/${i}_Hopper_RMA_one_node_RMA ./${i}/${i}_Hopper_RMA_many_node_RMA > ./${i}/${i}_Hopper_RMA_Detailed_Timing
done
