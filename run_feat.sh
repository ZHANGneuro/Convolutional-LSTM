#!/bin/bash

declare -a a_attempt=(0)
declare -a ks_pool=(12)
#declare -a ks_pool=(1 3 5 7 9 11 13)
declare -a abs_pool=(7)
for ith_a in "${a_attempt[@]}";
  do
  for i in "${ks_pool[@]}";
  do
    for j in "${abs_pool[@]}";
    do
    (
      python /Users/bo/Documents/PycharmProjects/dingwei_relational_memory/shell_CLC.py $i $j $ith_a
  #    echo $i $j
    )&
    if (( $(wc -w <<<$(jobs -p)) % 1 == 0 )); then wait; fi
    done
  done
done




