
pids=($(ps aux | grep root | grep [t]rain.py | grep -v 'killtrain.sh' | grep -v 'grep' | awk '{print $2}'))
for pid in ${pids[@]}
do
#   echo $(($pid))
  kill -9 $(($pid))
  if [ $? -eq 0 ]; then
    echo -e "\033[0;32m# succeed to kill $pid.\033[0m" 
  else
    echo -e "\033[0;31m# fail to kill $pid.\033[0m" 
  fi
done
exit 0