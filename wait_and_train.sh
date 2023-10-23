while ps -p 2124 > /dev/null; do
  sleep 60
done

tmux new-session -d -s train_task 'bash ./train_pure.sh > /dev/null 2>&1'