#export container_twinspect_test=$(sudo docker ps | grep "twinspect:test" | awk  '{print $1}')
export container_twinspect_test=$(sudo docker ps | grep "twinspect_dump" | awk  '{print $1}')
sudo docker cp twinspect/algos/clip_pch.py "$container_twinspect_test":/twinspect/twinspect/algos/clip_pch.py
sudo docker cp twinspect/metrics/eff.py "$container_twinspect_test":/twinspect/twinspect/metrics/eff.py
sudo docker cp config.yml "$container_twinspect_test":/twinspect/config.yml


# sudo docker exec -it $container_twinspect_test bash
