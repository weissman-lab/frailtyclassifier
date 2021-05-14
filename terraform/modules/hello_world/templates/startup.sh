#!/bin/bash

echo 'Hello from startup.sh'

IP=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
echo "$IP"

# # sudo apt-get update && sudo apt-get dist-upgrade -y
sudo apt-get install nfs-common python3-pip -y

echo "attach shared drive"
sudo cp /etc/fstab /etc/fstabed
sudo echo "170.166.23.6:/data      /share                  nfs     rw              1 2     " >> /etc/fstab
sudo mkdir /share
sudo mount -a

curl -X POST -H 'Content-type: application/json' --data '{"text":"'$IP'"}' https://hooks.slack.com/services/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

# docker
sudo apt-get remove docker docker-engine docker.io containerd runc -Y
sudo apt-get update -Y
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common -Y

curl https://get.docker.com | sh && sudo systemctl --now enable docker

sudo docker build -t dockertest /share/gwlab/frailty/dockertest/
sudo systemctl start docker
sudo docker run --gpus all -it -v /share/gwlab/frailty:/usr/src/app dockertest _06_data_processing.py -b 01 --do_final_80_20
curl -X POST -H 'Content-type: application/json' --data '{"text":"data processing done"}' https://hooks.slack.com/services/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
sudo docker run --gpus all -it -v /share/gwlab/frailty:/usr/src/app dockertest _06_AL_train.py -b 01 --earlystopping --model_type bioclinicalbert
curl -X POST -H 'Content-type: application/json' --data '{"text":"multitask bcb done"}' https://hooks.slack.com/services/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

