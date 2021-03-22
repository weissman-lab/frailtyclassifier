# Usage

## Note ingestion

To create data from curated webanno projects, use `shell_scripts/ingest.sh`.  Modify that script to point to whichever webanno zip by modifying the target of the grep statement, and ensure that the environment variables for the embeddings and the output directory are as intended.  Unless something changes (this was written 19 March 2021), those should be 

```
embeddings=./data/w2v_oa_all_300d.bin
outdir=./output/notes_labeled_embedded_SENTENCES/
```

The shell script calls the `main` function in `_05_ingest_to_sentences.py`.

## Creation of training/validation data, and full training data

Call `_06_data_processing.py -b <<batchstring>> --do_folds --do_full` where batchsting is an integer with two digits and a leading zero if neccesary, e.g.: `05`.  This will create a directory at `./output/saved_models/AL<<batchstring>>`, which will have a directory `processed_data` that contains all that's needed to run the AL pipeline; all the folds and all the repeats.

`--do_folds` makes all of the data subsets for cross-validation
`--do_full` makes the final data set, for use after cross-validation has selected the number of epochs

## Doing cross-validation

The main script is `_07_AL_CV.py`.  It takes the `batchstring` argument, as above.  It will loop through the hyperparameter grid, serially fitting all the models in it.  This is several hundred models.

### Using terraform and a swarm of machines

To run in parallel, use the following startup bash script from a terraform job:

```
IP=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
echo "$IP"

# # sudo apt-get update && sudo apt-get dist-upgrade -y
sudo apt-get install nfs-common python3-pip -y

echo "attach shared drive"
sudo cp /etc/fstab /etc/fstabed
sudo echo "170.166.23.6:/data      /share                  nfs     rw              1 2     " >> /etc/fstab
sudo echo "10.146.0.247:/gwshare   /gwshare        nfs     rw      1 2" >> /etc/fstab
sudo mkdir /share
sudo mkdir /gwshare
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

sudo usermod -aG docker signals
sudo docker build -t frailty_active_learning /gwshare/frailty/docker/
sudo systemctl start docker
curl -X POST -H 'Content-type: application/json' --data '{"text":"'$IP'"}' https://hooks.slack.com/services/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

sudo docker run --gpus all -v /share/gwlab/frailty:/usr/src/app frailty_active_learning _07_AL_CV.py -b <<batchstring>>
IP=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
curl -X POST -H 'Content-type: application/json' --data '{"text":"'$IP'"}' https://hooks.slack.com/services/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
```

Note the hard-coded batchstring down near the bottom.   

This will spin up the number of machines specified in `terraform.tfvars` (not part of this repo).  The machines will loop through the hyperparameter grid in parallel.  There is a system in place to avoid most clobbering/overwriting, based on tokens.  When a given worker is done looping through the hyperparameter grid, it will send a slack message to Andrew Crane-Droesch (this is hard-coded).

## Fitting the final model

This is done with `_08_AL_train.py`.  It only takes `<<batchstring>>` as an argument.  