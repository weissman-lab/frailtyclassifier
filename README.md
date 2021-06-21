# Frailty Classifier

## Data extraction and processing

All data were extracted from electronic health records of patients of University of Pennsylvania health system, and as such cannot be shared.  Code used to extract and process the data can be found in `./data_pulling_cleaning`.

## Note ingestion

Clinical notes were uploaded to a secure [webanno](https://webanno.github.io/webanno/) server.  To create data from curated webanno projects, use `shell_scripts/ingest.sh`.  Modify that script to point to whichever webanno zip by modifying the target of the grep statement, and ensure that the environment variables for the embeddings and the output directory are as intended.  These are: 

```
embeddings=./data/w2v_oa_all_300d.bin
outdir=./output/notes_labeled_embedded_SENTENCES/
```

The shell script calls the `main` function in `_05_ingest_to_sentences.py`. We used 300-dimensional word2vec embeddings trained on the PubMed Open Access Subset that are available for download [here](https://github.com/weissman-lab/clinical_embeddings).

## Creation of training/validation data, and full training data

Call `_06_data_processing.py -b <<batchstring>> --do_folds --do_full` where batchsting is an integer with two digits and a leading zero if neccesary, e.g.: `05`.  This will create a directory at `./output/saved_models/AL<<batchstring>>`, which will have a directory `processed_data` that contains all that's needed to run the AL pipeline; all the folds and all the repeats.

`--do_folds` makes all of the data subsets for cross-validation
`--do_full` makes the final data set, for use after cross-validation has selected the number of epochs

## Doing cross-validation

The main script for neural network models is `_07_AL_CV.py`.  It takes the `batchstring` argument, as above.  It will loop through the hyperparameter grid, serially fitting all the models in it.  This is several hundred models.

The main script for elastic net models is `_08_enet.R` and the main script for random forest models is `_08_rf.R`. They both take `batchstring` as an argument, and they will run each set of hyperparameters for a set number of folds and repeats.

### Using terraform and a swarm of machines

To run in parallel, use terraform to provision a swarm of virtual machines, and give them the following bash script to run as part of a cloud init:

```
IP=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)

sudo apt-get install nfs-common python3-pip -y

echo "attach shared NFS drive to machines"
sudo echo "your_shared_drive_address   /share        nfs     rw      1 2" >> /etc/fstab
sudo mkdir /share
sudo mount -a

# install docker on VMs
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

sudo docker run --gpus all -v /share/gwlab/frailty:/usr/src/app frailty_active_learning _07_AL_CV.py -b <<batchstring>>
```

Note the hard-coded batchstring down near the bottom.   

The machines will loop through the hyperparameter grid in parallel.  There is a system in place to avoid most clobbering/overwriting, based on tokens.  

Note that the code will try to send slack messages.  The easiset way to enable these is to change the default argument in `utils.misc.send_message_to_slack`

## Fitting the final model and predicting unlabeled notes

For neural network models, this is done with `_08_AL_train.py`.  It only takes `<<batchstring>>` as an argument.  

## Predicting unlabeled notes

Done with `_09_AL_predict.py`.  It only takes `<<batchstring>>` as an argument.  Unlabeled notes are batched into 100 batches by the last 2 digits of their `PAT_ID`, and the script loops through the batches, saving json.bz2 data frames with the entropies of the notes predicted.  As with cross-validation, this can be run in parallel on a cluster, using terraform.  Clobbering is avoided through a tokening system.

## Pulling the subsequent AL batch

Done with `_10_pull_best_notes.py`.  It only takes `<<batchstring>>` as an argument.  

## Testing
Done with `_11_test.py` for neural network models and `_11_test_rf_enet.R` for random forest and elastic net models.

## Notes on environment

Development was done on a mac, using conda.  All neural nets were trained on linux (ubuntu) using docker via the dockerfile in `./docker`

