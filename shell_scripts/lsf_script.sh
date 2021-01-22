
#!/bin/bash
#BSUB -J ACD_job            # LSF job name
#BSUB -o ACD_job.%J.out     # Name of the job output file 
#BSUB -e ACD_job.%J.error   # Name of the job error file
#BSUB -M 999999999   			 
#BSUB -R "span[hosts=1] rusage [mem=999999999]" 
#BSUB -q gpu


echo "GPU Job 1"
conda activate frailty_tf
echo "tensorflow version:" 
python -c "import tensorflow as tf; print(tf.__version__)"
echo "python version:"
python -V
python _08_window_classifier.py

