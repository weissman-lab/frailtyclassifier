
 #!/bin/bash
 #BSUB -J ACD_job            # LSF job name
 #BSUB -o ACD_job.%J.out     # Name of the job output file 
 #BSUB -e ACD_job.%J.error   # Name of the job error file

conda activate frailty_tf
python _08_window_classifier.py