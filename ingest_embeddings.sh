

#!/bin/bash

zipfile=/Users/crandrew/projects/GW_PAIR_frailty_classifier/annotation/old_misc/frailty_phenotype_batch_1_2020-03-02_1328.zip
embeddings=/Users/crandrew/projects/pwe/output/trained_models/w2v_oa_all_300d.bin
structured_data_path=/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/impdat_dums.csv
outdir=output/

python _05_ingest_annotations.py -z $zipfile -e $embeddings -s $structured_data_path -o $outdir

