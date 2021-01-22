

#!/bin/bash


# embeddings=/proj/cwe/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin
# structured_data_path=/media/drv2/andrewcd2/frailty/output/impdat_dums.csv
# outdir=output/notes_labeled_embedded/


# files=$(ls -d /media/drv2/andrewcd2/frailty/annotation/* | grep .zip)
# for f in $files
# do 
# 	echo $f
# 	python _05_ingest_annotations.py -z $f -e $embeddings -s $structured_data_path -o $outdir
# done






#files=$(ls -d /Users/crandrew/projects/GW_PAIR_frailty_classifier/annotation/* | grep .zip)
#for f in $files
#do 
#	echo $f
#	python _05_ingest_annotations.py -z $f -e $embeddings -s $structured_data_path -o $outdir
#done


outdir="./output/notes_labeled_embedded/"
zipfile="/Users/crandrew/projects/GW_PAIR_frailty_classifier/annotation/frailty_phenotype_AL_01_2020-08-13_1218.zip"
embeddings="/Users/crandrew/projects/pwe/output/trained_models/w2v_oa_all_300d.bin"
structured_data_path="/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/impdat_dums.csv"

python _05_ingest_annotations.py -z $zipfile -e $embeddings -s $structured_data_path -o $outdir

zipfile="/Users/crandrew/projects/GW_PAIR_frailty_classifier/annotation/frailty_phenotype_AL_01_ADDENDUM_2020-08-13_1218.zip"

python _05_ingest_annotations.py -z $zipfile -e $embeddings -s $structured_data_path -o $outdir