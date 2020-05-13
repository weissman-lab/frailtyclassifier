

#!/bin/bash


embeddings=/proj/cwe/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin
structured_data_path=/media/drv2/andrewcd2/frailty/output/impdat_dums.csv
outdir=output/notes_labeled_embedded/


files=$(ls -d /media/drv2/andrewcd2/frailty/annotation/* | grep .zip)
for f in $files
do 
	echo $f
	python _05_ingest_annotations.py -z $f -e $embeddings -s $structured_data_path -o $outdir
done

