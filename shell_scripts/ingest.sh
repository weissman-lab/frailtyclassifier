

#!/bin/bash


embeddings=./data/w2v_oa_all_300d.bin
outdir=./output/notes_labeled_embedded_SENTENCES/test/




files=$(ls -d ./annotation/test/* | grep .zip )
for f in $files
do 
	echo $f
	python _05_ingest_to_sentence.py -z $f -e $embeddings -o $outdir
done

