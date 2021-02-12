
#!/bin/bash

for i in {240..269}; 
	# do echo $i;
	do rsync -azvPh -e "ssh -i ~/projects/pennsignals_azure)vmss/.ssh/azure/acs_key" signals@10.146.0.9:/share/gwlab/frailty/output/saved_models/AL03/cv_models/model_pickle_cv_"$i".pkl /Users/crandrew/projects/GW_PAIR_frailty_classifier/output/saved_models/AL03/cv_models/; 

done

