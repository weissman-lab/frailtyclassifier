

# Structured data

Structured data extracted from the EHR, across our corpus, is summarized in table XX.  Infrequently-observed values of categorical variables were set to "other", and certain similar rare categories were merged.  Much of the data is missing.  For the purpose of 


Hyperparameter sampling:
random draws with replacement from notes, OOB being holdout set.  
record oos loss per combination
to get estimates for active learning, two approachs:
1.  Pick best (wrong)
2.  Model best (sort of right)
3.  Average.  Can assess OOB error.
4.  Weighted average based on model results


Purpose of sampling with replacement is to avoid loss of power that comes with sample splitting in the context of small datasets.

# Learning rate scheduler