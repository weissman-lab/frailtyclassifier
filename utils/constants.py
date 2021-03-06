
SENTENCE_LENGTH = 20 # set standard sentence length. Inputs will be truncated or padded

ROBERTA_MAX_TOKS = 200

TAGS = ['Fall_risk', 'Msk_prob',  'Nutrition', 'Resp_imp']

OUT_VARNAMES = ['Msk_prob', 'Nutrition', 'Resp_imp', 'Fall_risk']

STR_VARNAMES = ['n_encs', 'n_ed_visits', 'n_admissions', 'days_hospitalized',
                'mean_sys_bp', 'mean_dia_bp', 'sd_sys_bp', 'sd_dia_bp',
                'bmi_mean', 'bmi_slope', 'max_o2', 'spo2_worst', 'ALBUMIN',
                'ALKALINE_PHOSPHATASE', 'AST', 'BILIRUBIN', 'BUN', 'CALCIUM',
                'CO2', 'CREATININE', 'HEMATOCRIT', 'HEMOGLOBIN', 'LDL', 'MCHC',
                'MCV', 'PLATELETS', 'POTASSIUM', 'PROTEIN', 'RDW', 'SODIUM',
                'WBC', 'sd_ALBUMIN', 'sd_ALKALINE_PHOSPHATASE', 'sd_AST',
                'sd_BILIRUBIN', 'sd_BUN', 'sd_CALCIUM', 'sd_CO2',
                'sd_CREATININE', 'sd_HEMATOCRIT', 'sd_HEMOGLOBIN', 'sd_LDL',
                'sd_MCHC', 'sd_MCV', 'sd_PLATELETS', 'sd_POTASSIUM',
                'sd_PROTEIN', 'sd_RDW', 'sd_SODIUM', 'sd_WBC', 'n_ALBUMIN',
                'n_ALKALINE_PHOSPHATASE', 'n_AST', 'n_BILIRUBIN', 'n_BUN',
                'n_CALCIUM', 'n_CO2', 'n_CREATININE', 'n_HEMATOCRIT',
                'n_HEMOGLOBIN', 'n_LDL', 'n_MCHC', 'n_MCV', 'n_PLATELETS',
                'n_POTASSIUM', 'n_PROTEIN', 'n_RDW', 'n_SODIUM', 'n_WBC',
                'FERRITIN', 'IRON', 'MAGNESIUM', 'TRANSFERRIN',
                'TRANSFERRIN_SAT', 'sd_FERRITIN', 'sd_IRON', 'sd_MAGNESIUM',
                'sd_TRANSFERRIN', 'sd_TRANSFERRIN_SAT', 'n_FERRITIN', 'n_IRON',
                'n_MAGNESIUM', 'n_TRANSFERRIN', 'n_TRANSFERRIN_SAT', 'PT',
                'sd_PT', 'n_PT', 'PHOSPHATE', 'sd_PHOSPHATE', 'n_PHOSPHATE',
                'PTT', 'sd_PTT', 'n_PTT', 'TSH', 'sd_TSH', 'n_TSH',
                'n_unique_meds', 'elixhauser', 'n_comorb', 'AGE', 'SEX_Female',
                'SEX_Male', 'MARITAL_STATUS_Divorced', 'MARITAL_STATUS_Married',
                'MARITAL_STATUS_Other', 'MARITAL_STATUS_Single',
                'MARITAL_STATUS_Widowed', 'EMPY_STAT_Disabled',
                'EMPY_STAT_Full Time', 'EMPY_STAT_Not Employed',
                'EMPY_STAT_Other', 'EMPY_STAT_Part Time', 'EMPY_STAT_Retired',
                'MV_n_encs', 'MV_n_ed_visits', 'MV_n_admissions',
                'MV_days_hospitalized', 'MV_mean_sys_bp', 'MV_mean_dia_bp',
                'MV_sd_sys_bp', 'MV_sd_dia_bp', 'MV_bmi_mean', 'MV_bmi_slope',
                'MV_max_o2', 'MV_spo2_worst', 'MV_ALBUMIN',
                'MV_ALKALINE_PHOSPHATASE', 'MV_AST', 'MV_BILIRUBIN', 'MV_BUN',
                'MV_CALCIUM', 'MV_CO2', 'MV_CREATININE', 'MV_HEMATOCRIT',
                'MV_HEMOGLOBIN', 'MV_LDL', 'MV_MCHC', 'MV_MCV', 'MV_PLATELETS',
                'MV_POTASSIUM', 'MV_PROTEIN', 'MV_RDW', 'MV_SODIUM', 'MV_WBC',
                'MV_sd_ALBUMIN', 'MV_sd_ALKALINE_PHOSPHATASE', 'MV_sd_AST',
                'MV_sd_BILIRUBIN', 'MV_sd_BUN', 'MV_sd_CALCIUM', 'MV_sd_CO2',
                'MV_sd_CREATININE', 'MV_sd_HEMATOCRIT', 'MV_sd_HEMOGLOBIN',
                'MV_sd_LDL', 'MV_sd_MCHC', 'MV_sd_MCV', 'MV_sd_PLATELETS',
                'MV_sd_POTASSIUM', 'MV_sd_PROTEIN', 'MV_sd_RDW',
                'MV_sd_SODIUM', 'MV_sd_WBC', 'MV_n_ALBUMIN',
                'MV_n_ALKALINE_PHOSPHATASE', 'MV_n_AST', 'MV_n_BILIRUBIN',
                'MV_n_BUN', 'MV_n_CALCIUM', 'MV_n_CO2', 'MV_n_CREATININE',
                'MV_n_HEMATOCRIT', 'MV_n_HEMOGLOBIN', 'MV_n_LDL', 'MV_n_MCHC',
                'MV_n_MCV', 'MV_n_PLATELETS', 'MV_n_POTASSIUM', 'MV_n_PROTEIN',
                'MV_n_RDW', 'MV_n_SODIUM', 'MV_n_WBC', 'MV_FERRITIN',
                'MV_IRON', 'MV_MAGNESIUM', 'MV_TRANSFERRIN',
                'MV_TRANSFERRIN_SAT', 'MV_sd_FERRITIN', 'MV_sd_IRON',
                'MV_sd_MAGNESIUM', 'MV_sd_TRANSFERRIN',
                'MV_sd_TRANSFERRIN_SAT', 'MV_n_FERRITIN', 'MV_n_IRON',
                'MV_n_MAGNESIUM', 'MV_n_TRANSFERRIN', 'MV_n_TRANSFERRIN_SAT',
                'MV_PT', 'MV_sd_PT', 'MV_n_PT', 'MV_PHOSPHATE',
                'MV_sd_PHOSPHATE', 'MV_n_PHOSPHATE', 'MV_PTT', 'MV_sd_PTT',
                'MV_n_PTT', 'MV_TSH', 'MV_sd_TSH', 'MV_n_TSH',
                'MV_n_unique_meds', 'MV_elixhauser', 'MV_n_comorb', 'MV_AGE',
                'MV_SEX', 'MV_MARITAL_STATUS', 'MV_EMPY_STAT']

if __name__ == '__main__':
    pass