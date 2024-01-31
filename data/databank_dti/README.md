## Key features of each dataset in the databank
### ADNI
- more than half of the participants are cognitively impaired
- contains memory score, executive function score, language score, visuospatial score.
    - data stored in `./data/subject_info/raw/ADNI/Assessments/Neuropsychological/ADSP_PHC_COGN_10_05_22_30Aug2023.csv`. 
    - Dictionary: `./data/subject_info/raw/ADNI/Assessments/Neuropsychological/ADSP_PHC_COGN_DICT_10_05_22_30Aug2023.csv`.
    - Doc: `./data/subject_info/raw/ADNI/Assessments/Neuropsychological/ADNI_Cognition_Methods_Psychometric_Analyses_Oct2022.pdf`. 

### BIOCARD
- ~200 subjects in total
- 2 subjects were impaired (not MCI) and turned back to normal
- 1 subject was scanned at all phases, including normal, MCI, impaired (not MCI), and dementia

### BLSA
- healthy ageing population

### ICBM
- healthy ageing
- small dataset collected from multi sites

### NACC
- in addition to diagnosis, diagnosis_detail is available for detailed classification

### OASIS3
- most are healthy ageing
- cognitive test scores are available for most sessions in the csv file: `./data/subject_info/raw/OASIS3/pychometrics/csv/OASIS3_UDSc1_cognitive_assessments.csv`. 
- Some interesting (and complete) columns: 
    - CATEGORY FLUENCY - ANIMALS
    - WMS Associate Learning (summary score)
    - FREE AND CUED SELECTIVE REMINDING TEST (total score)

### OASIS4
- most are cognitively impaired
- in addition to diagnosis, diagnosis_detail is available for detailed classification

### ROSMAPMARS
- most are normal, some are MCI, few are AD

### UKBB
- large dataset
- use the label "CNS_controls_2" to determine training set

### VMAP
- normal and MCI subjects (~1:1)

### WRAP
- most are normal

## Data Merging
After cleaning each dataset's spreadsheet one by one (see `./data/subject_info/clean`), 
we merge them together into one spreadsheet with standardized values.
Since we are required to save all UKBB related data on GDPR, 
the resulting spreadsheets, which contain UKBB rows, are saved at `/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet`.