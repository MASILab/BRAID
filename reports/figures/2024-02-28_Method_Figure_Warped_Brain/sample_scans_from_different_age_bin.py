import pandas as pd

df = pd.read_csv('/tmp/.GoneAfterReboot/spreadsheet/braid_test.csv')

ages = [(20+i*15) for i in range(6)]

for age in ages:
    age_min = age - 5
    age_max = age + 5

    c = (df['age'] >= age_min) & (df['age'] <= age_max) & (df['control_label']==1) & (~df['dataset'].isin(['HCPA']))
    
    df.loc[c, ].sample(40).to_csv(f"reports/figures/2024-02-28_Method_Figure_Warped_Brain/csv/{age}.csv", index=False)
