import pandas as pd
path = '6932c39b908b6_detailed_problem_statements_and_datasets/Data for problem Statement 1/NEST 2.0 Data files_Anonymized/NEST 2.0 Data files_Anonymized.xlsx'
xls = pd.ExcelFile(path)
print('Sheets:', xls.sheet_names)
for sheet in xls.sheet_names:
    df = xls.parse(sheet, nrows=3)
    print(f"\n== {sheet} ==")
    print(df.columns.tolist())
