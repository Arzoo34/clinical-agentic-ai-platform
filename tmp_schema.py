import pandas as pd, pathlib
base = pathlib.Path('QC Anonymized Study Files/Study 1_CPID_Input Files - Anonymization')
files = sorted(base.glob('*.xlsx'))
print(f"Found {len(files)} files in {base}")
for f in files:
    xls = pd.ExcelFile(f)
    print(f"\n== {f.name} ==")
    for sheet in xls.sheet_names[:1]:
        df = xls.parse(sheet, nrows=3)
        print(f"sheet: {sheet}")
        print(df.columns.tolist())
