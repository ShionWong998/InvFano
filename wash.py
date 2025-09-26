import pandas as pd

df = pd.read_csv('data.csv')

values = {
    'x1':(3,12),
    'x2':(3,12),
    'phi1':(10,100),
    'phi2':(10,100),
    'alpha':(0,1),
    'ns':(0,1)
}

for col, (min, max) in values.items():
    df[col] = (df[col] - min) / (max - min)

df.to_csv('wash_all.csv', index=False)