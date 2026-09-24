import pandas as pd
from sklearn.model_selection import train_test_split
import sys

df = pd.read_csv(sys.argv[1])

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}")

train_df.to_csv('train.csv')
val_df.to_csv('val.csv')
