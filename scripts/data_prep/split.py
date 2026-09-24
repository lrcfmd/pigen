import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "alex_mp20_PDD.csv"
SEED = 42

df = pd.read_csv(CSV_PATH)

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=SEED,
    shuffle=True
)

# Optional: reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Check sizes
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)
