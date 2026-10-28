import glob
import pandas as pd

csv_files = glob.glob("*.csv")


df_list = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(df_list, ignore_index=True)

no_duped_ids = combined_df.drop_duplicates(subset=["id"])

#df = pd.read_csv("nova_inaturalist_data.csv")

no_duped_ids.to_csv("nova_inaturalist_data.csv", index=False)

print(
    f"merged {len(csv_files)} datasets: {len(no_duped_ids)} total unique observations"
)