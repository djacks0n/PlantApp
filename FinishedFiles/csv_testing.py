import pandas as pd

# 1. Load your dataset
df = pd.read_csv("final_nova_inaturalist_data.csv")

# 2. Define the exact 16 species for your 8 pairs
target_species = [
    # Pair 1: Tree-of-Heaven vs Winged Sumac
    "Ailanthus altissima", "Rhus copallinum",
    
    # Pair 2: Autumn Olive vs Pawpaw
    "Elaeagnus umbellata", "Asimina triloba",
    
    # Pair 3: Wineberry vs Black Raspberry
    "Rubus phoenicolasius", "Rubus occidentalis",
    
    # Pair 4: Japanese Honeysuckle vs Trumpet Creeper
    "Lonicera japonica", "Campsis radicans",
    
    # Pair 5: English Ivy vs Virginia Creeper
    "Hedera helix", "Parthenocissus quinquefolia",
    
    # Pair 6: Mile-a-minute vs Arrow-leaved Tearthumb
    "Persicaria perfoliata", "Persicaria sagittata",
    
    # Pair 7: Garlic Mustard vs White Avens
    "Alliaria petiolata", "Geum canadense",
    
    # Pair 8: Lesser Celandine vs Golden Ragwort
    "Ficaria verna", "Packera aurea"
]

# 3. Filter the dataframe to ONLY include these 16 species
df_experiment = df[df["scientific_name"].isin(target_species)].copy()

# 4. Save this as a new CSV so you don't overwrite your master file
output_filename = "lookalikes_16_classes_dataset.csv"
df_experiment.to_csv(output_filename, index=False)

# 5. Print a quick summary to verify it worked
print(f"Original dataset rows: {len(df)}")
print(f"Filtered dataset rows: {len(df_experiment)}")
print("\nFinal Image Counts per Species:")
print("-" * 35)
print(df_experiment["scientific_name"].value_counts())