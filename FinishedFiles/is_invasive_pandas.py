import pandas as pd
#df = dataframe

df = pd.read_csv("shortened_nova_inaturalist_data.csv")
col_to_add = []

for i in range(332239):
    if (df.iloc[i])["scientific_name"] in ["Ailanthus altissima", "Acer platanoides", "Pyrus calleryana", "Paulownia tomentosa", "Celastrus orbiculatus", "Lonicera japonica", "Elaeagnus umbellata", "Rosa multiflora", "Rubus phoenicolasius", "Pueraria montana", "Ampelopsis brevipedunculata", "Euonymus alatus", "Hedera helix", "Microstegium vimineum", "Alliaria petiolata", "Persicaria perfoliata", "Ficaria verna", "Phragmites australis"]:
        col_to_add.append("Yes")
    else:
        col_to_add.append("No")

#print(f"Length: {len(col_to_add)}")

df["Invasive"] = col_to_add

df.to_csv("final_nova_inaturalist_data.csv", index=False)

print(df)