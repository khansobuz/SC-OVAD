import pandas as pd
df = pd.read_csv(r"C:\Users\khanm\Desktop\lab_project\Open_vocab\ucf_extra\ucf_CLIP_rgbtest.csv",
                 sep=None, engine="python")
df.columns = [c.strip() for c in df.columns]
df["label"] = df["label"].str.strip()
print(df["label"].value_counts())
print("\nTotal:", len(df))
print("Anomaly only:")
anom = df[df["label"] != "Normal"]
print(anom["label"].value_counts())