from pathlib import Path
import pandas as pd

root_dir = "data/100k"
root_path = Path(root_dir)

for dataset in root_path.iterdir():
    print(dataset.name)
    seen = set()
    df = pd.DataFrame(columns=["image", "labels"])
    data_dir = f"{root_dir}/{dataset.name}"
    data_path = Path(data_dir)
    for item in data_path.iterdir():
        filename = item.name.split(".")[0]
        if filename not in seen:
            df.loc[len(df)] = [f"{data_dir}/{filename}.jpg", f"{data_dir}/{filename}.json"]

    df.to_csv(f"{root_dir}/{dataset.name}.csv")