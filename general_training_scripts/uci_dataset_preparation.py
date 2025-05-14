from ucimlrepo import fetch_ucirepo 
import os

my_map = {
    "abalone": 1,
    "wine_quality": 186,
    "obesity": 544,
    "bike_sharing": 275,
    "power_consumption": 235,
    "forest_fires": 162,
    "heart_failure": 519,
    "real_estate_taiwan": 477,
    "energy_efficiency": 242,
}

for dataset_name, dataset_id in my_map.items():
    print(f"Dataset: {dataset_name}, ID: {dataset_id}")
  
    # fetch dataset 
    dataset = fetch_ucirepo(id=dataset_id) 

    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets

    # Create a directory if it doesn't exist
    os.makedirs("/Users/janniklasgroeneveld/repositories/iclr2025_7302/datasets", exist_ok=True)

    # Combine features and targets
    X["target"] = y

    # Save to CSV
    X.to_csv(f"/Users/janniklasgroeneveld/repositories/iclr2025_7302/datasets/{dataset_name}.csv", index=False)

    # metadata 
    print(dataset.metadata) 

    # variable information 
    print(dataset.variables)
