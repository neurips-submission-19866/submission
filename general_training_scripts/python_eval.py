import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

DATASET_PATH_MAP = {
    "california_housing": "datasets/housing.csv",
    "wine_quality":       "datasets/wine_quality.csv",
    "automobile":         "datasets/automobile.csv",
    "abalone":            "datasets/abalone.csv",
    "bike_sharing":       "datasets/bike_sharing.csv",
	"forest_fires":      "datasets/forest_fires.csv",
	"heart_failure":     "datasets/heart_failure.csv",
	"real_estate_taiwan": "datasets/real_estate_taiwan.csv",
}

TARGET_MAP = {
    "california_housing": "median_house_value",
    "wine_quality":       "target",
    "automobile":         "target",
    "abalone":            "target",
    "bike_sharing":       "target",
	"forest_fires":      "target",
	"heart_failure":     "target",
	"real_estate_taiwan": "target",
}

def _preprocess_california_housing(df):
    df = df.fillna(0)
    df["ocean_proximity"] = df["ocean_proximity"].replace({
        "ISLAND": "0.0",
        "NEAR BAY": "1.0",
        "NEAR OCEAN": "2.0",
        "<1H OCEAN": "3.0",
        "INLAND": "4.0"
    }).astype(float)
    return df

def _preprocess_wine_quality(df):
    df["target"] = df["target"].astype(float)
    return df

def _preprocess_automobile(df):
    df["target"] = df["target"].astype(float)
    return df

def _preprocess_abalone(df):
    df["target"] = df["target"].astype(float)
    df["Sex"] = df["Sex"].replace({"M": "0", "F": "1", "I": "2"}).astype(float)
    return df

def _preprocess_bike_sharing(df):
    df.drop(columns=["dteday"], inplace=True, errors="ignore")
    df["target"] = df["target"].astype(float)
    return df

def _preprocess_forest_fires(df):
    df["target"] = df["target"].astype(float)
    df["month"] = df["month"].replace({
        "jan":"1","feb":"2","mar":"3","apr":"4","may":"5","jun":"6",
        "jul":"7","aug":"8","sep":"9","oct":"10","nov":"11","dec":"12"
    }).astype(float)
    df["day"] = df["day"].replace({
        "mon":"1","tue":"2","wed":"3","thu":"4","fri":"5","sat":"6","sun":"7"
    }).astype(float)
    return df

def _preprocess_heart_failure(df):
    df["target"] = df["target"].astype(float)
    return df

def _preprocess_real_estate_taiwan(df):
    df["target"] = df["target"].astype(float)
    return df

PREPROCESS_MAP = {
    "california_housing": _preprocess_california_housing,
    "wine_quality": _preprocess_wine_quality,
    "automobile": _preprocess_automobile,
    "abalone": _preprocess_abalone,
    "bike_sharing": _preprocess_bike_sharing,
    "forest_fires": _preprocess_forest_fires,
    "heart_failure": _preprocess_heart_failure,
    "real_estate_taiwan": _preprocess_real_estate_taiwan,
}

def plot_2d_kde(real_values, predicted_values, set_name="train", epoch=1, output_dir="training_output", dataset_name=""):
	x = real_values.flatten()
	y = predicted_values.flatten()
	kd = gaussian_kde(np.vstack([x, y]))
	xi = np.linspace(x.min(), x.max(), 256)
	yi = np.linspace(y.min(), y.max(), 256)
	X, Y = np.meshgrid(xi, yi)
	Z = kd(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
	plt.figure()
	plt.imshow(Z, origin="lower", aspect="auto",
			   extent=[x.min(), x.max(), y.min(), y.max()], cmap="viridis")
	plt.title(f"Real vs. Predicted (2D KDE)\n({set_name} set, epoch {epoch})")
	plt.xlabel("Real Value")
	plt.ylabel("Predicted Value")
	os.makedirs(output_dir, exist_ok=True)
	fname = os.path.join(output_dir, f"{dataset_name}_PyTorch_NN_2d_kde_{set_name}_epoch{epoch}.png")
	plt.savefig(fname)
	plt.close()

def plot_kde_distribution(data, set_name="train", epoch=1, output_dir="training_output",
						  plot_title="KDE Plot", x_label="Values", overlay_normal=False, dataset_name=""):
	d = data.flatten()
	kd = gaussian_kde(d)
	x_vals = np.linspace(d.min(), d.max(), 250)
	pdf = kd(x_vals)
	plt.figure()
	plt.plot(x_vals, pdf, label="KDE", linewidth=2)
	plt.title(plot_title)
	plt.xlabel(x_label)
	plt.ylabel("Density")
	if overlay_normal:
		norm_pdf = lambda x: np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
		plt.plot(x_vals, [norm_pdf(v) for v in x_vals],
				 label="Standard Normal PDF", linewidth=2, linestyle="--")
	plt.legend()
	os.makedirs(output_dir, exist_ok=True)
	fname = os.path.join(output_dir, f"{dataset_name}_PyTorch_NN_" + plot_title.replace(" ","_") + f"_epoch{epoch}.png")
	plt.savefig(fname)
	plt.close()

def evaluate_model(model, X, Y, set_name="train", epoch=1, output_dir="training_output", dataset_name=""):
	model.eval()
	with torch.no_grad():
		preds = model(torch.from_numpy(X).float()).cpu().numpy().flatten()
	mse = np.mean((Y - preds)**2)
	rmse = np.sqrt(mse)
	residuals = preds - Y
	var_res = np.var(residuals) if len(residuals) > 1 else 0.0
	calibration_values = residuals / (np.sqrt(var_res) + 1e-8)
	within_ = lambda r: np.mean(np.abs(calibration_values) <= r)
	print(f"Percentage of calibration values within:")
	print(f"[-1, 1]: {within_(1)}")
	print(f"[-2, 2]: {within_(2)}")
	print(f"[-3, 3]: {within_(3)}")

	plot_kde_distribution(calibration_values, set_name, epoch, output_dir,
		f"Calibration vs Standard Normal\n({set_name} set, epoch {epoch})", "z = (μ̂ - y) / σ̂", True, dataset_name)
	plot_kde_distribution(residuals, set_name, epoch, output_dir,
		f"Residuals\n({set_name} set, epoch {epoch})", "residual = μ̂ - y", False, dataset_name)
	plot_kde_distribution(Y, set_name, epoch, output_dir,
		f"Real Values\n({set_name} set, epoch {epoch})", "Real Value", False, dataset_name)
	plot_kde_distribution(preds, set_name, epoch, output_dir,
		f"Predicted Values\n({set_name} set, epoch {epoch})", "Predicted Value", False, dataset_name)
	plot_2d_kde(Y, preds, set_name, epoch, output_dir, dataset_name)

	print(f"=== Evaluation on {set_name} set (epoch {epoch}) ===")
	print(f"MSE  = {mse}")
	print(f"RMSE = {rmse}\n")

def run_python_evaluation(dataset_name, num_epochs, batch_size):
	np.random.seed(98)
	torch.manual_seed(98)

	# Load dataset via map
	df = pd.read_csv(DATASET_PATH_MAP[dataset_name])
	# Preprocess dataset
	df = PREPROCESS_MAP[dataset_name](df)

	# Retrieve target column from map
	target_col = TARGET_MAP[dataset_name]
	y = df[target_col].values
	X = df.drop(target_col, axis=1).values

	X = X.T
	Y = y.reshape(1, -1)

	# Shuffle
	indices = np.arange(X.shape[1])
	np.random.shuffle(indices)
	X = X[:, indices]
	Y = Y[:, indices]

	# Split
	split_idx = int(0.8 * X.shape[1])
	X_train, X_val = X[:, :split_idx], X[:, split_idx:]
	Y_train, Y_val = Y[:, :split_idx], Y[:, split_idx:]

	# Normalize
	X_mean, X_std = X_train.mean(axis=1, keepdims=True), X_train.std(axis=1, keepdims=True) + 1e-8
	Y_mean, Y_std = Y_train.mean(), Y_train.std() + 1e-8
	X_train = (X_train - X_mean)/X_std
	X_val   = (X_val - X_mean)/X_std
	Y_train = (Y_train - Y_mean)/Y_std
	Y_val   = (Y_val - Y_mean)/Y_std

	model = nn.Sequential(
		nn.Linear(X_train.shape[0], 64), nn.ReLU(),
		nn.Linear(64, 64), nn.ReLU(),
		nn.Linear(64, 1)
	)
	optimizer = Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
	criterion = nn.MSELoss()
	train_data = TensorDataset(
		torch.from_numpy(X_train.T).float(),
		torch.from_numpy(Y_train.T).float()
	)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

	train_losses = []
	validation_losses = []
	for epoch in range(1, num_epochs + 1):
		model.train()
		for xb, yb in train_loader:
			optimizer.zero_grad()
			pred = model(xb).view(-1)
			loss = criterion(pred, yb.view(-1))
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())

			with torch.no_grad():
				val_preds = model(torch.from_numpy(X_val.T).float()).cpu().numpy().flatten()
				val_loss = criterion(torch.from_numpy(val_preds), torch.from_numpy(Y_val.flatten())).item()
				validation_losses.append(val_loss)

		print(f"Completed epoch {epoch}/{num_epochs}.")
  
	evaluate_model(model, X_train.T, Y_train.flatten(), "train", epoch, "training_output", dataset_name)
	evaluate_model(model, X_val.T,   Y_val.flatten(),   "val",   epoch, "training_output", dataset_name)

	EMA_losses = [train_losses[0]] if train_losses else [0.0]
	for i in range(1, len(train_losses)):
		EMA_losses.append(0.9*EMA_losses[-1] + 0.1*train_losses[i])

	EMA_val_losses = [validation_losses[0]] if validation_losses else [0.0]
	for i in range(1, len(validation_losses)):
		EMA_val_losses.append(0.9*EMA_val_losses[-1] + 0.1*validation_losses[i])

	return (EMA_losses, EMA_val_losses)

if __name__ == "__main__":
	# Example usage if run directly:
	ema_train, ema_val = run_python_evaluation("california_housing", 3, 64)
	print("EMA Train (sample):", ema_train[:5])
	print("EMA Val   (sample):", ema_val[:5])
