#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def plot_2d_kde(real_values, predicted_values, set_name="train", epoch=1, output_dir="training_output"):
    x, y = real_values.flatten(), predicted_values.flatten()
    kd = gaussian_kde(np.vstack([x, y]))
    xi = np.linspace(x.min(), x.max(), 256)
    yi = np.linspace(y.min(), y.max(), 256)
    X, Y = np.meshgrid(xi, yi)
    Z = kd(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    plt.figure()
    plt.imshow(Z, origin="lower", aspect="auto", extent=[x.min(), x.max(), y.min(), y.max()], cmap="viridis")
    plt.title(f"Real vs. Predicted (2D KDE)\n({set_name} set, epoch {epoch})")
    plt.xlabel("Real Value")
    plt.ylabel("Predicted Value")
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"2d_kde_{set_name}_epoch{epoch}.png")
    plt.savefig(fname)
    plt.close()

def plot_kde_distribution(data, set_name="train", epoch=1, output_dir="training_output", plot_title="KDE Plot", x_label="Values", overlay_normal=False):
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
        plt.plot(x_vals, [norm_pdf(v) for v in x_vals], label="Standard Normal PDF", linewidth=2, linestyle="--")
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, plot_title.replace(" ","_") + f"_epoch{epoch}.png")
    plt.savefig(fname)
    plt.close()

def evaluate_model(model, X, Y, set_name="train", epoch=1, output_dir="training_output"):
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
    plot_kde_distribution(calibration_values, set_name, epoch, output_dir, f"Calibration vs Standard Normal\n({set_name} set, epoch {epoch})", "z = (μ̂ - y) / σ̂", True)
    plot_kde_distribution(residuals, set_name, epoch, output_dir, f"Residuals\n({set_name} set, epoch {epoch})", "residual = μ̂ - y", False)
    plot_kde_distribution(Y, set_name, epoch, output_dir, f"Real Values\n({set_name} set, epoch {epoch})", "Real Value", False)
    plot_kde_distribution(preds, set_name, epoch, output_dir, f"Predicted Values\n({set_name} set, epoch {epoch})", "Predicted Value", False)
    plot_2d_kde(Y, preds, set_name, epoch, output_dir)
    print(f"=== Evaluation on {set_name} set (epoch {epoch}) ===")
    print(f"MSE  = {mse}")
    print(f"RMSE = {rmse}\n")

def main():
    np.random.seed(98)
    torch.manual_seed(98)
    df = pd.read_csv("datasets/housing.csv")
    df = df.fillna(0)
    df["ocean_proximity"] = df["ocean_proximity"].replace({"ISLAND":"0.0","NEAR BAY":"1.0","NEAR OCEAN":"2.0","<1H OCEAN":"3.0","INLAND":"4.0"}).astype(float)
    y = df["median_house_value"].values
    X = df.drop("median_house_value", axis=1).values
    X = X.T
    Y = y.reshape(1, -1)
    X_train, X_val = X[:, :int(0.8*X.shape[1])], X[:, int(0.8*X.shape[1]):]
    Y_train, Y_val = Y[:, :int(0.8*Y.shape[1])], Y[:, int(0.8*Y.shape[1]):]
    X_mean, X_std = X_train.mean(axis=1, keepdims=True), X_train.std(axis=1, keepdims=True) + 1e-8
    Y_mean, Y_std = Y_train.mean(), Y_train.std() + 1e-8
    X_train = (X_train - X_mean)/X_std
    X_val = (X_val - X_mean)/X_std
    Y_train = (Y_train - Y_mean)/Y_std
    Y_val = (Y_val - Y_mean)/Y_std
    model = nn.Sequential(nn.Linear(X_train.shape[0],64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,1))
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    batch_size = 64
    train_data = TensorDataset(torch.from_numpy(X_train.T).float(), torch.from_numpy(Y_train.T).float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    num_epochs = 3
    num_training_its = 3
    print("Starting training on California Housing dataset...")
    train_losses = []
    validation_losses = []
    for epoch in range(1, num_epochs+1):
        model.train()
        for _ in range(num_training_its):
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
                    
        evaluate_model(model, X_train.T, Y_train.flatten(), "train", epoch)
        evaluate_model(model, X_val.T, Y_val.flatten(), "val", epoch)
        print(f"Completed epoch {epoch}/{num_epochs}.")
    print("Training finished.")
    
    EMA_losses = [train_losses[0]]
    for i in range(1, len(train_losses)):
        EMA_losses.append(0.9*EMA_losses[-1] + 0.1*train_losses[i])
        
    EMA_losses_val = [validation_losses[0]]
    for i in range(1, len(validation_losses)):
        EMA_losses_val.append(0.9*EMA_losses_val[-1] + 0.1*validation_losses[i])
    
    # Plot line plot with training losses
    plt.figure()
    plt.plot(EMA_losses, label="Training Loss", linewidth=2)
    plt.plot(EMA_losses_val, label="Validation Loss", linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_output/training_loss.png")
    plt.close()

if __name__ == "__main__":
    main()
