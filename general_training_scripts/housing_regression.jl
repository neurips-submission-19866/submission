#!/usr/bin/env julia

# ============================================================================
# Example training script for California Housing Prices
# with Dataset struct + train/val split
# ============================================================================
# This version replaces the scatter plot with a 2D KDE plot.
# ============================================================================

# -- Load libraries and local code -------------------------------------------
include("../lib/utils.jl")        # Utility functions
include("../lib/datasets.jl")    # Contains the Dataset struct & constructor
include("../lib/factor_graph.jl") # Factor graph functionality

using Pkg
using CSV
using DataFrames
using Random

using Statistics
# using StatsBase          # for quantile
using KernelDensity      # for kde (supports 1D or 2D, etc.)
using Plots              # for plotting
using FilePathsBase: mkpath  # or Base.Filesystem.mkpath

# ---------------- 2D KDE plot function --------------------------------------
function plot_2d_kde(
    real_values,
    predicted_values;
    set_name = "train",
    epoch = 1,
    output_dir = "training_output"
)
    # Flatten to 1D vectors just in case they are matrices:
    x = vec(real_values)
    y = vec(predicted_values)

    # Compute 2D KDE
    kd = kde((x, y))

    # Create grid of points for plotting
    x_grid = collect(range(minimum(x), maximum(x), length=256))
    y_grid = collect(range(minimum(y), maximum(y), length=256))

    # Create a heatmap of the 2D density
    plt = heatmap(
        x_grid,          # Use collected grid instead of kd.x[1]
        y_grid,          # Use collected grid instead of kd.x[2]
        kd.density,
        title  = "Real vs. Predicted (2D KDE)\n($(set_name) set, epoch $epoch)",
        xlabel = "Real Value",
        ylabel = "Predicted Value",
        fill   = true,
        color  = :viridis
    )

    mkpath(output_dir)
    filename = joinpath(output_dir, "2d_kde_$(set_name)_epoch$(epoch).png")
    savefig(plt, filename)
    println("Saved 2D KDE plot to: $filename")
end

# ---------------- 1D KDE plot function --------------------------------------
function plot_kde_distribution(
    data;
    set_name = "train",
    epoch = 1,
    output_dir = "training_output",
    plot_title = "KDE Plot",
    x_label = "Values",
    y_label = "Density",
    overlay_normal = false
)
    kd = kde(vec(data))
    x_vals = range(minimum(kd.x), maximum(kd.x), length=250)

    # Optional standard normal PDF
    norm_pdf(x) = @inline (1 / sqrt(2π)) * exp(-0.5 * x^2)

    mkpath(output_dir)

    plt = plot(
        kd.x,
        kd.density,
        label    = "KDE",
        linewidth = 2,
        title    = plot_title,
        xlabel   = x_label,
        ylabel   = y_label
    )

    if overlay_normal
        plot!(
            x_vals,
            [norm_pdf(x) for x in x_vals],
            label     = "Standard Normal PDF",
            linewidth = 2,
            linestyle = :dash
        )
    end

    # For the filename, replace spaces in the title with underscores and append epoch
    filename = joinpath(output_dir, replace(plot_title, " " => "_") * "_epoch$(epoch).png")
    savefig(plt, filename)
    println("Saved KDE plot to: $filename")
end

"""
    evaluate_model(fg, X, Y; set_name="train", epoch=1, output_dir="training_output")

Evaluate the model by predicting on (X, Y), computing MSE, RMSE, variance percentiles,
and checking calibration by comparing (pred_mean - y) / pred_std to a standard
normal distribution. A kernel density plot of that distribution vs. the standard
normal is saved to `output_dir`. Also plots the residual distribution, real values,
predicted values, and a 2D KDE of (real, predicted).

# Arguments
- `fg::FactorGraph`: your trained (or in-training) factor graph
- `X::AbstractArray`: features
- `Y::AbstractArray`: ground truth targets
- `set_name::String`: e.g. "train" or "val"
- `epoch::Int`: current epoch, used for labeling output
- `output_dir::String`: folder to store plots (created if it doesn't exist)

"""
function evaluate_model(
    fg,
    X,
    Y;
    set_name = "train",
    epoch = 1,
    output_dir = "training_output"
)
    preds = predict(fg, X)
    pred_means = mean.(preds)
    pred_vars  = variance.(preds)

    mse  = Statistics.mean((Y .- pred_means).^2)
    rmse = sqrt(mse)

    ptiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    var_percentiles = quantile(sqrt.(pred_vars), ptiles)

    calibration_values = (pred_means .- Y) ./ sqrt.(pred_vars)
    residuals = pred_means .- Y

    within_range(range) = count(abs.(calibration_values) .<= range) / length(calibration_values)
    println("Percentage of calibration values within:")
    println("[-1, 1]: ", within_range(1))
    println("[-2, 2]: ", within_range(2))
    println("[-3, 3]: ", within_range(3))

    # 1D KDE: calibration vs normal
    plot_kde_distribution(
        calibration_values;
        set_name = set_name,
        epoch = epoch,
        output_dir = output_dir,
        plot_title = "Calibration vs Standard Normal\n($(set_name) set, epoch $epoch)",
        x_label = "z = (μ̂ - y) / σ̂",
        overlay_normal = true
    )

    # 1D KDE: residual distribution
    plot_kde_distribution(
        residuals;
        set_name = set_name,
        epoch = epoch,
        output_dir = output_dir,
        plot_title = "Residuals\n($(set_name) set, epoch $epoch)",
        x_label = "residual = μ̂ - y",
        overlay_normal = false
    )

    # 1D KDE: real values
    plot_kde_distribution(
        Y;
        set_name = set_name,
        epoch = epoch,
        output_dir = output_dir,
        plot_title = "Real Values\n($(set_name) set, epoch $epoch)",
        x_label = "Real Value",
        overlay_normal = false
    )

    # 1D KDE: predicted values
    plot_kde_distribution(
        pred_means;
        set_name = set_name,
        epoch = epoch,
        output_dir = output_dir,
        plot_title = "Predicted Values\n($(set_name) set, epoch $epoch)",
        x_label = "Predicted Value",
        overlay_normal = false
    )

    # 2D KDE for Real vs. Predicted
    plot_2d_kde(Y, pred_means; set_name=set_name, epoch=epoch, output_dir=output_dir)

    println("=== Evaluation on $(set_name) set (epoch $epoch) ===")
    println("MSE  = $mse")
    println("RMSE = $rmse")
    println("Predicted variance percentiles at $(ptiles) => $var_percentiles\n")
end

# -- Set random seed for reproducibility -------------------------------------
Random.seed!(98)

# -- Load the housing dataset ------------------------------------------------
df = CSV.read("datasets/housing.csv", DataFrame)

total_missing = sum(ismissing.(Matrix(df)))
println("Total missing values: $total_missing")

# Replace missing values with 0 (column-by-column)
for c in names(df)
    replace!(df[!, c], missing => 0)
end
disallowmissing!(df)

# -- Replace ocean_proximity categories with float values --------------------
replace!(df."ocean_proximity", 
         "ISLAND"    => "0.0",
         "NEAR BAY"  => "1.0",
         "NEAR OCEAN"=> "2.0",
         "<1H OCEAN" => "3.0",
         "INLAND"    => "4.0")
df."ocean_proximity" = parse.(Float64, df."ocean_proximity")

# -- Extract features (X) and target (Y) --------------------------------------
# Adjust the column name for the target as needed; here we assume
# the column for the house value is named :median_house_value

y = df."median_house_value"                             # Vector
X = select(df, Not(:median_house_value)) |> Matrix      # Matrix

# Typically factor graphs in the provided codebase expect data in the format
# (features..., batch). We'll transpose so shape = (#features, #samples).
X = X'               # shape: (num_features, num_samples)
Y = reshape(y, 1, :) # shape: (1, num_samples)

# -- Build our dataset with train/val split ----------------------------------
# The Dataset constructor (from datasets.jl) will split into X_train, Y_train, X_val, Y_val
# By default it uses 80% of the data for training, 20% for validation.
X = hcat(X)

println("Type of X: ", typeof(X))
println("Size of X: ", size(X))
println("Type of Y: ", typeof(Y))
println("Size of Y: ", size(Y))

dset = Dataset(X, Y)  # You can pass train_perc=0.8 if you want to customize
dset = normalize_X!(dset)
dset = normalize_Y!(dset)

println("Training set has size: ", size(dset.X_train), " => X_train")
println("Training set has size: ", size(dset.Y_train), " => Y_train")
println("Validation set has size: ", size(dset.X_val), " => X_val")
println("Validation set has size: ", size(dset.Y_val), " => Y_val")

# -- Generate desired model architecture -------------------------------------
function generate_architecture(input_shape, action_space_dimensions, regression_basic_architecture)
    return [
        (input_shape,),
        (:Linear, 64),
        (:LeakyReLU, 0.1),
        (:Linear, 64),
        (:LeakyReLU, 0.1),
        (:Linear, action_space_dimensions),
        (:Regression, regression_basic_architecture^2),
    ]
end

# For a simple single-output regression, set action_space_dimensions = 1
# and choose a small integer for regression_basic_architecture (like 1)
input_shape               = size(dset.X_train, 1)  # number of features
action_space_dimensions   = 1
regression_basic_arch_val = 0.4

architecture = generate_architecture(input_shape,
                                     action_space_dimensions,
                                     regression_basic_arch_val)

# -- Create the factor graph model -------------------------------------------
batch_size = 64
fg = create_factor_graph(architecture, batch_size)

# -- Create Trainer and train -----------------------------------------------
trainer = Trainer(fg, dset.X_train, dset.Y_train)

num_epochs       = 10
num_training_its = 3

training_losses = Vector{Float64}()
validation_losses = Vector{Float64}()

println("Starting training on California Housing dataset...")
train(trainer; num_epochs=num_epochs, num_training_its=num_training_its, silent=false, training_losses=training_losses, validation_losses=validation_losses, validation_X=dset.X_val, validation_Y=dset.Y_val)

# Evaluate on training data
evaluate_model(fg, dset.X_train, dset.Y_train; set_name="train", epoch=0)

# Evaluate on validation data
evaluate_model(fg, dset.X_val, dset.Y_val; set_name="val", epoch=0)
println("Training finished.")

EMA_losses = [training_losses[1]]
for i in 2:length(training_losses)
    push!(EMA_losses, 0.9 * EMA_losses[end] + 0.1 * training_losses[i])
end

EMA_val_losses = [validation_losses[1]]
for i in 2:length(validation_losses)
    push!(EMA_val_losses, 0.9 * EMA_val_losses[end] + 0.1 * validation_losses[i])
end

# Clamp EMA losses to [0, 2]
EMA_losses = clamp.(EMA_losses, 0, 2)
EMA_val_losses = clamp.(EMA_val_losses, 0, 2)

# Create line plot for EMA losses.
plot(EMA_losses, label="EMA Train Loss", xlabel="Iteration", ylabel="Loss", title="Training vs Validation EMA Loss")
plot!(EMA_val_losses, label="EMA Val Loss")
savefig("training_output/ema_loss_plot.png")
