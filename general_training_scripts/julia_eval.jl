module JuliaEval

using CSV
using DataFrames
using Statistics
using Random
using Plots
using KernelDensity
using FilePathsBase: mkpath

include("../lib/utils.jl")        # Utility functions
include("../lib/datasets.jl")    # Contains the Dataset struct & constructor
include("../lib/factor_graph.jl") # Factor graph functionality

# Simple dictionary for dataset-specific preprocessing.
const PREPROCESS_MAP = Dict{String, Function}(
    "california_housing" => df -> begin
        replace!(df[!, :ocean_proximity], "ISLAND" => "0.0", "NEAR BAY" => "1.0", "NEAR OCEAN" => "2.0", "<1H OCEAN" => "3.0", "INLAND" => "4.0")
        df[!, :ocean_proximity] = parse.(Float64, df[!, :ocean_proximity])
        for c in names(df)
            replace!(df[!, c], missing => 0)
        end
        disallowmissing!(df)
        return df
    end,
    # In case of wine_quality, please convert the target to a Float64
    "wine_quality" => df -> begin
        df[!, :target] = Float64.(df[!, :target])
        return df
    end,
    "automobile" => df -> begin
        df[!, :target] = Float64.(df[!, :target])
        return df
    end,
    "abalone" => df -> begin
        df[!, :target] = Float64.(df[!, :target])
        replace!(df[!, :Sex], "M" => "0", "F" => "1", "I" => "2")
        df[!, :Sex] = parse.(Float64, df[!, :Sex])
        return df
    end,
    "bike_sharing" => df -> begin
        # Drop the dteday column
        select!(df, Not(:dteday))
        # Convert the target to a Float64
        df[!, :target] = Float64.(df[!, :target])
        return df
    end,
    "forest_fires" => df -> begin
        df[!, :target] = Float64.(df[!, :target])
        replace!(df[!, :month], "jan" => "1", "feb" => "2", "mar" => "3", "apr" => "4", "may" => "5", "jun" => "6",
            "jul" => "7", "aug" => "8", "sep" => "9", "oct" => "10", "nov" => "11", "dec" => "12")
        replace!(df[!, :day], "mon" => "1", "tue" => "2", "wed" => "3", "thu" => "4", "fri" => "5", "sat" => "6", "sun" => "7")
        df[!, :month] = parse.(Float64, df[!, :month])
        df[!, :day] = parse.(Float64, df[!, :day])
        return df
    end,
    "heart_failure" => df -> begin
        df[!, :target] = Float64.(df[!, :target])
        return df
    end,
    "real_estate_taiwan" => df -> begin
        df[!, :target] = Float64.(df[!, :target])
        return df
    end,
)

# Maps dataset names to their CSV paths
const DATASET_PATH_MAP = Dict{String,String}(
    "california_housing" => "datasets/housing.csv",
    "wine_quality"       => "datasets/wine_quality.csv",
    "automobile"         => "datasets/automobile.csv",
    "abalone"            => "datasets/abalone.csv",
    "bike_sharing"       => "datasets/bike_sharing.csv",
    "forest_fires"       => "datasets/forest_fires.csv",
    "heart_failure"      => "datasets/heart_failure.csv",
    "real_estate_taiwan" => "datasets/real_estate_taiwan.csv",
)

# Maps dataset names to their target column
const TARGET_MAP = Dict{String,Symbol}(
    "california_housing" => :median_house_value,
    "wine_quality"       => :target,
    "automobile"         => :target,
    "abalone"            => :target,
    "bike_sharing"       => :target,
    "forest_fires"       => :target,
    "heart_failure"      => :target,
    "real_estate_taiwan" => :target,
)

# ---------------- Utility: 2D KDE plot --------------------------------------
function plot_2d_kde(real_values, predicted_values; dataset_name="unknown_dataset", set_name="train", epoch=1, output_dir="training_output")
    x = vec(real_values)
    y = vec(predicted_values)
    kd = kde((x, y))
    x_grid = collect(range(minimum(x), maximum(x), length=256))
    y_grid = collect(range(minimum(y), maximum(y), length=256))
    plt = heatmap(x_grid, y_grid, kd.density,
        title  = "Real vs. Predicted (2D KDE)\n($(set_name) set, epoch $epoch)",
        xlabel = "Real Value",
        ylabel = "Predicted Value",
        fill   = true,
        color  = :viridis)
    mkpath(output_dir)
    filename = joinpath(output_dir, "$(dataset_name)_BNN_2d_kde_$(set_name)_epoch$(epoch).png")
    savefig(plt, filename)
end

function plot_absolute_diff_2d_kde(abs_diff_vec; dataset_name="unknown_dataset", set_name="abs_diff", epoch=1, output_dir="training_output")
    # Kick out all values if either x is greater than 1 or y is greater than 5
    # abs_diff_vec = [Pair(p.first, p.second) for p in abs_diff_vec if p.first <= 0.5 && p.second <= 0.005]
    abs_diff_vec = abs_diff_vec[10000:end]
    x = [p.first for p in abs_diff_vec]
    y = [p.second for p in abs_diff_vec]
    # Assert all ys greater than zero and not nan or inf
    for i in 1:length(y)
        if y[i] <= 0 || isnan(y[i]) || isinf(y[i])
            y[i] = 1e-8
        end
    end
    plot_2d_kde(x, y; dataset_name=dataset_name, set_name=set_name, epoch=epoch, output_dir=output_dir)
end

# ---------------- Utility: 1D KDE plot --------------------------------------
function plot_kde_distribution(data; dataset_name="unknown_dataset", set_name = "train", epoch = 1, output_dir = "training_output",
    plot_title = "KDE Plot", x_label = "Values", y_label = "Density", overlay_normal = false)
    kd = kde(vec(data))
    x_vals = range(minimum(kd.x), maximum(kd.x), length=250)
    mkpath(output_dir)
    plt = plot(kd.x, kd.density, label="KDE", linewidth=2,
        title=plot_title, xlabel=x_label, ylabel=y_label)
    if overlay_normal
        norm_pdf(x) = @inline (1 / sqrt(2π)) * exp(-0.5 * x^2)
        plot!(x_vals, [norm_pdf(x) for x in x_vals], label="Standard Normal PDF",
            linewidth=2, linestyle=:dash)
    end
    filename = joinpath(output_dir, "$(dataset_name)_BNN_" * replace(plot_title, " " => "_") * "_epoch$(epoch).png")
    savefig(plt, filename)
end

# ---------------- Evaluation function ---------------------------------------
function evaluate_model(fg, X, Y; dataset_name="unknown_dataset", set_name="train", epoch=1, output_dir="training_output")
    preds = predict(fg, X)
    pred_means = mean.(preds)
    pred_vars  = variance.(preds)
    mse  = Statistics.mean((Y .- pred_means).^2)
    rmse = sqrt(mse)
    ptiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    var_percentiles = quantile(sqrt.(pred_vars), ptiles)
    calibration_values = (pred_means .- Y) ./ sqrt.(pred_vars .+ 1e-8)  # avoid /0
    residuals = pred_means .- Y
    within_range(range) = count(abs.(calibration_values) .<= range) / length(calibration_values)
    # 1D KDEs
    plot_kde_distribution(calibration_values; dataset_name=dataset_name, set_name=set_name, epoch=epoch,
        output_dir=output_dir, plot_title="Calibration vs Standard Normal\n($(set_name) set, epoch $epoch)",
        x_label="z = (μ̂ - y) / σ̂", overlay_normal=true)
    plot_kde_distribution(residuals; dataset_name=dataset_name, set_name=set_name, epoch=epoch, output_dir=output_dir,
        plot_title="Residuals\n($(set_name) set, epoch $epoch)", x_label="residual = μ̂ - y")
    plot_kde_distribution(Y; dataset_name=dataset_name, set_name=set_name, epoch=epoch, output_dir=output_dir,
        plot_title="Real Values\n($(set_name) set, epoch $epoch)", x_label="Real Value")
    plot_kde_distribution(pred_means; dataset_name=dataset_name, set_name=set_name, epoch=epoch, output_dir=output_dir,
        plot_title="Predicted Values\n($(set_name) set, epoch $epoch)", x_label="Predicted Value")
    plot_2d_kde(Y, pred_means; dataset_name=dataset_name, set_name=set_name, epoch=epoch, output_dir=output_dir)
    println("Percentage of calibration values within:")
    println("[-1, 1]: ", within_range(1))
    println("[-2, 2]: ", within_range(2))
    println("[-3, 3]: ", within_range(3))
    println("=== Evaluation on $(set_name) set (epoch $epoch) ===")
    println("MSE  = $mse")
    println("RMSE = $rmse")
    println("Predicted variance percentiles at $(ptiles) => $var_percentiles\n")
end

# ---------------- Main function to run everything ---------------------------
function run_julia_evaluation(dataset_name::String, num_epochs::Int, batch_size::Int)
    Random.seed!(98)

    # Use DATASET_PATH_MAP to pick the CSV file
    df = CSV.read(DATASET_PATH_MAP[dataset_name], DataFrame)
    df = PREPROCESS_MAP[dataset_name](df)

    # Use TARGET_MAP to pick the correct target column
    target_col = TARGET_MAP[dataset_name]
    y = df[!, target_col]

    X = select(df, Not(target_col)) |> Matrix
    X = X'
    X = hcat(X)
    Y = reshape(y, 1, :)
    dset = Dataset(X, Y)
    dset = normalize_X!(dset)
    dset = normalize_Y!(dset)

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

    input_shape               = size(dset.X_train, 1)
    action_space_dimensions   = 1
    regression_basic_arch_val = 0.4
    architecture = generate_architecture(input_shape, action_space_dimensions, regression_basic_arch_val)

    fg = create_factor_graph(architecture, batch_size)
    trainer = Trainer(fg, dset.X_train, dset.Y_train)

    training_losses = Float64[]
    validation_losses = Float64[]

    absolute_difference_vector = train(trainer; num_epochs=num_epochs, num_training_its=3, silent=false,
          training_losses=training_losses, validation_losses=validation_losses,
          validation_X=dset.X_val, validation_Y=dset.Y_val)

    # plot_absolute_diff_2d_kde(absolute_difference_vector; dataset_name=dataset_name)
    # println("Absolute difference vector: ", absolute_difference_vector)
    println("The lenght of the absolute difference vector is: ", length(absolute_difference_vector))

    evaluate_model(fg, dset.X_train, dset.Y_train; dataset_name=dataset_name, set_name="train", epoch=0)
    evaluate_model(fg, dset.X_val, dset.Y_val; dataset_name=dataset_name, set_name="val", epoch=0)

    EMA_losses = [training_losses[1]]
    for i in 2:length(training_losses)
        push!(EMA_losses, 0.9 * EMA_losses[end] + 0.1 * training_losses[i])
    end

    EMA_val_losses = [validation_losses[1]]
    for i in 2:length(validation_losses)
        push!(EMA_val_losses, 0.9 * EMA_val_losses[end] + 0.1 * validation_losses[i])
    end

    return (EMA_losses, EMA_val_losses)
end

end # module JuliaEval
