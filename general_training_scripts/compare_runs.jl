#!/usr/bin/env julia

ENV["PYTHON"] = "/Users/janniklasgroeneveld/.local/share/virtualenvs/iclr2025_7302-nNcKoKJ_/bin/python"
ENV["PYTHONPATH"] = "/Users/janniklasgroeneveld/repositories/iclr2025_7302"

using Plots
using PyCall

# We assume julia_eval.jl and python_eval.py are in the same folder.
# Adjust paths as needed.
include("julia_eval.jl")
py_eval = pyimport("general_training_scripts.python_eval")

function create_plots(dataset_name, ema_train_julia, ema_val_julia, ema_train_python, ema_val_python)
    plt = plot(ema_train_julia, label="BNN Train", linewidth=2, color=1)
    plot!(plt, ema_val_julia, label="BNN Val", linewidth=2, color=2)
    plot!(plt, ema_train_python, label="PyTorch Train", linewidth=2, color=3)
    plot!(plt, ema_val_python, label="PyTorch Val", linewidth=2, color=4)

    min_ema_train_julia = minimum(ema_train_julia)
    min_ema_val_julia = minimum(ema_val_julia)
    min_ema_train_python = minimum(ema_train_python)
    min_ema_val_python = minimum(ema_val_python)

    plot!(plt, [1, length(ema_train_julia)], [min_ema_train_julia, min_ema_train_julia],
          label="", linestyle=:dot, color=1)
    plot!(plt, [1, length(ema_val_julia)], [min_ema_val_julia, min_ema_val_julia],
          label="", linestyle=:dot, color=2)
    plot!(plt, [1, length(ema_train_python)], [min_ema_train_python, min_ema_train_python],
          label="", linestyle=:dot, color=3)
    plot!(plt, [1, length(ema_val_python)], [min_ema_val_python, min_ema_val_python],
          label="", linestyle=:dot, color=4)

    xlabel!("Iteration")
    ylabel!("Loss")
    title!("Comparison on $(dataset_name) Dataset")
    savefig(plt, "training_output/$(dataset_name)_common_comparison_plot.pdf")
    println("Saved combined plot to common_comparison_plot.pdf")
end

"""
Collects performance into the passed-in `performance_table`.
"""
function run_and_plot_for_dataset(dataset_name, num_epochs, batch_size, performance_table)
    (ema_train_julia, ema_val_julia) = JuliaEval.run_julia_evaluation(dataset_name, num_epochs, batch_size)
    (ema_train_python, ema_val_python) = py_eval.run_python_evaluation(dataset_name, num_epochs, batch_size)
    
    # Create plots with raw (loss) data
    create_plots(dataset_name, ema_train_julia, ema_val_julia, ema_train_python, ema_val_python)

    # Take the root of all losses to plot the RMSEs
    # Append _RMSE to the dataset name for clarity
    dataset_name_rmse = "$(dataset_name)_RMSE"
    create_plots(dataset_name_rmse, sqrt.(ema_train_julia), sqrt.(ema_val_julia),
                 sqrt.(ema_train_python), sqrt.(ema_val_python))

    # Compute minimum RMSE values
    min_rmse_bnn_train = minimum(sqrt.(ema_train_julia))
    min_rmse_bnn_val = minimum(sqrt.(ema_val_julia))
    min_rmse_pytorch_train = minimum(sqrt.(ema_train_python))
    min_rmse_pytorch_val = minimum(sqrt.(ema_val_python))

    # Compute ratio (in %) of BNN val to PyTorch val
    percentage = min_rmse_bnn_val / min_rmse_pytorch_val * 100

    # Store the results in the performance table
    push!(performance_table, (
        dataset=dataset_name,
        bnn_train=min_rmse_bnn_train,
        bnn_val=min_rmse_bnn_val,
        pytorch_train=min_rmse_pytorch_train,
        pytorch_val=min_rmse_pytorch_val,
        percentage=percentage
    ))

    println("Finished evaluation for $(dataset_name) dataset.")
end

"""
Converts the performance table into a LaTeX tabular format and prints it out.
"""
function performance_table_to_latex(performance_table)
    println("\\begin{table}[ht]")
    println("\\centering")
    println("\\begin{tabular}{l|cccc|c}")
    println("Dataset & BNN Train & BNN Val & PyTorch Train & PyTorch Val & BNN Val / PyTorch Val (\\%)\\\\")
    println("\\hline")
    for row in performance_table
        println(
            "\$(\\mathrm{" * row.dataset * "})\$ & "
            * string(round(row.bnn_train, digits=4)) * " & "
            * string(round(row.bnn_val, digits=4)) * " & "
            * string(round(row.pytorch_train, digits=4)) * " & "
            * string(round(row.pytorch_val, digits=4)) * " & "
            * string(round(row.percentage, digits=2)) * "\\%\\\\"
        )
    end
    println("\\end{tabular}")
    println("\\caption{Comparison of minimum RMSE for BNN (Julia) and PyTorch approaches. The data was obtained by running the respective training scripts for 500 epochs and measuring the root mean squared error on training and validation splits.}")
    println("\\label{tab:performance_comparison}")
    println("\\end{table}")
end

function main()
    num_epochs = 5000
    batch_size = 256

    # This table will store our performance rows
    performance_table = []

    for dataset_name in ["california_housing"] #  ["abalone","wine_quality", "california_housing", "bike_sharing", "forest_fires", "heart_failure", "real_estate_taiwan"]
        run_and_plot_for_dataset(dataset_name, num_epochs, batch_size, performance_table)
    end

    # After all runs, output the performance table as LaTeX
    performance_table_to_latex(performance_table)
end

main()
