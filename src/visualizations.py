import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_report(candidate_results, trainer):
    # Ensure the output directory exists
    output_dir = "../results/evo-2"
    os.makedirs(output_dir, exist_ok=True)

    # Create a summary table: highest predicted efficiency per gene.
    summary_table = candidate_results.groupby("Gene")["Predicted_Efficiency"].max().reset_index()
    summary_table.columns = ["Gene", "Max Predicted Efficiency"]
    summary_csv_path = os.path.join(output_dir, "summary_table.csv")
    summary_table.to_csv(summary_csv_path, index=False)
    print(f"Summary table saved to {summary_csv_path}")

    # Plot distribution of predicted efficiencies.
    plt.figure()
    candidate_results["Predicted_Efficiency"].hist(bins=30)
    plt.title("Distribution of Predicted siRNA Efficiency")
    plt.xlabel("Predicted Efficiency")
    plt.ylabel("Frequency")
    distribution_path = os.path.join(output_dir, "predicted_efficiency_distribution.png")
    plt.savefig(distribution_path)
    plt.close()
    print(f"Efficiency distribution plot saved to {distribution_path}")

    # Plot training loss over epochs.
    epochs = []
    losses = []
    for log in trainer.state.log_history:
        if "loss" in log:
            epochs.append(log.get("epoch", len(epochs) + 1))
            losses.append(log["loss"])
    if epochs and losses:
        plt.figure()
        plt.plot(epochs, losses, marker="o")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        loss_plot_path = os.path.join(output_dir, "training_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Training loss plot saved to {loss_plot_path}")
    
    # Plot evaluation metrics (loss and MSE) over epochs.
    eval_epochs = []
    eval_losses = []
    eval_mses = []
    for log in trainer.state.log_history:
        if "eval_loss" in log and "epoch" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])
            eval_mses.append(log.get("eval_mse", None))
    if eval_epochs:
        plt.figure()
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval Loss")
        # Plot Eval MSE if available.
        if any(m is not None for m in eval_mses):
            plt.plot(eval_epochs, eval_mses, marker="o", label="Eval MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Evaluation Metric")
        plt.title("Evaluation Metrics Over Epochs")
        plt.legend()
        eval_plot_path = os.path.join(output_dir, "evaluation_metrics.png")
        plt.savefig(eval_plot_path)
        plt.close()
        print(f"Evaluation metrics plot saved to {eval_plot_path}")

    # Write a summary report text file.
    report_lines = []
    report_lines.append("siRNA Candidate Prediction Summary Report")
    report_lines.append("========================================\n")
    report_lines.append("Summary Table (Max Predicted Efficiency per Gene):")
    report_lines.append(summary_table.to_string(index=False))
    report_lines.append("\nMetrics:")
    if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
        report_lines.append(f"Best validation metric: {trainer.state.best_metric}")
    else:
        report_lines.append("Best validation metric not available.")
    report_content = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Summary report saved to {report_path}")