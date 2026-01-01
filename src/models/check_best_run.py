import mlflow
from mlflow.tracking import MlflowClient

def check_best():
    try:
        experiment = mlflow.get_experiment_by_name("Spoken Digit Recognition")
        if experiment is None:
            print("Experiment not found.")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_accuracy DESC"]
        )
        
        if len(runs) == 0:
            print("No runs found.")
            return

        best_run = runs.iloc[0]
        print(f"--- Best Run Details ---")
        print(f"Run ID: {best_run.run_id}")
        print(f"Parameters:")
        print(f"  LR: {best_run['params.learning_rate']}")
        print(f"  Model: {best_run['params.model_type']}")
        print(f"  Epochs: {best_run['params.epochs']}")
        print(f"  Time Mask: {best_run.get('params.time_mask', 'N/A')}")
        print(f"  Freq Mask: {best_run.get('params.freq_mask', 'N/A')}")
        print(f"Metrics:")
        print(f"  Val Accuracy: {best_run['metrics.val_accuracy']:.2f}%")
        print(f"  Val F1: {best_run['metrics.val_f1']:.4f}")
        print(f"  Val Precision: {best_run['metrics.val_precision']:.4f}")
        print(f"  Val Recall: {best_run['metrics.val_recall']:.4f}")
        
    except Exception as e:
        print(f"Error checking runs: {e}")

if __name__ == "__main__":
    check_best()
