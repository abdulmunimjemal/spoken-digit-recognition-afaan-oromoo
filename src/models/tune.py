import subprocess
import itertools

def run_tuning():
    learning_rates = [0.001, 0.0005]
    augmentations = [
        {'time': 30, 'freq': 15}, # Default (Strong)
        {'time': 15, 'freq': 10}  # Light
    ]
    epochs = 60
    model_type = 'deeper'
    
    configs = list(itertools.product(learning_rates, augmentations))
    
    print(f"Running {len(configs)} experiments...")
    
    for lr, aug in configs:
        time_mask = aug['time']
        freq_mask = aug['freq']
        
        print(f"\n--- Starting Run: LR={lr}, TimeMask={time_mask}, FreqMask={freq_mask} ---")
        
        cmd = [
            "python", "run.py", "train",
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--model_type", model_type,
            "--time_mask", str(time_mask),
            "--freq_mask", str(freq_mask)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("--- Run Completed ---\n")
        except subprocess.CalledProcessError as e:
            print(f"!!! Run Failed with error: {e} !!!")

if __name__ == "__main__":
    run_tuning()
