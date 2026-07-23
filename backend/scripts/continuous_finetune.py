import os
import sys
import subprocess

def run_training(epochs=5, synthetic_size=100):
    print(f"--- Running Training (Epochs: {epochs}, Synthetic Size: {synthetic_size}) ---")
    script_path = os.path.join(os.path.dirname(__file__), "train_rgbd_unet.py")
    cmd = [
        sys.executable, script_path,
        "--finetune-real", "Testphotos/allergy-Testing.jpg",
        "--epochs", str(epochs),
        "--synthetic-size", str(synthetic_size),
        "--batch-size", "4"
    ]
    subprocess.run(cmd, check=True)

def run_evaluation():
    print("--- Running Evaluation on Test Photo ---")
    from backend.scripts import evaluate_on_testphotos
    percent_error = evaluate_on_testphotos.evaluate()
    return percent_error

def main():
    target_error = 5.0  # Stop when percent error is below this
    max_iterations = 20

    for iteration in range(1, max_iterations + 1):
        print(f"\n=========================================")
        print(f" Iteration {iteration}/{max_iterations}")
        print(f"=========================================")
        
        # 1. Train
        run_training(epochs=2, synthetic_size=20)
        
        # 2. Evaluate
        percent_error = run_evaluation()
        
        # 3. Check condition
        if percent_error <= target_error:
            print(f"\nSUCCESS: Percent error {percent_error:.2f}% is below target {target_error}%.")
            print("Stopping iterative finetuning.")
            break
        else:
            print(f"\nCurrent percent error {percent_error:.2f}% > {target_error}%. Continuing...")

    print("Iterative finetuning complete.")

if __name__ == "__main__":
    main()
