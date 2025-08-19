from src.utils import set_seed
import os

set_seed(0)

print("🚀 Running MNIST CNN Project Pipeline...")

# 1️⃣ Train Model
os.system("python src/train.py")

# 2️⃣ Evaluate on Test Set
os.system("python src/evaluate.py")

print("✅ Full CNN pipeline completed!")