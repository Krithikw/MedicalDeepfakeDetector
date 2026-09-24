import numpy as np
import os

pairs = [
    (201, 316, 362),
    (319, 125, 323)
]

for s, x, y in pairs:

    exp1_path = os.path.join(
        "outputs", "roi_dataset", "train",
        f"EXP1_7507_{s}_{x}_{y}.npy"
    )

    exp2_path = os.path.join(
        "outputs", "roi_dataset", "test",
        f"EXP2_7507_{s}_{x}_{y}.npy"
    )

    a = np.load(exp1_path).astype(np.float32)
    b = np.load(exp2_path).astype(np.float32)

    diff = np.abs(a - b)

    print(f"\nSlice {s}")
    print(f"  Shape          : {a.shape}")
    print(f"  EXP1 mean      : {a.mean():.6f}")
    print(f"  EXP2 mean      : {b.mean():.6f}")
    print(f"  EXP1 std       : {a.std():.6f}")
    print(f"  EXP2 std       : {b.std():.6f}")
    print(f"  MAE            : {diff.mean():.6f}")
    print(f"  RMSE           : {np.sqrt(np.mean((a - b) ** 2)):.6f}")
    print(f"  Max difference : {diff.max():.6f}")

    print("  Meaningful differences:")
    for threshold in [0.001, 0.005, 0.010, 0.020, 0.050]:
        count = (diff > threshold).sum()
        percentage = 100 * count / diff.size
        print(
            f"    > {threshold:.3f}: "
            f"{count:5d} pixels ({percentage:6.2f}%)"
        )
