import numpy as np
from scipy.stats import wilcoxon

def evaluate_significance(baseline_losses, variant_losses):
    """
    Performs a Wilcoxon signed-rank test.
    Used for non-parametric, paired samples (same seeds used for both models).
    """
    assert len(baseline_losses) == len(variant_losses), "Mismatched seed counts"
    
    baseline = np.array(baseline_losses)
    variant = np.array(variant_losses)
    
    # Calculate Mean and Std
    b_mean, b_std = np.mean(baseline), np.std(baseline)
    v_mean, v_std = np.mean(variant), np.std(variant)
    
    print(f"Baseline (Raissi): {b_mean:.4e} ± {b_std:.4e}")
    print(f"New Variant:       {v_mean:.4e} ± {v_std:.4e}")
    
    # FIX: Safety check to handle identical inputs before throwing a zero-difference exception
    if np.array_equal(baseline, variant) or np.allclose(baseline - variant, 0):
        print("\nWilcoxon Signed-Rank Test:")
        print("[-] Result is identical. Total difference between runs is exactly zero.")
        print("❌ Result is NOT significant. The models perform identically.")
        return

    # Wilcoxon Test
    statistic, p_value = wilcoxon(baseline, variant)
    
    # FIX: Repaired the double backslash raw print newline string escape
    print("\nWilcoxon Signed-Rank Test:")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Result is STATISTICALLY SIGNIFICANT (p < 0.05). The variant is fundamentally different.")
        if v_mean < b_mean:
            print("🏆 The variant is definitively BETTER.")
    else:
        print("❌ Result is NOT significant. The difference is likely just random noise.")

# Example Usage:
# evaluate_significance(
#     baseline_losses=[0.0012, 0.0015, 0.0011, 0.0014, 0.0013], # Seed 1-5 results
#     variant_losses= [0.0008, 0.0009, 0.0007, 0.0008, 0.0008]  # Seed 1-5 results
# )
