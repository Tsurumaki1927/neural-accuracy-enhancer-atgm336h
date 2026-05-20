import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib

# Use English labels to avoid font issues
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False

def quick_accuracy_check_english():
    """Quick check of GPS accuracy with English labels"""
    print("="*60)
    print("GPS Accuracy Quick Check")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('gps_train_dataset.csv')
    df = df.dropna(subset=['true_lon', 'true_lat'])
    print(f"   Data points: {len(df)}")
    
    # Calculate original GPS errors
    print("\n2. Calculating original GPS errors...")
    gps_lon = df['gps_lon'].values
    gps_lat = df['gps_lat'].values
    true_lon = df['true_lon'].values
    true_lat = df['true_lat'].values
    
    # Error in degrees
    lon_error_deg = np.abs(gps_lon - true_lon)
    lat_error_deg = np.abs(gps_lat - true_lat)
    
    # Convert to meters
    lon_error_m = lon_error_deg * 111320 * np.cos(np.radians(true_lat))
    lat_error_m = lat_error_deg * 110540
    total_error_m = np.sqrt(lon_error_m**2 + lat_error_m**2)
    
    print(f"\n[Original GPS Error Statistics]")
    print(f"  Mean Error:     {np.mean(total_error_m):.2f} meters")
    print(f"  Median Error:   {np.median(total_error_m):.2f} meters")
    print(f"  Std Deviation:  {np.std(total_error_m):.2f} meters")
    print(f"  Max Error:      {np.max(total_error_m):.2f} meters")
    print(f"  Min Error:      {np.min(total_error_m):.2f} meters")
    print(f"  RMSE:           {np.sqrt(np.mean(total_error_m**2)):.2f} meters")
    
    # Load model info
    print("\n3. Loading LSTM model...")
    checkpoint = torch.load('lstm_gps_model.pth', map_location='cpu', weights_only=False)
    print("   Model loaded successfully")
    print(f"   Model config: {checkpoint['model_config']}")
    
    # Generate plots
    print("\n4. Generating comparison charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error distribution histogram
    axes[0, 0].hist(total_error_m, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(total_error_m), color='blue', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(total_error_m):.2f}m')
    axes[0, 0].axvline(np.median(total_error_m), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(total_error_m):.2f}m')
    axes[0, 0].set_xlabel('Error (meters)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Original GPS Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error CDF
    sorted_errors = np.sort(total_error_m)
    cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    axes[0, 1].plot(sorted_errors, cdf, 'b-', linewidth=2)
    axes[0, 1].axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    p90 = np.percentile(total_error_m, 90)
    axes[0, 1].axvline(p90, color='red', linestyle=':', alpha=0.5, 
                       label=f'P90: {p90:.2f}m')
    axes[0, 1].set_xlabel('Error (meters)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Error Cumulative Distribution (CDF)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Longitude vs Latitude error scatter
    scatter = axes[1, 0].scatter(lon_error_m, lat_error_m, c=total_error_m, 
                                 cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=axes[1, 0], label='Total Error (meters)')
    axes[1, 0].set_xlabel('Longitude Error (meters)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Latitude Error (meters)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Longitude vs Latitude Error', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    GPS Original Measurement Error Summary
    
    Data Statistics:
    • Sample Count: {len(df):,}
    • Mean Error: {np.mean(total_error_m):.2f} m
    • Median: {np.median(total_error_m):.2f} m
    • Std Dev: {np.std(total_error_m):.2f} m
    • RMSE: {np.sqrt(np.mean(total_error_m**2)):.2f} m
    
    Percentiles:
    • P50 (Median): {np.percentile(total_error_m, 50):.2f} m
    • P75: {np.percentile(total_error_m, 75):.2f} m
    • P90: {np.percentile(total_error_m, 90):.2f} m
    • P95: {np.percentile(total_error_m, 95):.2f} m
    • P99: {np.percentile(total_error_m, 99):.2f} m
    
    Extremes:
    • Min Error: {np.min(total_error_m):.2f} m
    • Max Error: {np.max(total_error_m):.2f} m
    """
    axes[1, 1].text(0.1, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('original_gps_error_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Chart saved: original_gps_error_analysis.png")
    plt.show()
    
    print("\n" + "="*60)
    print("Note:")
    print("  To see LSTM model accuracy improvement, run:")
    print("  python show_accuracy_improvement.py")
    print("="*60)

if __name__ == "__main__":
    quick_accuracy_check_english()
