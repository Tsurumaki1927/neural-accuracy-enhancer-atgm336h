import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Use English fonts to avoid font issues
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'

class LSTMModel(nn.Module):
    """LSTM Neural Network Model"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_model_and_data(model_path='lstm_gps_model.pth', data_path='gps_train_dataset.csv'):
    """Load model and data"""
    print("Loading model and data...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    seq_length = checkpoint['seq_length']
    
    model = LSTMModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feature_scaler = checkpoint['feature_scaler']
    target_scaler = checkpoint['target_scaler']
    
    # Load data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['true_lon', 'true_lat'])
    
    features = ['gps_lon', 'gps_lat', 'gps_sat_num']
    targets = ['true_lon', 'true_lat']
    
    X = df[features].values
    y = df[targets].values
    
    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y)
    
    print(f"Data loaded: {len(df)} records")
    
    return model, feature_scaler, target_scaler, X_scaled, y_scaled, seq_length, df

def create_sequences(X, y, seq_length=10):
    """Create time series sequences"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

def calculate_errors(pred, true):
    """Calculate various error metrics"""
    # Longitude and latitude errors (degrees)
    lon_error_deg = np.abs(pred[:, 0] - true[:, 0])
    lat_error_deg = np.abs(pred[:, 1] - true[:, 1])
    
    # Convert to meters
    lon_error_m = lon_error_deg * 111320 * np.cos(np.radians(true[:, 1]))
    lat_error_m = lat_error_deg * 110540
    
    # Total error (Euclidean distance)
    total_error_m = np.sqrt(lon_error_m**2 + lat_error_m**2)
    
    return {
        'lon_error_deg': lon_error_deg,
        'lat_error_deg': lat_error_deg,
        'lon_error_m': lon_error_m,
        'lat_error_m': lat_error_m,
        'total_error_m': total_error_m
    }

def comprehensive_evaluation(model, X_test, y_test, target_scaler, feature_scaler, seq_length, df):
    """Comprehensive model performance evaluation"""
    print("\n" + "="*70)
    print("Starting Comprehensive Accuracy Improvement Evaluation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prediction
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(device)
        pred_scaled = model(test_tensor).cpu().numpy()
    
    # Inverse transform
    pred = target_scaler.inverse_transform(pred_scaled)
    true = target_scaler.inverse_transform(y_test)
    
    # Calculate LSTM prediction errors
    lstm_errors = calculate_errors(pred, true)
    
    # Calculate original GPS errors
    gps_indices = range(seq_length - 1, len(df))
    if len(gps_indices) >= len(y_test):
        sample_indices = list(gps_indices)[-len(y_test):]
        original_gps_data = df.iloc[sample_indices][['gps_lon', 'gps_lat']].values
        
        orig_errors = calculate_errors(original_gps_data, true)
    else:
        orig_errors = lstm_errors  # fallback
    
    # Print detailed statistics
    print("\n[Original GPS Measurement Error Statistics]")
    print("-" * 70)
    print(f"Mean Longitude Error: {np.mean(orig_errors['lon_error_deg'])*1e6:.2f} micro-degrees ({np.mean(orig_errors['lon_error_m']):.2f} m)")
    print(f"Mean Latitude Error:  {np.mean(orig_errors['lat_error_deg'])*1e6:.2f} micro-degrees ({np.mean(orig_errors['lat_error_m']):.2f} m)")
    print(f"Mean Total Error:     {np.mean(orig_errors['total_error_m']):.2f} m")
    print(f"Median Error:         {np.median(orig_errors['total_error_m']):.2f} m")
    print(f"Std Deviation:        {np.std(orig_errors['total_error_m']):.2f} m")
    print(f"Max Error:            {np.max(orig_errors['total_error_m']):.2f} m")
    print(f"Min Error:            {np.min(orig_errors['total_error_m']):.2f} m")
    print(f"RMSE:                 {np.sqrt(np.mean(orig_errors['total_error_m']**2)):.2f} m")
    
    print("\n[LSTM Model Prediction Error Statistics]")
    print("-" * 70)
    print(f"Mean Longitude Error: {np.mean(lstm_errors['lon_error_deg'])*1e6:.2f} micro-degrees ({np.mean(lstm_errors['lon_error_m']):.2f} m)")
    print(f"Mean Latitude Error:  {np.mean(lstm_errors['lat_error_deg'])*1e6:.2f} micro-degrees ({np.mean(lstm_errors['lat_error_m']):.2f} m)")
    print(f"Mean Total Error:     {np.mean(lstm_errors['total_error_m']):.2f} m")
    print(f"Median Error:         {np.median(lstm_errors['total_error_m']):.2f} m")
    print(f"Std Deviation:        {np.std(lstm_errors['total_error_m']):.2f} m")
    print(f"Max Error:            {np.max(lstm_errors['total_error_m']):.2f} m")
    print(f"Min Error:            {np.min(lstm_errors['total_error_m']):.2f} m")
    print(f"RMSE:                 {np.sqrt(np.mean(lstm_errors['total_error_m']**2)):.2f} m")
    
    # Calculate accuracy improvement
    improvement_mean = (1 - np.mean(lstm_errors['total_error_m']) / np.mean(orig_errors['total_error_m'])) * 100
    improvement_median = (1 - np.median(lstm_errors['total_error_m']) / np.median(orig_errors['total_error_m'])) * 100
    improvement_rmse = (1 - np.sqrt(np.mean(lstm_errors['total_error_m']**2)) / np.sqrt(np.mean(orig_errors['total_error_m']**2))) * 100
    
    print("\n[Accuracy Improvement Results]")
    print("-" * 70)
    print(f"Mean Error Reduction:      {improvement_mean:.2f}%")
    print(f"Median Error Reduction:    {improvement_median:.2f}%")
    print(f"RMSE Reduction:            {improvement_rmse:.2f}%")
    print(f"Stability Improvement:     {(1 - np.std(lstm_errors['total_error_m'])/np.std(orig_errors['total_error_m']))*100:.2f}%")
    print("="*70)
    
    return pred, true, orig_errors, lstm_errors, improvement_mean

def plot_comprehensive_results(true, orig_errors, lstm_errors, improvement):
    """Plot comprehensive accuracy comparison"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Error distribution comparison histogram
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.linspace(0, max(np.percentile(orig_errors['total_error_m'], 95), 
                               np.percentile(lstm_errors['total_error_m'], 95)), 50)
    ax1.hist(orig_errors['total_error_m'], bins=bins, alpha=0.5, label='Original GPS', 
             color='red', edgecolor='black', linewidth=0.5)
    ax1.hist(lstm_errors['total_error_m'], bins=bins, alpha=0.5, label='LSTM Prediction', 
             color='blue', edgecolor='black', linewidth=0.5)
    ax1.axvline(np.mean(orig_errors['total_error_m']), color='red', linestyle='--', 
                linewidth=2, label=f'Original Mean: {np.mean(orig_errors["total_error_m"]):.2f}m')
    ax1.axvline(np.mean(lstm_errors['total_error_m']), color='blue', linestyle='--', 
                linewidth=2, label=f'LSTM Mean: {np.mean(lstm_errors["total_error_m"]):.2f}m')
    ax1.set_xlabel('Error (meters)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'GPS Error Distribution Comparison (Accuracy Improvement: {improvement:.2f}%)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Distribution Function (CDF)
    ax2 = fig.add_subplot(gs[1, 0])
    sorted_orig = np.sort(orig_errors['total_error_m'])
    sorted_lstm = np.sort(lstm_errors['total_error_m'])
    cdf_orig = np.arange(1, len(sorted_orig)+1) / len(sorted_orig)
    cdf_lstm = np.arange(1, len(sorted_lstm)+1) / len(sorted_lstm)
    
    ax2.plot(sorted_orig, cdf_orig, 'r-', linewidth=2, label='Original GPS')
    ax2.plot(sorted_lstm, cdf_lstm, 'b-', linewidth=2, label='LSTM Prediction')
    ax2.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(np.percentile(orig_errors['total_error_m'], 90), color='red', 
                linestyle=':', alpha=0.5, label=f'Original P90: {np.percentile(orig_errors["total_error_m"], 90):.2f}m')
    ax2.axvline(np.percentile(lstm_errors['total_error_m'], 90), color='blue', 
                linestyle=':', alpha=0.5, label=f'LSTM P90: {np.percentile(lstm_errors["total_error_m"], 90):.2f}m')
    ax2.set_xlabel('Error (meters)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Error Cumulative Distribution (CDF)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = fig.add_subplot(gs[1, 1])
    data_to_plot = [orig_errors['total_error_m'], lstm_errors['total_error_m']]
    bp = ax3.boxplot(data_to_plot, labels=['Original GPS', 'LSTM Prediction'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax3.set_ylabel('Error (meters)', fontsize=11, fontweight='bold')
    ax3.set_title('Error Box Plot Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (mean_val, median_val) in enumerate(zip(
        [np.mean(orig_errors['total_error_m']), np.mean(lstm_errors['total_error_m'])],
        [np.median(orig_errors['total_error_m']), np.median(lstm_errors['total_error_m'])]
    )):
        ax3.text(i+1, mean_val + 0.5, f'Mean:{mean_val:.1f}m', 
                ha='center', fontsize=9, fontweight='bold')
    
    # 4. Scatter plot: Prediction vs True
    ax4 = fig.add_subplot(gs[1, 2])
    scatter = ax4.scatter(true[:, 0], true[:, 1], c=lstm_errors['total_error_m'], 
                         cmap='viridis', alpha=0.6, s=20, edgecolors='none')
    plt.colorbar(scatter, ax=ax4, label='Error (meters)')
    ax4.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax4.set_title('Prediction Error Spatial Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Error time series (first 500 points)
    ax5 = fig.add_subplot(gs[2, 0])
    sample_size = min(500, len(orig_errors['total_error_m']))
    ax5.plot(range(sample_size), orig_errors['total_error_m'][:sample_size], 
            'r-', alpha=0.6, linewidth=1, label='Original GPS')
    ax5.plot(range(sample_size), lstm_errors['total_error_m'][:sample_size], 
            'b-', alpha=0.6, linewidth=1, label='LSTM Prediction')
    ax5.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Error (meters)', fontsize=11, fontweight='bold')
    ax5.set_title('Error Time Series Comparison (First 500 Samples)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Error improvement percentage distribution
    ax6 = fig.add_subplot(gs[2, 1])
    improvements = (1 - lstm_errors['total_error_m'] / orig_errors['total_error_m']) * 100
    ax6.hist(improvements, bins=40, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2, 
                label=f'Avg Improvement: {np.mean(improvements):.1f}%')
    ax6.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Accuracy Improvement (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Per-Sample Accuracy Improvement Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Key metrics summary table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    metrics_text = f"""
    Accuracy Improvement Summary
    
    Original GPS:
    - Mean Error: {np.mean(orig_errors['total_error_m']):.2f} m
    - Median: {np.median(orig_errors['total_error_m']):.2f} m
    - RMSE: {np.sqrt(np.mean(orig_errors['total_error_m']**2)):.2f} m
    - P90: {np.percentile(orig_errors['total_error_m'], 90):.2f} m
    
    LSTM Model:
    - Mean Error: {np.mean(lstm_errors['total_error_m']):.2f} m
    - Median: {np.median(lstm_errors['total_error_m']):.2f} m
    - RMSE: {np.sqrt(np.mean(lstm_errors['total_error_m']**2)):.2f} m
    - P90: {np.percentile(lstm_errors['total_error_m'], 90):.2f} m
    
    Improvement:
    ✓ Avg Improvement: {improvement:.2f}%
    ✓ Stability: {(1-np.std(lstm_errors['total_error_m'])/np.std(orig_errors['total_error_m']))*100:.1f}%
    """
    
    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig('comprehensive_accuracy_improvement.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive evaluation chart saved: comprehensive_accuracy_improvement.png")
    plt.show()

def main():
    print("="*70)
    print("GPS Accuracy Improvement Comprehensive Evaluation System")
    print("="*70)
    
    # Load model and data
    model, feature_scaler, target_scaler, X_scaled, y_scaled, seq_length, df = \
        load_model_and_data()
    
    # Create sequences
    print(f"\nCreating time series sequences (window size={seq_length})...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    print(f"Sequence data shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Use all data for evaluation (or use test set)
    print("\nEvaluating with full dataset...")
    X_test = X_seq
    y_test = y_seq
    
    # Comprehensive evaluation
    pred, true, orig_errors, lstm_errors, improvement = \
        comprehensive_evaluation(model, X_test, y_test, target_scaler, feature_scaler, seq_length, df)
    
    # Generate comprehensive result charts
    print("\nGenerating visualization charts...")
    plot_comprehensive_results(true, orig_errors, lstm_errors, improvement)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
