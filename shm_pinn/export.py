"""
Export module for saving results to timestamped folders.
"""
import os
import shutil
from datetime import datetime
import pandas as pd
from dataclasses import asdict
import matplotlib.pyplot as plt


def create_results_folder():
    """
    Create a timestamped folder for results.
    Returns the folder path and timestamp.
    """
    # Create main results directory if it doesn't exist
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(base_dir, timestamp)
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"Results will be saved to: {results_folder}")
    return results_folder, timestamp


def save_config_to_excel(cfg, filepath):
    """
    Save config variables and values to Excel.
    
    Args:
        cfg: Config object
        filepath: Full path to save the Excel file
    """
    # Convert config to dictionary
    config_dict = asdict(cfg)
    
    # Create DataFrame
    df = pd.DataFrame(list(config_dict.items()), columns=['Variable', 'Value'])
    
    # Save to Excel
    df.to_excel(filepath, index=False, sheet_name='Config')
    print(f"Config saved to: {filepath}")


def save_stats_to_excel(stats, filepath):
    """
    Save training statistics to Excel.
    
    Args:
        stats: Dictionary containing training statistics
        filepath: Full path to save the Excel file
    """
    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: Summary statistics
        summary_data = {
            'Metric': ['Final Total Loss', 'Final Physics Loss', 'Final IC Loss', 
                      'Final Data Loss', 'Training Time (seconds)'],
            'Value': [
                stats['final_total'],
                stats['final_phys'],
                stats['final_ic'],
                stats['final_data'],
                stats['train_time_sec']
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Loss curves (epoch by epoch)
        curves_data = {
            'Epoch': list(range(1, len(stats['curve_total']) + 1)),
            'Total Loss': stats['curve_total'],
            'Physics Loss': stats['curve_phys'],
            'IC Loss': stats['curve_ic'],
            'Data Loss': stats['curve_data']
        }
        df_curves = pd.DataFrame(curves_data)
        df_curves.to_excel(writer, sheet_name='Loss Curves', index=False)
    
    print(f"Stats saved to: {filepath}")


def save_plot(fig, filepath):
    """
    Save matplotlib figure as PNG.
    
    Args:
        fig: matplotlib figure object
        filepath: Full path to save the PNG file
    """
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")


def save_code_snapshot(timestamp, results_folder):
    """
    Save copies of all Python files needed to run main.py with timestamp suffix.
    Creates a 'code_snapshot' subfolder to keep results organized.
    
    Args:
        timestamp: Timestamp string (YYYYMMDD_HHMMSS)
        results_folder: Path to the results folder
    """
    # Create code_snapshot subfolder
    snapshot_folder = os.path.join(results_folder, f'code_snapshot_{timestamp}')
    os.makedirs(snapshot_folder, exist_ok=True)
    
    # Get the shm_pinn directory path
    source_dir = os.path.dirname(__file__)
    
    # List of Python files to copy (all the module files)
    python_files = [
        '__init__.py',
        'main.py',
        'config.py',
        'data.py',
        'model.py',
        'train.py',
        'evaluate.py',
        'export.py',
        'apply.py'
    ]
    
    copied_files = []
    failed_files = []
    
    for filename in python_files:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(snapshot_folder, filename)
        
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_files.append(filename)
            else:
                failed_files.append(f"{filename} (not found)")
        except Exception as e:
            failed_files.append(f"{filename} ({str(e)})")
    
    print(f"Code snapshot saved to: {snapshot_folder}")
    print(f"  ✓ Copied {len(copied_files)} files: {', '.join(copied_files)}")
    if failed_files:
        print(f"  ⚠ Could not copy: {', '.join(failed_files)}")


def export_results(cfg, stats, fig):
    """
    Main export function to save all results.
    
    Args:
        cfg: Config object
        stats: Dictionary containing training statistics
        fig: matplotlib figure object
    
    Returns:
        str: Path to the results folder
    """
    # Create timestamped folder
    results_folder, timestamp = create_results_folder()
    
    # Save config to Excel
    config_path = os.path.join(results_folder, 'config.xlsx')
    save_config_to_excel(cfg, config_path)
    
    # Save snapshot of all code files with timestamp
    save_code_snapshot(timestamp, results_folder)
    
    # Save stats to Excel
    stats_path = os.path.join(results_folder, 'stats.xlsx')
    save_stats_to_excel(stats, stats_path)
    
    # Save plot as PNG
    plot_path = os.path.join(results_folder, 'plot.png')
    save_plot(fig, plot_path)
    
    print(f"\n{'='*60}")
    print(f"All results exported successfully to:")
    print(f"{results_folder}")
    print(f"{'='*60}\n")
    
    return results_folder
