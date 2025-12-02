
import sys
import os
import numpy as np

# Add parent directory to path to import input_data
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dsfe import config, preprocess, train_eval


if __name__ == "__main__":
    # 1. Load Data
    print("Loading data from input_data.py...")
    try:
        import input_data
        if not hasattr(input_data, 'norm_mod_data'):
            raise ImportError("norm_mod_data not found in input_data")
        data = input_data.norm_mod_data
        fs = input_data.sfreq
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Cannot proceed without real data for full evaluation.")
        sys.exit(1)

    print(f"Data Loaded. Sampling Rate: {fs} Hz")
    
    # Configuration Summary
    print(f"Configuration: FTA={config.USE_FTA}, RG={config.USE_RG}, FDCC={config.USE_FDCC}, ReliefF={config.USE_RELIEFF}, Ensemble={config.USE_ENSEMBLE}")
    
    results = []

    # 2. Iterate over all subjects/runs
    for subject in data:
        for modality in data[subject]:
            # Filter for specific modality if needed (e.g., only 'MI')
            # if modality != 'MI': continue 
            
            # Prepare list of sessions to process (either individual or merged)
            sessions_to_process = []

            if config.MERGE_RUNS:
                # --- Merge all runs for this subject/modality ---
                all_X_list = []
                all_y_list = []
                for date in data[subject][modality]:
                    for run in data[subject][modality][date]:
                        session_data = data[subject][modality][date][run]
                        # Extract X
                        if 'data' not in session_data: continue
                        d = session_data['data']
                        if not (isinstance(d, dict) and 'epoch' in d): continue
                        X_temp = d['epoch']
                        # Extract y
                        if 'label' not in session_data: continue
                        y_temp = session_data['label']
                        # Check
                        if len(X_temp) != len(y_temp): continue
                        
                        all_X_list.append(X_temp)
                        all_y_list.append(y_temp)
                
                if all_X_list:
                    print(f"  Merging {len(all_X_list)} runs for {subject} | {modality}")
                    X_combined = np.concatenate(all_X_list, axis=0)
                    y_combined = np.concatenate(all_y_list, axis=0)
                    sessions_to_process.append((X_combined, y_combined, 'merged', 'merged'))
                else:
                    print(f"  No valid data found for {subject} | {modality}")

            else:
                # --- Process each run individually ---
                for date in data[subject][modality]:
                    for run in data[subject][modality][date]:
                        session_data = data[subject][modality][date][run]
                        # Extract X
                        if 'data' not in session_data:
                            print(f"  [Skipping] {subject} {modality} {date} {run}: No data key.")
                            continue
                        d = session_data['data']
                        if isinstance(d, dict) and 'epoch' in d:
                            X_temp = d['epoch']
                        else:
                            print(f"  [Skipping] {subject} {modality} {date} {run}: No 'epoch'.")
                            continue
                        # Extract y
                        if 'label' in session_data:
                            y_temp = session_data['label']
                        else:
                            print(f"  [Skipping] {subject} {modality} {date} {run}: No label.")
                            continue
                        # Check
                        if len(X_temp) != len(y_temp):
                            print(f"  [Skipping] {subject} {modality} {date} {run}: Mismatch X/y.")
                            continue
                            
                        sessions_to_process.append((X_temp, y_temp, date, run))

            # Execute Processing & Evaluation
            for X, y, date, run in sessions_to_process:
                session_info = f"{subject} | {modality} | {date} | Run {run}"
                print(f"\nProcessing: {session_info}")
                
                # Preprocess
                X_prep = preprocess.preprocess_pipeline(X, fs)
                
                # Slice based on config (EPOCH_TMIN to EPOCH_TMAX)
                start_sample = int(config.EPOCH_TMIN * fs)
                end_sample = int(config.EPOCH_TMAX * fs)
                
                # Validate indices
                if start_sample < 0: start_sample = 0
                if end_sample > X_prep.shape[2]: end_sample = X_prep.shape[2]
                
                if start_sample >= end_sample:
                    print(f"  [Error] Invalid time window: {config.EPOCH_TMIN}s - {config.EPOCH_TMAX}s")
                    continue

                X_dsfe = X_prep[:, :, start_sample:end_sample]
                print(f"  -> Time Segment: {config.EPOCH_TMIN}s - {config.EPOCH_TMAX}s (Samples: {start_sample}-{end_sample}, Shape: {X_dsfe.shape})")
                    
                # Evaluate
                try:
                    acc = train_eval.evaluate_session(X_dsfe, y, fs)
                    print(f"  -> Accuracy: {acc:.4f}")
                    results.append({
                        'subject': subject,
                        'modality': modality,
                        'date': date,
                        'run': run,
                        'accuracy': acc
                    })
                except Exception as e:
                    print(f"  [Error] Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()

    # 3. Summary
    if not results:
        print("\nNo results collected.")
    else:
        print("\n" + "="*40)
        print("Evaluation Summary")
        print("="*40)
        
        # Convert to DataFrame for easier aggregation if pandas is available
        try:
            import pandas as pd
            df_res = pd.DataFrame(results)
            
            # Per Subject
            print("\nAverage Accuracy per Subject:")
            print(df_res.groupby('subject')['accuracy'].mean())
            
            # Per Modality
            print("\nAverage Accuracy per Modality:")
            print(df_res.groupby('modality')['accuracy'].mean())
            
            # Overall
            print(f"\nOverall Average Accuracy: {df_res['accuracy'].mean():.4f}")
            
            # Save results
            base_filename = 'dsfe_results.csv'
            output_filename = base_filename
            if os.path.exists(output_filename):
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f'dsfe_results_{timestamp}.csv'
            
            df_res.to_csv(output_filename, index=False)
            print(f"\nDetailed results saved to '{output_filename}'")
            
        except ImportError:
            # Fallback if pandas not found
            avg_acc = np.mean([r['accuracy'] for r in results])
            print(f"Overall Average Accuracy: {avg_acc:.4f}")

