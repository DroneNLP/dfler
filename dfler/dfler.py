import argparse
import sys
import os
import json
import time
import shutil
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch
from simpletransformers.ner import NERModel

# Relative imports
from .parse import read_android_log, read_ios_log
from .generate_report import generate_report

def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            return json.load(f)
    return {}

def get_wkhtml_path(config):
    if os.name == 'nt':
        return config.get('wkhtml_path', {}).get('windows')
    else:
        return config.get('wkhtml_path', {}).get('linux')

def check_evidence(config):
    print('Evidence checking in process...\n')
    evidence_dir = config.get('source_evidence')
    output_dir = config.get('output_dir')
    
    if not evidence_dir or not os.path.exists(evidence_dir):
        print(f"Error: Evidence directory '{evidence_dir}' not found.")
        return False

    files = os.listdir(evidence_dir)
    android_logs = []
    ios_logs = []
    folders = [d for d in files if os.path.isdir(os.path.join(evidence_dir, d))]

    if not folders:
        print("No sub-folders in the evidence folder")
        return False

    for folder in folders:
        folder_path = os.path.join(evidence_dir, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if folder == 'android':
            android_logs.extend(files)
        else:
            ios_logs.extend(files)

    all_logs = android_logs + ios_logs # Just for checking if empty

    if not all_logs:
        print('No found files in the evidence folder!')
        return False

    # Save raw_list.json
    raw_list_data = [] # Structure from original code seemed to be [[android_files], [ios_files]] roughly? 
    # Original code: 
    # if(folder == 'android'): android_logs.append(files) -> files is list. so android_logs is list of lists?
    # android_logs.extend(ios_logs) -> extends list of lists?
    # checking original code:
    # android_logs = []
    # files (list) -> android_logs.append(files) -> [[file1, file2]]
    # wait, if multiple folders?
    # it seems it expects specific structure. 
    # Let's recreate original structure for compatibility with generate_report
    
    # Re-reading original code logic:
    # android_logs = []
    # for folder in folders:
    #   files = ...
    #   if folder == 'android': android_logs.append(files)
    #   else: ios_logs.append(files)
    # android_logs.extend(ios_logs)
    # json.dump(android_logs)
    
    # So it produces [[android_files...], [ios_files...]] basically.
    
    final_list = []
    if android_logs: final_list.append(android_logs)
    if ios_logs: final_list.append(ios_logs)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'raw_list.json'), 'w') as file:
        json.dump(final_list, file)

    print('Found files: \n')
    print('iOS logs: ', ios_logs)
    print("\nAndroid logs: ", android_logs)
    print('Finish checking evidence...')
    return True

def construct_timeline(config):
    print('Forensic timeline construction is in process...\n')
    evidence_dir = config.get('source_evidence')
    output_dir = config.get('output_dir')
    
    if not os.path.exists(os.path.join(output_dir, 'raw_list.json')):
        print("Error: Previous step (Check) not complete. raw_list.json missing.")
        return False

    parsed_android_dir = os.path.join(output_dir, 'parsed', 'android')
    parsed_ios_dir = os.path.join(output_dir, 'parsed', 'ios')
    os.makedirs(parsed_android_dir, exist_ok=True)
    os.makedirs(parsed_ios_dir, exist_ok=True)

    full_evidence_path = os.path.abspath(evidence_dir)

    for path, subdirs, files in os.walk(full_evidence_path):
        if "android" in path:
            for filename in files:
                if "parsed" in filename: continue
                print(f"Extracting file: {filename}")
                read_android_log(path, filename, parsed_android_dir)
        elif "ios" in path:
            for filename in files:
                if "parsed" in filename: continue
                print(f"Extracting file: {filename}")
                read_ios_log(path, filename, parsed_ios_dir)

    # Combine parsed csvs
    path_list = []
    parsed_path = os.path.join(output_dir, 'parsed')
    for path, subdirs, files in os.walk(parsed_path):
        for filename in files:
            path_list.append(os.path.join(path, filename))

    if not path_list:
        print('No parsed evidence found.')
        return False

    parent_df = pd.DataFrame()
    for path in path_list:
        try:
            child_df = pd.read_csv(path, encoding='utf-8')
            parent_df = pd.concat([parent_df, child_df])
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if parent_df.empty:
        print("Merged dataframe is empty.")
        return False

    # Standardize and sort
    try:
        # Check if columns exist
        if 'time' in parent_df.columns and 'date' in parent_df.columns:
            time_col = parent_df["time"].copy()
            parent_df["timestamp"] = parent_df["date"].str.cat(time_col, sep =" ")
            parent_df.drop(columns = ['time', 'date'], inplace=True)
            parent_df = parent_df[['timestamp', 'message']]
            parent_df['timestamp'] = pd.to_datetime(parent_df['timestamp'])
            parent_df.sort_values(by='timestamp', inplace=True)
            
            parent_df.to_csv(os.path.join(output_dir, 'forensic_timeline.csv'), index=False, encoding="utf-8")
            print('Finish constructing timeline.')
            return True
        else:
            print("Error: Required columns 'date' and 'time' not found in parsed logs.")
            return False
    except Exception as e:
        print(f"Error processing timeline: {e}")
        return False

def run_ner(config):
    print('Entity Recognition is in process...\n')
    output_dir = config.get('output_dir')
    model_dir = config.get('model_dir')
    use_cuda = config.get('use_cuda', False)

    # If model_dir is an existing local directory, ensure it contains the model bin
    # Otherwise, we assume it is a HuggingFace model ID (like dronenlp/DroNER)
    if os.path.isdir(model_dir) and not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        print(f'The local model directory exists, but pytorch_model.bin is missing at {model_dir}.')
        return False
    
    timeline_path = os.path.join(output_dir, 'forensic_timeline.csv')
    if not os.path.exists(timeline_path):
        print('The forensic timeline file is not found.')
        return False

    print("Loading model...\n")
    droner = NERModel("bert", model_dir, use_cuda=use_cuda)
    print("Model loaded.\n")

    timeline = pd.read_csv(timeline_path, encoding="utf-8")
    print('Start recognizing mentioned entities...')
    
    pred_list = []
    # Handle empty timeline
    if timeline.empty:
        print("Timeline is empty.")
    else:
        for row in tqdm(range(0, timeline.shape[0])):
            message = timeline.iloc[row, 1]
            if pd.isna(message): continue
            [entities], _ = droner.predict([str(message)])
            timestamp = timeline.iloc[row, 0]
            pred_list.append({"timestamp": timestamp, "entities": entities})

    with open(os.path.join(output_dir, 'ner_result.json'), 'w') as file:
        json.dump(pred_list, file)
    
    print('Finish recognizing mentioned entities...')
    return True

def run_report(config):
    print('Forensic report generation is in process...\n')
    output_dir = config.get('output_dir')
    
    if not os.path.exists(os.path.join(output_dir, 'ner_result.json')):
        print('The NER result is not found.')
        return False

    print('Start generating forensic report...')
    try:
        generate_report(config)
        print('Report has generated successfully.')
        return True
    except Exception as e:
        print(f'Error in generating report: {e}')
        # Check if it was wkhtmltopdf error
        wkhtml_path = config.get('wkhtml_path')
        if not wkhtml_path or not os.path.exists(wkhtml_path):
             print(f"Possible Cause: wkhtmltopdf not found at '{wkhtml_path}'. Please check configuration.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Drone Flight Log Entity Recognizer (DFLER)")
    
    # Global arguments
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--evidence", help="Evidence directory")
    parser.add_argument("--model", help="Model directory or HuggingFace model ID (default: dronenlp/DroNER)")
    
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.output: config['output_dir'] = args.output
    if args.evidence: config['source_evidence'] = args.evidence
    if args.model: config['model_dir'] = args.model
    
    # Defaults
    if 'model_dir' not in config:
        config['model_dir'] = "dronenlp/DroNER"
        
    if 'output_dir' not in config:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['output_dir'] = os.path.join('result', now)
    
    if 'use_cuda' not in config:
        config['use_cuda'] = torch.cuda.is_available()
        
    if 'app_version' not in config:
        config['app_version'] = "1.0.0" # Default
        
    # Resolve wkhtmltopdf path
    if 'wkhtml_path' not in config:
        config['wkhtml_path'] = {}
    
    # If using dictionary structure from original config
    if isinstance(config.get('wkhtml_path'), dict):
         real_wkhtml_path = get_wkhtml_path(config)
    else:
         real_wkhtml_path = config.get('wkhtml_path')

    # Update config flat parameter for generate_report compatibility
    config['wkhtml_path'] = real_wkhtml_path

    # Pipeline Execution
    if check_evidence(config):
        if construct_timeline(config):
            if run_ner(config):
                if run_report(config):
                    print("\nDFLER Pipeline completed successfully.")
                else:
                    print("\nDFLER Pipeline failed at report generation.")
            else:
                print("\nDFLER Pipeline failed at Entity Recognition.")
        else:
            print("\nDFLER Pipeline failed at Forensic timeline construction.")
    else:
        print("\nDFLER Pipeline failed at Evidence checking.")

if __name__ == "__main__":
    main()