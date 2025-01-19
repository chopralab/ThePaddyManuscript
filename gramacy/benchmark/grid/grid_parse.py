import os
import glob
import pandas as pd
import re
import argparse

def parse_log_file(filepath):
    """
    Parses a single log file to extract Run, Best MSE, and Elapsed time.

    Parameters:
        filepath (str): Path to the log file.

    Returns:
        List of dictionaries with keys: filename, Run, best_MSE, Time
    """
    pattern = re.compile(
        r"Run\s+(?P<run>\d+):\s+Best MSE:\s+(?P<mse>[\d.]+),\s+Elapsed time:\s+(?P<time>[\d.]+)\s+seconds"
    )
    data = []
    filename = os.path.basename(filepath)
    
    with open(filepath, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                run = int(match.group('run'))
                mse = float(match.group('mse'))
                time = float(match.group('time'))
                data.append({
                    'filename': filename,
                    'Run': run,
                    'best_MSE': mse,
                    'Time': time
                })
    return data

def Qmax_YT_extract(filename):
    """
    Extracts Qmax and YT values from the filename.

    Parameters:
        filename (str): The name of the file.

    Returns:
        Tuple containing Qmax (int) and YT (int). Returns (None, None) if extraction fails.
    """
    # Example filename:
    # PADDY_Qmax20_YT10_20241213150301_QMAX20_YT10_PADDYgenerational_GAUSIANscaled_summary.log
    # We need to extract Qmax20_YT10 from the first part after 'PADDY_'

    pattern = re.compile(r"PADDY_Qmax(?P<Qmax>\d+)_YT(?P<YT>\d+)_")
    match = pattern.search(filename)
    if match:
        Qmax = int(match.group('Qmax'))
        YT = int(match.group('YT'))
        return Qmax, YT
    else:
        # If pattern not found, return None
        return None, None

def process_logs(input_dir, output_dir, output_filename='compiled_results.csv'):
    """
    Processes all .summary.log files in the input directory, computes statistics,
    extracts Qmax and YT, sorts the DataFrame, and saves the compiled data.

    Parameters:
        input_dir (str): Directory to search for .summary.log files.
        output_dir (str): Directory to save the compiled CSV.
        output_filename (str): Name of the output CSV file.

    Returns:
        pandas.DataFrame: Compiled dataframe containing all extracted data and statistics.
    """
    # Ensure input_dir exists
    if not os.path.isdir(input_dir):
        raise NotADirectoryError("Input directory '{}' does not exist.".format(input_dir))

    # Create output_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Search for files ending with 'summary.log'
    search_pattern = os.path.join(input_dir, '*summary.log')
    log_files = glob.glob(search_pattern)

    if not log_files:
        print("No files ending with 'summary.log' found in '{}'.".format(input_dir))
        return pd.DataFrame(columns=[
            'filename', 'Run', 'best_MSE', 'Time',
            'Avg_Best_MSE', 'Std_Best_MSE', 'Avg_Time', 'Std_Time',
            'Qmax', 'YT'
        ])

    all_data = []
    for log_file in log_files:
        file_data = parse_log_file(log_file)
        all_data.extend(file_data)
        print("Processed file: {}, extracted {} runs.".format(log_file, len(file_data)))

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=[
        'filename', 'Run', 'best_MSE', 'Time'
    ])

    # Compute statistics for each filename
    stats_df = df.groupby('filename').agg(
        Avg_Best_MSE=pd.NamedAgg(column='best_MSE', aggfunc='mean'),
        Std_Best_MSE=pd.NamedAgg(column='best_MSE', aggfunc='std'),
        Avg_Time=pd.NamedAgg(column='Time', aggfunc='mean'),
        Std_Time=pd.NamedAgg(column='Time', aggfunc='std')
    ).reset_index()

    # Merge statistics back into the main DataFrame
    df = pd.merge(df, stats_df, on='filename', how='left')

    # Optionally, round the statistical columns for better readability
    df['Avg_Best_MSE'] = df['Avg_Best_MSE'].round(6)
    df['Std_Best_MSE'] = df['Std_Best_MSE'].round(6)
    df['Avg_Time'] = df['Avg_Time'].round(2)
    df['Std_Time'] = df['Std_Time'].round(2)

    # Extract Qmax and YT from filename
    df[['Qmax', 'YT']] = df['filename'].apply(
        lambda x: pd.Series(Qmax_YT_extract(x))
    )

    # Sort the DataFrame by Qmax, YT, Run in ascending order
    df_sorted = df.sort_values(by=['Qmax', 'YT', 'Run'], ascending=[True, True, True])

    # Reset index after sorting
    df_sorted.reset_index(drop=True, inplace=True)

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df_sorted.to_csv(output_path, index=False)
    print("Compiled and sorted data with statistics saved to '{}'.".format(output_path))

    return df_sorted

def main():
    parser = argparse.ArgumentParser(description="Parse summary.log files and compile results with statistics into a CSV.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing .summary.log files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save the CSV file.')
    parser.add_argument('--output_filename', type=str, default='compiled_results.csv', help='Name of the output CSV file.')

    args = parser.parse_args()

    try:
        df = process_logs(args.input_dir, args.output_dir, args.output_filename)
        print("\nSample of the compiled and sorted DataFrame:")
        print(df.head())  # Display first few rows
    except Exception as e:
        print("An error occurred: {}".format(e))

if __name__ == "__main__":
    main()
