import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_heatmap(df, output_path, title):
    """
    Generates and saves a heatmap from the provided DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing 'Qmax', 'YT', and 'Avg_Best_MSE'.
        output_path (str): Path to save the generated heatmap PNG.
        title (str): Title of the heatmap.
    """
    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = df.pivot(index='YT', columns='Qmax', values='Avg_Best_MSE')

    # Sort the index and columns for better visualization
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)

    plt.figure(figsize=(10, 8))
    sns.set(style="white")

    # Create the heatmap with annotations
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={'label': 'Avg Best MSE'},
        linewidths=.5,
        linecolor='gray'
    )

    # Set the title and labels
    ax.set_title(title, fontsize=16, pad=16)
    ax.set_xlabel('Qmax', fontsize=14)
    ax.set_ylabel('YT', fontsize=14)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to '{output_path}'.")

def process_csv_file(csv_file, output_dir):
    """
    Processes a single CSV file to generate a heatmap.

    Parameters:
        csv_file (str): Path to the CSV file.
        output_dir (str): Directory to save the heatmap PNG.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check if required columns exist
        required_columns = {'Qmax', 'YT', 'Avg_Best_MSE'}
        if not required_columns.issubset(df.columns):
            print(f"Skipping '{csv_file}': Missing required columns.")
            return

        # Drop duplicates in case there are multiple entries for the same Qmax and YT
        df_unique = df.drop_duplicates(subset=['Qmax', 'YT'])

        # Handle missing values
        if df_unique['Avg_Best_MSE'].isnull().any():
            print(f"Warning: Missing Avg_Best_MSE values in '{csv_file}'. These will be shown as empty in the heatmap.")

        # Generate the heatmap title based on the filename
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        title = f"Heatmap of Avg_Best_MSE for {base_name}"

        # Define the output path for the heatmap PNG
        png_filename = f"{base_name}.png"
        output_path = os.path.join(output_dir, png_filename)

        # Plot and save the heatmap
        plot_heatmap(df_unique, output_path, title)

    except Exception as e:
        print(f"An error occurred while processing '{csv_file}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps from compiled CSV files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing CSV files.')
    parser.add_argument('--output_dir', type=str, default='plots/', help='Path to the output directory to save heatmap PNG files.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure input_dir exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create output_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Search for CSV files in the input directory
    csv_pattern = os.path.join(input_dir, '*.csv')
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in '{input_dir}'. Processing...")

    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing '{csv_file}'...")
        process_csv_file(csv_file, output_dir)

    print("All heatmaps have been generated.")

if __name__ == "__main__":
    main()
