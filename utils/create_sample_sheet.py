import os
import csv
import argparse


def scan_directory(base_dir, output_csv):
    # List to store data rows
    data_rows = []

    # Walk through the directory
    for root, _, files in os.walk(base_dir):
        for file in files:
            # Check if the file is relevant (e.g., ends with .nd2)
            if file.endswith(".nd2") or file.endswith(".h5"):
                file_path = os.path.join(root, file)

                # Extract patient_id and check if 'MLH1' is in the path
                patient_id = file.split("_")[0]  # First element when split by '_'
                # fixed = 'MLH1' in file  # Check if 'MLH1' is in the file name
                if "PANCK" in file or "panck" in file:
                    fixed = True
                else:
                    fixed = False

                # Append the row
                data_rows.append(
                    {
                        "patient_id": patient_id,
                        "image": file_path,
                        "fixed": str(
                            fixed
                        ).lower(),  # Convert boolean to lowercase string
                    }
                )

    # Write the data to a CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["patient_id", "image", "fixed"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_rows)

    print(f"CSV file written to {output_csv}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Scan a directory and generate a CSV file with patient_id, image paths, and fixed status."
    )
    parser.add_argument("--base_dir", required=True, help="Base directory to scan")
    parser.add_argument(
        "--output_csv", default="sample_sheet.csv", help="Path to the output CSV file"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    scan_directory(args.base_dir, args.output_csv)


if __name__ == "__main__":
    main()
