import csv
import re

import pandas as pd


def load_templates(template_csv):
    templates = []
    with open(template_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_id = row["EventId"]
            template = row["EventTemplate"].strip()
            parts = template.split("<*>")
            escaped_parts = [re.escape(part) for part in parts]
            pattern = "^" + "(.*?)".join(escaped_parts) + "$"
            compiled = re.compile(pattern)
            templates.append((event_id, compiled))
    return templates


def process_log(log_file, templates, output_csv):
    with open(log_file, "r") as infile, open(output_csv, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Id", "Label", "Features"])
        for line in infile:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) < 2:
                continue  # Skip invalid lines

            # Extract Id from the second word
            event_id = words[1]

            # Determine Label based on the first word
            label = "Success" if words[0] == "-" else "Fail"

            # Extract message starting from the 10th word (index 9)
            message = " ".join(words[9:]) if len(words) >= 10 else ""

            # Find matching template
            features = ""
            for ev_id, pattern in templates:
                if pattern.match(message):
                    features = ev_id
                    break  # Use the first matching template

            writer.writerow([event_id, label, features])


if __name__ == "__main__":
    # Update these filenames according to your actual files
    df = pd.read_csv("BGL.csv", index_col=0)
    merged_df = (
        df.groupby("Id")["Label", "Feature"].apply(lambda x: x[0], list).reset_index()
    )
    merged_df = df.groupby("Id")["Feature"].apply(list).reset_index()
    grouped_df = df.groupby("Id").agg({"Label": "last", "Features": list}).reset_index()

    # Rename the column for clarity
    merged_df.rename(columns={"Feature": "Features"}, inplace=True)
# templates = load_templates("./data/BGL/BGL_templates.csv")
# process_log("./data/BGL/BGL.log", templates, "BGL.csv")
