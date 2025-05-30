import sys
import logging
import pandas as pd

logging.basicConfig(
    level=logging.WARNING, format="[%(asctime)s][%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

import csv
import re


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


templates = load_templates("data/OpenStack/OpenStack_2k.log_templates.csv")

# Regular expression pattern to parse each log line
pattern = re.compile(
    r"^(?P<Logrecord>\S+)\s+"
    r"(?P<Date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<Time>\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<Pid>\d+)\s+"
    r"(?P<Level>[A-Z]+)\s+"
    r"(?P<Component>[^\[]+)\s+"
    r"\[(?P<ADDR>[^\]]+)\]\s+"
    r"(?P<Content>.+)$"
)
# Match and extract fields

# Read the log file and write to CSV
with (
    open("data/OpenStack/openstack_normal2.log", "r") as log_file,
    open("data/OpenStack/openstack_normal2.csv", "w+", newline="") as csv_file,
):
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "LineId",
            "Date",
            "Time",
            "Pid",
            "Level",
            "Component",
            "Content",
            "EventId",  # This will be filled based on the templates
        ]
    )

    line_id = 1
    for line in log_file:
        line = line.strip()
        match = pattern.match(line)
        if not match:
            print(f"Line {line_id} does not match the pattern:\n {line}")
            continue
        # Extract fields from the matched line
        filename, date, time, pid, level, component, bracketed, content = match.groups()
        for ev_id, pattern_event in templates:
            if pattern_event.match(content):
                event_id = ev_id
                break  # Use the first matching template
        # Write to CSV
        csv_writer.writerow(
            [
                line_id,
                date,
                time,
                pid,
                level,
                component,
                content.strip() if content else "",  # Clean up whitespace
                event_id,
            ]
        )
    else:
        # Handle lines that don't match the pattern (optional)
        csv_writer.writerow([line_id, line] + [""] * 7)
    line_id += 1


def deeplog_df_transfer(df, label):
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df[["datetime", "EventId"]]
    deeplog_df = (
        df.set_index("datetime").resample("1min").apply(_custom_resampler).reset_index()
    )
    deeplog_df["Label"] = label
    return deeplog_df


def _custom_resampler(array_like):
    return list(array_like)


##########
# Parser #
##########
input_dir = "./data/OpenStack/"
output_dir = "./data/openstack_result/"


##################
# Transformation #
##################
df_abnormal = pd.read_csv("data/OpenStack/openstack_abnormal.csv")
df_normal1 = pd.read_csv("data/OpenStack/openstack_normal1.csv")
df_normal2 = pd.read_csv("data/OpenStack/openstack_normal2.csv")


#########
# Train #
#########
deeplog_train = deeplog_df_transfer(df_normal1, "Success")

###############
# Test Normal #
###############
deeplog_test_normal = deeplog_df_transfer(df_normal2, "Success")
#################
# Test Abnormal #
#################
deeplog_test_abnormal = deeplog_df_transfer(df_abnormal, "Fail")

full_deeplog_df = pd.concat(
    [deeplog_train, deeplog_test_normal, deeplog_test_abnormal], ignore_index=True
)
full_deeplog_df.to_csv(output_dir + "deeplog_full.csv", index=False, header=True)
