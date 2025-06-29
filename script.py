import os
import tarfile
import gzip
import subprocess
import os
from pathlib import Path
import shutil
from dask.distributed import Client
import dask.dataframe as dd

def extract_data():
    metrics_dir = os.environ.get("HOME") + "/lsc-project/data/MSMetrics"
    rtmcr_dir = os.environ.get("HOME") + "/lsc-project/data/MSRTMCR"

    extracted_metrics = os.environ.get("SCRATCH") + "/extracted/MSMetrics"
    extracted_rtmcr = os.environ.get("SCRATCH") + "/extracted/MSRTMCR"
    os.makedirs(extracted_metrics, exist_ok=True)
    os.makedirs(extracted_rtmcr, exist_ok=True)

    def extract_tar_gz(file_path, output_dir, expected_filename):
        expected_path = os.path.join(output_dir, expected_filename)
        if os.path.exists(expected_path):
            print(f"Already unpacked: {expected_filename}")
            return
        print(f"Unpacking: {file_path}")
        try:
            with gzip.open(file_path, 'rb') as f_in:
                with tarfile.open(fileobj=f_in) as tar:
                    tar.extractall(path=output_dir)
        except EOFError:
            print(f"EOFError: {file_path}")
        except tarfile.ReadError:
            print(f"tarfile.ReadError: {file_path}")

    for fname in os.listdir(metrics_dir):
        if fname.endswith(".tar.gz"):
            expected_csv = fname.replace(".tar.gz", "").replace("MSMetrics", "MSMetricsUpdate") + ".csv"
            extract_tar_gz(os.path.join(metrics_dir, fname), extracted_metrics, expected_csv)

    for fname in os.listdir(rtmcr_dir):
        if fname.endswith(".tar.gz"):
            expected_csv = fname.replace(".tar.gz", "").replace("MSRTMCR", "MCRRTUpdate") + ".csv"
            extract_tar_gz(os.path.join(rtmcr_dir, fname), extracted_rtmcr, expected_csv)

def process_data(start, end):
    extracted_metrics = os.environ.get("SCRATCH") + "/extracted/MSMetrics"
    extracted_rtmcr = os.environ.get("SCRATCH") + "/extracted/MSRTMCR"

    df_metrics = dd.read_csv(os.path.join(extracted_metrics, "*.csv"))
    df_rtmcr = dd.read_csv(os.path.join(extracted_rtmcr, "*.csv"))

    rt_cols = [col for col in df_rtmcr.columns if col.endswith("_rt")]
    mcr_cols = [col for col in df_rtmcr.columns if col.endswith("_mcr")]

    rpc_rt_cols = [col for col in df_rtmcr.columns if col.endswith("rpc_rt")]
    rpc_mcr_cols = [col for col in df_rtmcr.columns if col.endswith("rpc_mcr")]

    mq_rt_cols = [col for col in df_rtmcr.columns if col.endswith("mq_rt")]
    mq_mcr_cols = [col for col in df_rtmcr.columns if col.endswith("mq_mcr")]

    df_rtmcr["rt"] = df_rtmcr[rt_cols].sum(axis=1)
    df_rtmcr["mcr"] = df_rtmcr[mcr_cols].sum(axis=1)
    df_rtmcr["rpc_rt"] = df_rtmcr[rpc_rt_cols].sum(axis=1)
    df_rtmcr["rpc_mcr"] = df_rtmcr[rpc_mcr_cols].sum(axis=1)
    df_rtmcr["mq_rt"] = df_rtmcr[mq_rt_cols].sum(axis=1)
    df_rtmcr["mq_mcr"] = df_rtmcr[mq_mcr_cols].sum(axis=1)

    df_rtmcr_grouped = (
        df_rtmcr.groupby(["timestamp", "msinstanceid"])[["rt", "mcr", "http_mcr", "http_rt", "rpc_rt", "rpc_mcr", "mq_rt", "mq_mcr"]]
        .sum()
        .reset_index()
    )

    df_metrics_grouped = (
        df_metrics.groupby(["timestamp", "msinstanceid"])[["cpu_utilization", "memory_utilization"]]
        .sum()
        .reset_index()
    )

    df_final = dd.merge(df_metrics_grouped, df_rtmcr_grouped, on=["timestamp", "msinstanceid"], how="inner")
    df_final["msinstanceid"] = df_final["msinstanceid"].str.extract(r"(MS_\d+)", expand=False)

    df_agg = (
        df_final.groupby(["timestamp", "msinstanceid"])[["rt", "mcr", "cpu_utilization", "memory_utilization", "http_mcr", "http_rt", "rpc_rt", "rpc_mcr", "mq_rt", "mq_mcr"]]
        .mean()
        .reset_index()
    )

    results_dir = Path.home() / "lsc-project" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_agg.compute().to_csv(results_dir / f"results_{start}_{end}.csv", index=False)


WORKDIR = Path.home() / "lsc-project"
SCRATCH = os.environ.get("SCRATCH")
RESULTS_DIR = WORKDIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
if __name__ == '__main__':
    client = Client(n_workers=20)
    for day in range(14):
        for hour in range(0, 24):
            start = f"{day}d{hour:02d}"
            if day == 13 and hour == 23:
                break
            if hour < 23:
                end = f"{day}d{hour + 1:02d}"
            else:
                end = f"{day + 1}d00"

            print(f"Downloading: {start} → {end}")
            subprocess.run(
                    [f"{WORKDIR}/fetch.sh", f"start_date={start}", f"end_date={end}"],
                    check=True,
            )
            print("Extracting data")
            extract_data()
            print("Processing data")
            process_data(start, end)
            shutil.rmtree(WORKDIR / "data" / "MSMetrics", ignore_errors=True)
            shutil.rmtree(WORKDIR / "data" / "MSRTMCR", ignore_errors=True)
            shutil.rmtree(Path(SCRATCH) / "extracted", ignore_errors=True)