import pandas as pd
import argparse


def analyze_perf_csv(csv_file, layers, show_all, out_file=None):
    df = pd.read_csv(csv_file)
    df = df[df["DEVICE FW DURATION [ns]"] != "-"]
    df["DEVICE FW DURATION [ns]"] = df["DEVICE FW DURATION [ns]"].astype(int)
    df[
        [
            "INPUT_0_W",
            "INPUT_0_Z",
            "INPUT_0_Y",
            "INPUT_0_X",
            "INPUT_1_W",
            "INPUT_1_Z",
            "INPUT_1_Y",
            "INPUT_1_X",
            "OUTPUT_0_W",
            "OUTPUT_0_Z",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
        ]
    ] = (
        df[
            [
                "INPUT_0_W",
                "INPUT_0_Z",
                "INPUT_0_Y",
                "INPUT_0_X",
                "INPUT_1_W",
                "INPUT_1_Z",
                "INPUT_1_Y",
                "INPUT_1_X",
                "OUTPUT_0_W",
                "OUTPUT_0_Z",
                "OUTPUT_0_Y",
                "OUTPUT_0_X",
            ]
        ]
        .replace("-", 1)
        .astype(int)
    )
    sorted_df = df.sort_values(by="DEVICE FW DURATION [ns]", ascending=False)
    sum_duration = df["DEVICE FW DURATION [ns]"].sum()

    matmul_rows = sorted_df[sorted_df["OP CODE"].str.contains("Matmul")]

    matmul_rows.loc[:, "bytes"] = (
        matmul_rows["INPUT_1_W"] * matmul_rows["INPUT_1_Z"] * matmul_rows["INPUT_1_Y"] * matmul_rows["INPUT_1_X"]
    )
    matmul_rows.loc[:, "flops"] = 2 * matmul_rows["INPUT_0_Y"] * matmul_rows["INPUT_0_X"] * matmul_rows["OUTPUT_0_X"]
    matmul_rows["GB/s"] = matmul_rows["bytes"] / matmul_rows["DEVICE FW DURATION [ns]"]
    matmul_rows["TFLOP/s"] = matmul_rows["flops"] / matmul_rows["DEVICE FW DURATION [ns]"] / 1000
    matmul_rows["% DRAM (240)"] = 100 * matmul_rows["GB/s"] / 240  # Peak expected WH bandwidth
    matmul_rows["% FPU (82)"] = 100 * matmul_rows["TFLOP/s"] / 82  # Peak theoretical FP16 FPU performance
    matmul_rows["% TIME"] = 100 * matmul_rows["DEVICE FW DURATION [ns]"] / sum_duration
    matmul_rows["% TIME SUM"] = matmul_rows["% TIME"].cumsum()
    matmul_sum_duration = matmul_rows["DEVICE FW DURATION [ns]"].sum()

    # shorten some column names
    matmul_rows.rename(columns={"DEVICE FW DURATION [ns]": "DURATION [ns]"}, inplace=True)
    sorted_df.rename(columns={"DEVICE FW DURATION [ns]": "DURATION [ns]"}, inplace=True)

    # # calculate bytes and flops for all ops
    # sorted_df.loc[:, "TOTAL_BYTES"] = (
    #     sorted_df["INPUT_1_W"] * sorted_df["INPUT_1_Z"] * sorted_df["INPUT_1_Y"] * sorted_df["INPUT_1_X"]
    # )

    data_type = {"BFLOAT16": 2, "BFLOAT8_B": 1}
    sorted_df[["INPUT_0_DATA TYPE", "INPUT_1_DATA TYPE", "OUTPUT_0_DATA TYPE"]] = (
        sorted_df[["INPUT_0_DATA TYPE", "INPUT_1_DATA TYPE", "OUTPUT_0_DATA TYPE"]]
        .applymap(data_type.get)
        .fillna(0)
        .astype(int)
    )
    # sorted_df[['INPUT_0_DATA TYPE', 'INPUT_1_DATA TYPE', 'INPUT_2_DATA TYPE', 'OUTPUT_0_DATA TYPE']] = sorted_df[['INPUT_0_DATA TYPE', 'INPUT_1_DATA TYPE', 'INPUT_2_DATA TYPE', 'OUTPUT_0_DATA TYPE']].applymap(data_type.get).fillna(0).astype(int)

    # for all INPUT W, Z, Y, X set fillna(0) if it's '-'
    # columns_to_replace_values = [ "INPUT_0_W", "INPUT_0_Z", "INPUT_0_Y", "INPUT_0_X", "INPUT_1_W", "INPUT_1_Z", "INPUT_1_Y", "INPUT_1_X", "INPUT_2_X", "INPUT_2_Y", "INPUT_2_Z", "INPUT_2_W", "OUTPUT_0_W", "OUTPUT_0_Z", "OUTPUT_0_Y", "OUTPUT_0_X"]
    columns_to_replace_values = [
        "INPUT_0_W",
        "INPUT_0_Z",
        "INPUT_0_Y",
        "INPUT_0_X",
        "INPUT_1_W",
        "INPUT_1_Z",
        "INPUT_1_Y",
        "INPUT_1_X",
        "OUTPUT_0_W",
        "OUTPUT_0_Z",
        "OUTPUT_0_Y",
        "OUTPUT_0_X",
    ]
    sorted_df[columns_to_replace_values] = sorted_df[columns_to_replace_values].replace("-", 1).astype(int)

    sorted_df.loc[:, "TOTAL_BYTES"] = (
        (sorted_df["INPUT_0_MEMORY"].eq("DEV_0_DRAM_INTERLEAVED")).astype(int)
        * (
            sorted_df["INPUT_0_W"]
            * sorted_df["INPUT_0_Z"]
            * sorted_df["INPUT_0_Y"]
            * sorted_df["INPUT_0_X"]
            * sorted_df["INPUT_0_DATA TYPE"]
        )
        + (sorted_df["INPUT_1_MEMORY"].eq("DEV_0_DRAM_INTERLEAVED")).astype(int)
        * (
            sorted_df["INPUT_1_W"]
            * sorted_df["INPUT_1_Z"]
            * sorted_df["INPUT_1_Y"]
            * sorted_df["INPUT_1_X"]
            * sorted_df["INPUT_1_DATA TYPE"]
        )
        +
        # (sorted_df["INPUT_2_MEMORY"].eq("DEV_0_DRAM_INTERLEAVED")).astype(int) * (sorted_df["INPUT_2_W"] * sorted_df["INPUT_2_Z"] * sorted_df["INPUT_2_Y"] * sorted_df["INPUT_2_X"] * sorted_df['INPUT_2_DATA
        # TYPE']) +
        (sorted_df["OUTPUT_0_MEMORY"].eq("DEV_0_DRAM_INTERLEAVED")).astype(int)
        * (
            sorted_df["OUTPUT_0_W"]
            * sorted_df["OUTPUT_0_Z"]
            * sorted_df["OUTPUT_0_Y"]
            * sorted_df["OUTPUT_0_X"]
            * sorted_df["OUTPUT_0_DATA TYPE"]
        )
    )

    sorted_df.loc[:, "flops"] = 2 * sorted_df["INPUT_0_Y"] * sorted_df["INPUT_0_X"] * sorted_df["OUTPUT_0_X"]
    sorted_df["DRAM BW GB/s"] = sorted_df["TOTAL_BYTES"] * (1000**3) / (sorted_df["DURATION [ns]"] * (1024**3))
    sorted_df["TFLOP/s"] = sorted_df["flops"] / sorted_df["DURATION [ns]"] / 1000
    sorted_df["% DRAM (240)"] = 100 * sorted_df["DRAM BW GB/s"] / 240  # Peak expected WH bandwidth
    sorted_df["% FPU (82)"] = 100 * sorted_df["TFLOP/s"] / 82  # Peak theoretical FP16 FPU performance
    sorted_df["TOTAL MBs"] = sorted_df["TOTAL_BYTES"] / 1024 / 1024
    selected_columns = [
        "OP CODE",
        "% TIME",
        "% TIME SUM",
        "% DRAM (240)",
        "% FPU (82)",
        "DURATION [ns]",
        "GB/s",
        "TFLOP/s",
        "CORE COUNT",
        "INPUT_0_Y",
        "INPUT_0_X",
        "INPUT_1_Y",
        "INPUT_1_X",
        "OUTPUT_0_Y",
        "OUTPUT_0_X",
    ]
    print(matmul_rows[selected_columns])

    if show_all:
        selected_columns = [
            "OP CODE",
            "% TIME",
            "% TIME SUM",
            "DURATION [ns]",
            "% DEV CB WAIT",
            "DURATION SUM [ns]",
            "DRAM BW GB/s",
            "% DRAM (240)",
            "CORE COUNT",
            "INPUT_0_Y",
            "INPUT_0_X",
            "INPUT_1_Y",
            "INPUT_1_X",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
            "TOTAL MBs",
            "DEVICE COMPUTE CB WAIT FRONT [ns]",
        ]
        sorted_df["% TIME"] = 100 * sorted_df["DURATION [ns]"] / sum_duration
        sorted_df["% TIME SUM"] = sorted_df["% TIME"].cumsum()
        sorted_df["DURATION SUM [ns]"] = sorted_df["DURATION [ns]"].cumsum()
        sorted_df["% DEV CB WAIT"] = 100 * sorted_df["DEVICE COMPUTE CB WAIT FRONT [ns]"] / sorted_df["DURATION [ns]"]

        # trim all floats to 2 decimal places
        sorted_df = sorted_df.round(2)
        # print()
        print(sorted_df[selected_columns])
        selected_df = sorted_df[selected_columns]

        # save to file for further analysis
        if out_file:
            selected_df.to_csv(out_file, index=False, sep="\t")

    if layers:
        tokens_per_sec_user = 1000000000 / sum_duration / layers
        tokens_per_sec = 32 * tokens_per_sec_user
        print(f"Layer ms: {sum_duration / 1000000:.1f} ({matmul_sum_duration / sum_duration:.1%} matmul)")
        print(f"Tokens/sec/user: {tokens_per_sec_user:.1f}")
        print(f"Tokens/sec: {tokens_per_sec:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze perf CSV file")
    parser.add_argument(
        "-a", "--all", action="store_true", help="List ops in the CSV file - by default only matmul ops are shown."
    )
    parser.add_argument("-l", "--layers", type=int, help="Number of layers to extrapolate perf results up to.")
    parser.add_argument(
        "csv_file", type=str, help="Path to the perf CSV file from tt-metal for a single decoder layer."
    )
    parser.add_argument(
        "-o", "--out_file", required=False, type=str, help="Path to the output file for further analysis."
    )

    args = parser.parse_args()

    analyze_perf_csv(args.csv_file, layers=args.layers, show_all=args.all, out_file=args.out_file)


if __name__ == "__main__":
    main()
