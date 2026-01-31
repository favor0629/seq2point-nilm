#!/usr/bin/env python3
"""
create_trainset_ukdale.py
修改版：只处理 house_1（或 params_appliance 指定的 houses 列表，但默认使用 house 1），
并按时间顺序把单个 house 的数据切分为 train/val/test = 70%/10%/20%（可通过参数修改）。

输出三份 CSV（含 header），覆盖写入：
  <save_path>/<appliance>_training_.csv
  <save_path>/<appliance>_validation_.csv
  <save_path>/<appliance>_test_.csv
"""

from ukdale_parameters import *
import pandas as pd
import numpy as np
import time
import argparse
import os

# 默认参数（如需改，使用命令行覆盖）
DATA_DIRECTORY = '../../data/refit/UKDALE/'
SAVE_PATH = './kettle/'   # default; 会在运行时根据 appliance 覆盖
AGG_MEAN = 522
AGG_STD = 814
SAMPLE_SECONDS = 8

def load_dataframe(directory, building, channel, col_names=['time', 'data'], nrows=None):
    path = os.path.join(directory, 'house_' + str(building), 'channel_' + str(channel) + '.dat')
    df = pd.read_table(path,
                       sep=r"\s+",
                       nrows=nrows,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning example for NILM (house1 split)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the UKDALE dat files (root with house_X folders).')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                        help='appliance to process: kettle, microwave, fridge, dishwasher, washingmachine')
    parser.add_argument('--aggregate_mean', type=float, default=AGG_MEAN,
                        help='Mean of mains aggregate (used for normalization)')
    parser.add_argument('--aggregate_std', type=float, default=AGG_STD,
                        help='Std of mains aggregate (used for normalization)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='Directory to store output CSVs (will be created if missing).')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train ratio (time-ordered)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation ratio (time-ordered)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio (time-ordered)')
    parser.add_argument('--sample_seconds', type=int, default=SAMPLE_SECONDS, help='resample period in seconds (default 8S)')
    return parser.parse_args()

def main():
    args = get_arguments()
    appliance_name = args.appliance_name
    data_dir = args.data_dir
    sample_seconds = args.sample_seconds

    # ensure ratios sum to 1 (allow floating rounding)
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0; got sum = {:.6f}".format(total_ratio))

    # create save_path and ensure trailing slash is consistent
    save_path = os.path.join(args.save_path, '')   # normalize
    os.makedirs(save_path, exist_ok=True)

    # Validate appliance present in params_appliance
    if appliance_name not in params_appliance:
        raise ValueError("appliance '{}' not found in ukdale_parameters.py".format(appliance_name))

    # Decide which house(s) to use: use params_appliance mapping (should be [1]) but allow explicit override
    houses = params_appliance[appliance_name].get('houses', [1])
    # For your requirement, we only use house 1; ensure it's in the list
    if 1 not in houses:
        print("Warning: house 1 not in params_appliance['{}'].houses; forcing use of house 1".format(appliance_name))
        houses = [1]

    # For single-house processing we will process only house 1 and aggregate all rows into df_all
    df_all = pd.DataFrame(columns=['aggregate', appliance_name], dtype='float32')

    start_time = time.time()

    for h in houses:
        # channel of appliance for this house
        idx = params_appliance[appliance_name]['houses'].index(h)
        channel = params_appliance[appliance_name]['channels'][idx]

        mains_df = load_dataframe(data_dir, h, 1, col_names=['time', 'aggregate'])
        app_df = load_dataframe(data_dir, h, channel, col_names=['time', appliance_name])

        # parse timestamps to datetime (unit seconds)
        mains_df['time'] = pd.to_datetime(mains_df['time'].astype(np.int64), unit='s')
        mains_df.set_index('time', inplace=True)
        mains_df.columns = ['aggregate']
        # resample to fixed sampling frequency and fill small gaps (same logic as original)
        mains_df = mains_df.resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        mains_df = mains_df.dropna()

        app_df['time'] = pd.to_datetime(app_df['time'].astype(np.int64), unit='s')
        app_df.set_index('time', inplace=True)
        app_df = app_df.resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        app_df = app_df.dropna()

        # align and drop NaNs
        df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna().reset_index()

        if df_align.empty:
            print("Warning: no aligned data for house", h, " appliance", appliance_name)
            continue

        # normalization
        mean = params_appliance[appliance_name].get('mean', 0.0)
        std = params_appliance[appliance_name].get('std', 1.0)

        df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        # ensure time sorted
        df_align.sort_values('time', inplace=True)
        df_align.reset_index(drop=True, inplace=True)

        # append to df_all (for multi-house case; here only house1)
        df_all = pd.concat([df_all, df_align[['aggregate', appliance_name]]], ignore_index=True)

    total_rows = len(df_all)
    if total_rows == 0:
        raise RuntimeError("No data found after alignment and normalization. Check data_dir and dat files.")

    # compute split indices (time-ordered)
    n_train = int(np.floor(args.train_ratio * total_rows))
    n_val = int(np.floor(args.val_ratio * total_rows))
    n_test = total_rows - n_train - n_val

    if n_test <= 0:
        raise RuntimeError("Test set size <= 0. Check ratios and data size.")

    train_df = df_all.iloc[:n_train].copy()
    val_df = df_all.iloc[n_train:n_train + n_val].copy()
    test_df = df_all.iloc[n_train + n_val:].copy()

    # file names (overwrite existing files)
    train_file = os.path.join(save_path, appliance_name + '_training_.csv')
    val_file = os.path.join(save_path, appliance_name + '_validation_.csv')
    test_file = os.path.join(save_path, appliance_name + '_test_.csv')

    # write CSVs with header (overwrite)
    train_df.to_csv(train_file, index=False, header=True)
    val_df.to_csv(val_file, index=False, header=True)
    test_df.to_csv(test_file, index=False, header=True)

    elapsed = (time.time() - start_time) / 60.0
    print("Wrote files to:", save_path)
    print("  train:", train_file, "rows:", len(train_df))
    print("  val:  ", val_file, "rows:", len(val_df))
    print("  test: ", test_file, "rows:", len(test_df))
    print("Total rows:", total_rows, "elapsed: {:.2f} min".format(elapsed))

if __name__ == '__main__':
    main()
