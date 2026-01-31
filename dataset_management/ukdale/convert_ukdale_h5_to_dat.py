#!/usr/bin/env python3
"""
convert_ukdale_h5_to_dat.py

将 ukdale.h5 -> 原始 UK-DALE 风格的目录/文件:
  output_dir/house_<B>/channel_<C>.dat

每行格式:
  <unix_timestamp> <value>

用法示例:
  python convert_ukdale_h5_to_dat.py \
    --h5 /home/favor/datasets/UK_Dale/ukdale.h5 \
    --out /home/favor/datasets/UK_Dale_dat/ \
    --chunk 200000 \
    --dry_run

注意:
 - 脚本会查找每个 /building*/elec/meter*/table 数据集并导出。
 - 如果 dataset 是结构化 dtype(即 table rows)，脚本会尝试自动识别时间戳字段与数值字段。
 - 若无法识别字段，会在 dry-run 中报告并跳过写入。
"""
import os
import argparse
import h5py
import numpy as np

# 常见可能为时间戳或值的字段名（小写匹配）
TIME_CANDS = ('timestamp', 'time', 'date', 'index', 'datetime', 'ts')
VALUE_CANDS = ('power', 'active', 'apparent', 'value', 'reading', 'watts', 'power_w', 'power_watts')

def find_field(names, candidates):
    """
    在给定字段名元组 names 中寻找匹配 candidates（小写比较）。
    返回第一个匹配的原始字段名（保持原样），找不到则返回 None。
    """
    if names is None:
        return None
    lname = [n.lower() for n in names]
    for cand in candidates:
        if cand in lname:
            return names[lname.index(cand)]
    return None

def normalize_timestamp_field(ts_arr):
    """
    将多种 timestamp 表示转换为整数秒（Unix epoch）
    支持:
      - 整数/浮点（认为是 unix seconds 或 unix.seconds.fraction）
      - numpy.datetime64 -> 转为秒（astype('datetime64[s]').astype(int))
      - bytes/str -> 尝试 decode 后转为 int 或 parse via float
    返回 int numpy array
    """
    if np.issubdtype(ts_arr.dtype, np.integer):
        return ts_arr.astype(np.int64)
    if np.issubdtype(ts_arr.dtype, np.floating):
        return np.floor(ts_arr).astype(np.int64)
    # datetime64
    if np.issubdtype(ts_arr.dtype, np.datetime64):
        return ts_arr.astype('datetime64[s]').astype('int64')
    # bytes/strings
    if np.issubdtype(ts_arr.dtype, np.dtype('O')) or np.issubdtype(ts_arr.dtype, np.bytes_):
        # try decode if bytes
        try:
            # convert to numpy array of strings
            sarr = np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in ts_arr])
            # try numeric conversion
            if np.all([x.isdigit() for x in sarr]):
                return sarr.astype(np.int64)
            else:
                # try float then floor
                f = sarr.astype(np.float64)
                return np.floor(f).astype(np.int64)
        except Exception:
            raise ValueError("无法把 timestamp 字段解析为数值/日期: dtype=%s" % (ts_arr.dtype,))
    # fallback
    raise ValueError("未知的 timestamp dtype: %s" % (ts_arr.dtype,))

def write_dat_file(fp, ts_chunk, val_chunk):
    """
    把 ts_chunk, val_chunk 写入已经打开的文本文件对象 fp。
    ts_chunk, val_chunk 都为一维 numpy arrays，相同长度。
    一行格式: "<ts> <val>\n"
    """
    # 使用向量化生成字符串可能占内存，采用逐行写但在内层做简单 join 批量写
    lines = []
    for t, v in zip(ts_chunk, val_chunk):
        # 保证 t 是整数，v 保持原始浮点/整数格式
        lines.append(f"{int(t)} {v}\n")
    fp.writelines(lines)

def process_table_dataset(ds, out_dir, building, meter_name, chunk_size=100000, dry_run=False):
    """
    处理单个 dataset（h5py Dataset），写入 output dir。
    meter_name 例如 'meter10' -> channel id 10
    """
    # 目标输出文件路径
    try:
        ch_num = int(''.join([c for c in meter_name if c.isdigit()]))
    except Exception:
        ch_num = meter_name  # fallback, 但会变成字符串
    out_house_dir = os.path.join(out_dir, f"house_{building}")
    os.makedirs(out_house_dir, exist_ok=True)
    out_file = os.path.join(out_house_dir, f"channel_{ch_num}.dat")

    print(f"  dataset: {ds.name}  ->  {out_file}")

    # 如果只是 dry run，先检测字段、shape、dtype 并返回
    names = getattr(ds, 'dtype', None)
    if names is not None and getattr(names, 'names', None):
        field_names = names.names
    else:
        field_names = None

    # 尝试定位 timestamp field 与 value field
    t_field = find_field(field_names, TIME_CANDS) if field_names is not None else None
    v_field = find_field(field_names, VALUE_CANDS) if field_names is not None else None

    print("    detected fields:", field_names)
    print("    chosen time field:", t_field, " value field:", v_field)

    # 如果 dataset 是纯一维数值（没有 field_names），则认为 ds[:] 本身是值序列，timestamps 不存在 -> 不能写为标准 .dat
    if field_names is None:
        # 支持情况下：如果 dataset 仅包含两列二维数组且 dtype 是数值，尝试把第一列当 ts，第二列当 val
        if ds.ndim == 2 and ds.shape[1] >= 2:
            print("    dataset is 2D numeric, will treat col0 as time, col1 as value.")
            # we'll read col0, col1
            t_field_mode = 'col0'
        else:
            print("    dataset has no structured fields and is not 2D (skipping).")
            return

    if dry_run:
        # 打印样例不写入
        N = ds.shape[0]
        print(f"    dry-run: N={N}, dtype={ds.dtype}")
        # sample first non-empty chunk
        s0 = min(N, 10)
        try:
            sample = ds[:s0]
            print("    sample[0..%d]:" % s0, sample)
        except Exception as e:
            print("    sample read error:", e)
        return

    # 打开输出文件（覆盖写入）
    with open(out_file, 'w') as fp:
        total = ds.shape[0]
        read_idx = 0
        while read_idx < total:
            end = min(total, read_idx + chunk_size)
            chunk = ds[read_idx:end]
            # 提取 ts and val based on dtype
            if field_names is not None:
                # structured array
                if t_field is None or v_field is None:
                    # try to fallback: if there are exactly 2 fields, take first as time, second as value
                    if len(field_names) >= 2:
                        t_field = field_names[0] if t_field is None else t_field
                        v_field = field_names[1] if v_field is None else v_field
                    else:
                        raise RuntimeError(f"无法识别时间/数值字段: {field_names} in {ds.name}")
                ts_arr = chunk[t_field]
                val_arr = chunk[v_field]
            else:
                # numeric array 2D
                ts_arr = chunk[:, 0]
                val_arr = chunk[:, 1]

            # normalize/convert timestamps to integer seconds
            try:
                ts_int = normalize_timestamp_field(ts_arr)
            except Exception as e:
                # 如果 timestamp 解析失败，尝试将其视作 unix seconds float
                try:
                    ts_int = np.floor(ts_arr.astype(np.float64)).astype(np.int64)
                except Exception:
                    raise

            # ensure val_arr is numeric
            if np.issubdtype(val_arr.dtype, np.number):
                vals = val_arr.astype(np.float64)
            else:
                # try decode bytes->float or convert to numeric
                vals = np.array([float(x.decode() if isinstance(x, (bytes,bytearray)) else x) for x in val_arr], dtype=np.float64)

            # write to file
            write_dat_file(fp, ts_int, vals)
            read_idx = end

    print(f"  wrote {out_file} (rows ~ {total})")

def main(h5_path, out_dir, chunk_size=100000, dry_run=False, buildings=None):
    """
    遍历 h5 文件结构，查找 /building*/elec/meter*/table 数据集并处理。
    buildings: 可传入 [1,2] 指定仅处理这些 building，否则处理所有 building*
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        # find building groups
        for bname, bobj in f.items():
            # we expect group names like 'building1'
            if not bname.lower().startswith('building'):
                continue
            if buildings is not None:
                try:
                    bnum = int(''.join([c for c in bname if c.isdigit()]))
                    if bnum not in buildings:
                        continue
                except Exception:
                    pass
            # find elec group
            elec = bobj.get('elec')
            if elec is None:
                print("warning: no 'elec' in", bname)
                continue
            # iterate meters
            for meter_name, meter_obj in elec.items():
                # skip _i_table or other metadata groups: only process groups named meter*
                if not meter_name.lower().startswith('meter'):
                    continue
                # look for 'table' dataset under meter
                ds = meter_obj.get('table')
                if ds is None:
                    # try other datasets: pick first dataset under this group
                    found = None
                    for k, v in meter_obj.items():
                        if isinstance(v, h5py.Dataset):
                            found = v
                            break
                    if found is None:
                        print("  no dataset found under", meter_name, "skipping")
                        continue
                    ds = found
                # process this dataset
                try:
                    process_table_dataset(ds, out_dir, bname.replace('building',''), meter_name, chunk_size=chunk_size, dry_run=dry_run)
                except Exception as e:
                    print("  ERROR processing", ds.name, ":", e)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Convert ukdale.h5 to house/channel .dat files")
    p.add_argument('--h5', required=True, help='Path to ukdale.h5')
    p.add_argument('--out', required=True, help='Output root directory for house_x/channel_y.dat')
    p.add_argument('--chunk', type=int, default=100000, help='Chunk size to read from HDF5')
    p.add_argument('--dry_run', action='store_true', help='Only inspect datasets and do not write files')
    p.add_argument('--buildings', type=int, nargs='*', default=None, help='Optional list of building numbers to convert, e.g. --buildings 1 2')
    args = p.parse_args()
    main(args.h5, args.out, chunk_size=args.chunk, dry_run=args.dry_run, buildings=args.buildings)
