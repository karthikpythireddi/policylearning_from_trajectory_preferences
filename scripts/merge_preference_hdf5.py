#!/usr/bin/env python3
"""
merge_preference_hdf5.py

Merge multiple preference HDF5 files (same task) into one.

Usage:
  python scripts/merge_preference_hdf5.py \
    preference_data/gr1/PnPCounterToPan_*_preferences.hdf5 \
    --output preference_data/gr1/PnPCounterToPan_preferences_merged.hdf5
"""
import argparse
import os
import h5py


def merge(input_paths: list[str], output_path: str):
    total_pairs = 0
    counts = {"success_vs_failure": 0, "efficiency": 0,
               "reward_comparison": 0, "noise_injection": 0}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as dst:
        for src_path in input_paths:
            with h5py.File(src_path, "r") as src:
                task_name = src["metadata"].attrs.get("task_name", "unknown")
                n = int(src["metadata"].attrs["n_pairs"])
                print(f"  {src_path}: {n} pairs")

                for i in range(n):
                    src_key = f"pair_{i}"
                    dst_key = f"pair_{total_pairs}"
                    src.copy(src_key, dst, name=dst_key)
                    ptype = dst[dst_key].attrs["preference_type"]
                    if ptype in counts:
                        counts[ptype] += 1
                    total_pairs += 1

        meta = dst.create_group("metadata")
        meta.attrs["n_pairs"]              = total_pairs
        meta.attrs["task_name"]            = task_name
        meta.attrs["n_success_vs_failure"] = counts["success_vs_failure"]
        meta.attrs["n_efficiency"]         = counts["efficiency"]
        meta.attrs["n_reward_comparison"]  = counts["reward_comparison"]
        meta.attrs["n_noise_injection"]    = counts["noise_injection"]

    print(f"\nMerged {total_pairs} pairs -> {output_path}")
    print(f"  success_vs_failure: {counts['success_vs_failure']}")
    print(f"  efficiency:         {counts['efficiency']}")
    print(f"  reward_comparison:  {counts['reward_comparison']}")
    print(f"  noise_injection:    {counts['noise_injection']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files to merge")
    parser.add_argument("--output", required=True, help="Output merged HDF5 path")
    args = parser.parse_args()
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
