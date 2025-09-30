import argparse
from pathlib import Path

def find_failed_inference(root_dir):
    list_failed = []
    for path_log in Path(root_dir).glob("**/log.txt"):
        path_pred_csv = path_log.parent / "final" / "braid_predictions.csv"
        if not path_pred_csv.exists():
            list_failed.append(path_log)
    return list_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find failed BRAID inference runs")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the root directory of the derivatives")
    args = parser.parse_args()
    
    list_failed = find_failed_inference(args.root_dir)
    print(f"Found {len(list_failed)} failed inference runs:")
    for path_log in list_failed:
        print(path_log)
        