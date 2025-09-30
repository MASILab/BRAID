import pdb
import tqdm
import argparse
from pathlib import Path
from typing import Union


def _has_session_layer(path_subject: Union[str, Path]) -> bool:
    path_subject = Path(path_subject)
    assert path_subject.is_dir(), f"{path_subject} is not a directory."
    return any(p.is_dir() and p.name.startswith('ses-') for p in path_subject.iterdir())


def symlink_qa_png(bids_name, qa_root):
    path_qa_dir = Path(qa_root) / bids_name / "BRAID"
    path_qa_dir.mkdir(parents=True, exist_ok=True)
    
    path_derivatives = Path(f"/nfs2/harmonization/BIDS/{bids_name}/derivatives")
    
    for path_subject in path_derivatives.glob("sub-*"):
        if path_subject.is_dir():
            cross_sectional = not _has_session_layer(path_subject)
            break
    search_pattern = "sub-*/BRAID*" if cross_sectional else "sub-*/ses-*/BRAID*"
    
    for path_braid in path_derivatives.glob(search_pattern):
        path_png = path_braid / 'final' / 'QA.png'
        if path_png.is_file():
            parts = path_png.relative_to(path_derivatives).parts
            if cross_sectional:
                assert len(parts) == 4, f"Unexpected path structure: {path_png}"
                filename = "_".join(parts[:2]) + ".png"
            else:
                assert len(parts) == 5, f"Unexpected path structure: {path_png}"
                filename = "_".join(parts[:3]) + ".png"
            
            path_symbolic = path_qa_dir / filename
            if path_symbolic.exists():
                print(f"Warning: {path_symbolic} already exists.")
            else:
                path_symbolic.symlink_to(path_png)
    
    print(f"Symbolic links created in {path_qa_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Create symbolic links for BRAID QA PNG images of a dataset in a flat directory structure, "
        "which is easier to view by the QA tool, using filenames to encode subject/session/run ids."
        ))
    parser.add_argument("-n", "--bids_name", type=str, required=True, help="Dataset name in BIDS.")
    parser.add_argument("-r", "--qa_root", type=str, required=False, default="/nfs2/harmonization/ADSP_QA", help="Root directory of the QA images.")
    args = parser.parse_args()
    
    symlink_qa_png(
        bids_name=args.bids_name,
        qa_root=args.qa_root
    )
