import os
import subprocess

def set_up_traintestset_symlink(
    databank_t1w_dir = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w',
    traintestset_dir = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/T1wAgePredict',
    suffix = '_T1w_MNI152_Warped.nii.gz'
):
    
    for root, dirs, files in os.walk(databank_t1w_dir):
        for fn in files:
            if fn.endswith(suffix):
                fn_target = os.path.join(root, fn)
                fn_link = fn_target.replace(databank_t1w_dir, traintestset_dir)
                
                subprocess.run(['mkdir', '-p', os.path.dirname(fn_link)])
                subprocess.run(['ln', '-s', fn_target, fn_link])
                                
if __name__ == "__main__":
    set_up_traintestset_symlink()