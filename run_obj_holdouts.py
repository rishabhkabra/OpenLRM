import torch
import subprocess
import tqdm
import shutil
import os
import logging

if __name__ == '__main__':
    assert (torch.cuda.is_available())
    parent_dir = '/home/rkabra_google_com/data/objaverse_xl-holdouts/'
    uids = os.listdir(parent_dir)
    # uids = ['000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10']
    uids = [uid for uid in uids if len(uid) == 64]
    for uid in tqdm.tqdm(uids):
        filename = os.path.join(parent_dir, uid, 'textured_000.png')
        shutil.copyfile(filename, f'assets/{uid}.png')
        try:
            subprocess.call(f'python -m openlrm.launch infer.lrm --infer ./configs/infer-b.yaml model_name=zxhezexin/openlrm-mix-base-1.1 image_input=assets/{uid}.png export_video=true export_mesh=true', shell=True, cwd='/home/rkabra_google_com/OpenLRM')
        except Exception:
            logging.exception("Failed uid {uid}")
