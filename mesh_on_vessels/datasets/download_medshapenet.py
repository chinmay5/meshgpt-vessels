import argparse
import os
import shutil
from pathlib import Path

import requests


def search_and_download_organ(dest_dir, organ):

    """
    Search the online MedShapeNet database
    using organ nomenclature, such as liver,
    and download the corresponding .stl files
    """

    if not os.path.exists('./temp/'):
        os.mkdir('./temp/')
    r = requests.get("https://medshapenet.ikim.nrw/uploads/MedShapeNetDataset.txt", stream=True)
    with open('./temp/MedShapeNetDataset.txt', 'wb') as f:
        f.write(r.content)
    print(f'searching {organ}...')
    matched_urls = []
    with open('./temp/MedShapeNetDataset.txt', 'r') as inF:
        for line in inF:
            if organ in line:
                matched_urls.append(line)
    if len(matched_urls) == 0:
        print(f'found {len(matched_urls)} entries of {organ}')
        if os.path.exists('./temp/'):
            shutil.rmtree('./temp/')
    else:
        save_folder = f'{dest_dir}/{organ}/'
        if not os.path.exists(save_folder):
            Path(save_folder).mkdir(parents=True, exist_ok=True)
        print(
            f'found {len(matched_urls)} entries of {organ}, started downloading... files are saved in folder {save_folder}')
        counter = 0
        print('_________ urls:')
        for url in matched_urls:
            print(url)
            r = requests.get(url.strip(), stream=True)
            filename = save_folder + organ + '_' + '{0:05}'.format(counter) + '.stl'
            counter += 1
            with open(filename, 'wb') as f:
                f.write(r.content)
        print(f'Download complete! Files are stored in folder {save_folder}')
        if os.path.exists('./temp/'):
            shutil.rmtree('./temp/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download organs from med_shape_net")

    # Required argument
    parser.add_argument('dest_dir', type=str, help='Save location of the downloaded data')
    parser.add_argument('organ', type=str, help='The organ to download')

    args = parser.parse_args()
    search_and_download_organ(args.dest_dir, args.organ)
    # e.g. --dest_dir=/net/cephfs/shares/menze.dqbm.uzh/chinmay/dataset/mesh_dataset/med_shapenet --organ=liver