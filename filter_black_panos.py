import argparse
import datetime
import glob
import logging
import multiprocessing as mp
import os
import pandas as pd
import subprocess

from itertools import islice
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from subprocess import DEVNULL, STDOUT
from time import perf_counter

logging.basicConfig(filename=f'filter_black_panos.log', level=logging.INFO)
logging.info(f'FILTER SESSION TIMESTAMP: {datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")}')

FILTERED_PANOS_CSV = "filtered_panos.csv"
BATCH_TXT_FOLDER = "batches"
REMOTE_DIR = "sidewalk_panos/Panoramas/"
SFTP_KEY_PATH = "PATH TO SFTP KEY"

def acquire_n_panos(remote_dir, local_dir, pano_subdirs, thread_id):
    sftp_command_list = ['cd {}'.format(remote_dir), 'lcd {}'.format(local_dir)]

    # create collection of commands
    for pano_subdir in pano_subdirs:
        # get jpg for pano id
        sftp_command_list.append(f'-get {os.path.join(pano_subdir, "*.jpg")}')
    
    thread_batch_txt = f'{BATCH_TXT_FOLDER}/batch{thread_id}.text'
    bash_command = f'sftp -b {thread_batch_txt} -P 9000 -i {SFTP_KEY_PATH} ml-sftp@sftp.cs.washington.edu'
    with open(thread_batch_txt, 'w', newline='') as sftp_file:
        for sftp_command in sftp_command_list:
            sftp_file.write("%s\n" % sftp_command)
        sftp_file.write('quit\n')

    sftp = subprocess.Popen(bash_command.split(), shell=False)
    result = sftp.communicate()
    print(result)
    if sftp.returncode != 0:
        print("sftp failed on one or more commands: {0}".format(sftp_command_list))

def bulk_scrape_panos(pano_sub_dirs_file, local_dir, remote_dir):
    t_start = perf_counter()

    if not os.path.isdir(BATCH_TXT_FOLDER):
        os.makedirs(BATCH_TXT_FOLDER)

    # accumulate list of pano id subfolders to pull panos from
    with open(pano_sub_dirs_file) as f:
        lines = f.readlines()
        pano_sub_dirs = [line.rstrip() for line in lines]

    pano_sub_dirs = pano_sub_dirs[:30]
    
    # get available cpu_count
    cpu_count = mp.cpu_count() if mp.cpu_count() <= 8 else 8

    # split pano set into chunks for multithreading
    pano_sub_dirs_size = len(pano_sub_dirs)
    i = 0
    processes = []
    while i < pano_sub_dirs_size:
        chunk_size = (pano_sub_dirs_size - i) // cpu_count
        pano_sub_dirs_chunk = pano_sub_dirs[i: i + chunk_size]
        print(pano_sub_dirs_chunk)
        process = mp.Process(target=acquire_n_panos, args=(remote_dir, local_dir, pano_sub_dirs_chunk, cpu_count))
        processes.append(process)
        cpu_count -= 1
        i += chunk_size

    # start processes
    for p in processes:
        p.start()

    # join processes once finished
    for p in processes:
        p.join()

    # remove batch txts
    for file in os.scandir(BATCH_TXT_FOLDER):
        os.remove(file.path)

    t_stop = perf_counter()
    execution_time = t_stop - t_start

    # print("Finished Scraping.")
    # print()

    return pano_sub_dirs_size, execution_time

def filter_black_panos(panos):
    t_start = perf_counter()
    with mp.Manager() as manager:
        # get available cpu_count
        cpu_count = mp.cpu_count()

        # panos to be filtered out
        filtered_panos = manager.list()

        # split pano set into chunks for multithreading
        pano_set_size = len(panos)
        i = 0
        processes = []
        while i < pano_set_size:
            chunk_size = (pano_set_size - i) // cpu_count
            panos_chunk = set(islice(panos, i, i + chunk_size))
            process = mp.Process(target=filter_n_panos, args=(panos_chunk, filtered_panos))
            processes.append(process)
            cpu_count -= 1
            i += chunk_size

        # start processes
        for p in processes:
            p.start()

        # join processes once finished
        for p in processes:
            p.join()

        t_stop = perf_counter()
        execution_time = t_stop - t_start
        return list(filtered_panos), execution_time

def filter_n_panos(panos, filtered_panos):
    for pano in panos:
        print(f'checking {pano}')
        try:
            with Image.open(pano) as im:
                top_left = im.crop((0, 0, 300, 300))
                if top_left.convert("L").getextrema() == (0, 0):
                    if im.convert("L").getextrema() == (0, 0):
                        # remove black pano
                        logging.info(f'{pano} is fully black')
                        filtered_panos.append({"pano_id": pano})
                        # os.remove(pano)
                    else:
                        logging.info(f'{pano} has a black upper left quadrant')
        except Exception as e:
            print(e)
            filtered_panos.append({"pano_id": pano})
            logging.info(f'{pano} has image problem')

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pano_dir', help='pano_dir - the directory containing panos to filter, i.e. pano-downloads/')
    parser.add_argument('pano_sub_dirs_file', help='pano_sub_dirs_file - file containing a list of pano subdirs to extract from')
    args = parser.parse_args()

    pano_dir = args.pano_dir
    pano_sub_dirs_file = args.pano_sub_dirs_file

    if not os.path.isdir(pano_dir):
        os.makedirs(pano_dir)

    num_pano_sub_dirs, scrape_exec_time = bulk_scrape_panos(pano_sub_dirs_file, pano_dir, REMOTE_DIR)

    panos = glob.glob(os.path.join(pano_dir, "*.jpg"))

    filtered_panos, filter_exec_time = filter_black_panos(panos)

    filtered_panos_df = pd.DataFrame.from_records(filtered_panos)
    filtered_panos_df.to_csv(FILTERED_PANOS_CSV, index=False)

    print()
    print("====================================================================================================")
    print(f'Total scrape execution time in seconds: {scrape_exec_time}')
    print(f'Total filter execution time in seconds: {filter_exec_time}')
