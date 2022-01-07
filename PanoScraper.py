import csv
import glob
import multiprocessing as mp
import os
import random
import subprocess
from time import perf_counter
from itertools import islice
from datatypes.label import Label
from datatypes.panorama import Panorama
from datatypes.point import Point
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

GSV_IMAGE_WIDTH  = 13312
GSV_IMAGE_HEIGHT = 6656

# null crops per pano
NULLS_PER_PANO = 0

def bulk_scrape_panos(n, start_row, path_to_labeldata_csv, local_dir, remote_dir, output_csv_name):
    # TODO: find way to clear to pano_downloads folder and batch.txt file
    # on execution.
    t_start = perf_counter()
    panos = {}
    row_count = n
    start_row = start_row # 1-indexed, ignore the header row

    # create a csv reader to read input csv
    csv_file = open(path_to_labeldata_csv)
    csv_f = csv.reader(csv_file)

    # create a csv writer to write the output csvs
    path_to_output_csv = os.path.join(local_dir, output_csv_name)
    csv_output = open(path_to_output_csv, 'w')
    csv_w = csv.writer(csv_output)
    fields = Label.header_row()
    csv_w.writerow(fields)

    # accumulate list of pano ids to gather from sftp
    for row in islice(csv_f, start_row, start_row + row_count):
        print(row[0])
        pano_id = row[0]
        if pano_id != 'tutorial':
            csv_w.writerow(row)
            if not pano_id in panos:
                panos[pano_id] = Panorama()
            panos[pano_id].add_feature(row)

    # get null rows from panos
    for pano_id in panos:
        null_rows = get_null_rows(panos[pano_id])
        for null_row in null_rows:
            csv_w.writerow(null_row)
    
    # get available cpu_count
    cpu_count = mp.cpu_count() if mp.cpu_count() <= 8 else 8

    # split pano set into chunks for multithreading
    pano_set = panos.keys()
    pano_set_size = len(pano_set)
    i = 0
    processes = []
    while i < pano_set_size:
        chunk_size = (pano_set_size - i) // cpu_count
        pano_ids = set(islice(pano_set, i, i + chunk_size))
        process = mp.Process(target=acquire_n_panos, args=(remote_dir, local_dir, pano_ids, cpu_count))
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

    print("Finished Scraping.")
    print()

    return pano_set_size, execution_time

# Get a collection of "null" rows from a pano.
def get_null_rows(pano, min_dist = 70, bottom_space = 1600, side_space = 300):
    null_rows = []
    while len(null_rows) < NULLS_PER_PANO:
        x = random.uniform(side_space, GSV_IMAGE_WIDTH - side_space)
        y = random.uniform(- (GSV_IMAGE_HEIGHT/2 - bottom_space), 0)
        point = Point(x, y)
        valid_null = True
        for feat in pano.all_feats():
            if point.dist(feat.point()) <= min_dist:
                valid_null = False
                break
        if valid_null:
            # Using 0 for "null" label_type_id.
            row = [pano.pano_id, x, y, 0, pano.photog_heading, pano.photog_pitch, None, None, None]
            null_rows.append(row)
    return null_rows

def acquire_n_panos(remote_dir, local_dir, pano_ids, thread_id):
    sftp_command_list = ['cd {}'.format(remote_dir), 'lcd {}'.format(local_dir)]

    # create collection of commands
    for pano_id in pano_ids:
        # get first two characters of pano id
        two_chars = pano_id[:2]

        # get jpg for pano id
        sftp_command_list.append('-get ./{prefix}/{full_id}.jpg'.format(prefix=two_chars, full_id=pano_id))
    
    thread_batch_txt = 'batch{}.text'.format(thread_id)
    bash_command = "sftp -b {} -P 9000 -i alphie-sftp/alphie_pano ml-sftp@sftp.cs.washington.edu".format(thread_batch_txt)
    with open(thread_batch_txt, 'w') as sftp_file:
        for sftp_command in sftp_command_list:
            sftp_file.write("%s\n" % sftp_command)
        sftp_file.write('quit\n')

    sftp = subprocess.Popen(bash_command.split(), shell=False)
    result = sftp.communicate()
    print(result)
    if sftp.returncode != 0:
        print("sftp failed on one or more commands: {0}".format(sftp_command_list))

def clean_panos(path_to_panos):
    t_start = perf_counter()

    # get list of pano paths
    panos = glob.glob(path_to_panos + "/*.jpg")

    # get available cpu_count
    cpu_count = mp.cpu_count() if mp.cpu_count() <= 8 else 8
    # cpu_count = 1

    # split pano set into chunks for multithreading
    pano_set_size = len(panos)
    i = 0
    processes = []
    while i < pano_set_size:
        chunk_size = (pano_set_size - i) // cpu_count
        print(chunk_size)
        pano_ids = set(islice(panos, i, i + chunk_size))
        print(pano_ids)
        process = mp.Process(target=clean_n_panos, args=(pano_ids,))
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
    return execution_time

def clean_n_panos(panos):
    for pano_path in panos:
        print(pano_path)
        with Image.open(pano_path) as p:
            original_size = p.size
            if original_size != (GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT):
                # check if pano needs cleaning by looking for black space
                try:
                    pix = p.load()
                    if pix[GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT] == (0,0,0) and pix[original_size[0] - 1, original_size[1] - 1] == (0, 0, 0):
                        print("resizing ", pano_path)
                        im = p.crop((0, 0, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT))
                        im = im.resize(original_size)
                        im.save(pano_path)
                except Exception as e:
                    print("error on ", p)
                    print(p.size)
                    print(e)

