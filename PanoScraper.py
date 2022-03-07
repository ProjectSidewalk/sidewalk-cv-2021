import glob
import multiprocessing as mp
import random
import subprocess

from datatypes.panorama import Panorama
from datatypes.point import Point
from itertools import islice
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from time import perf_counter

GSV_IMAGE_WIDTH  = 13312
GSV_IMAGE_HEIGHT = 6656

# null crops per pano
NULLS_PER_PANO = 0

BLACK_THRESHOLD = (10, 10, 10)

def bulk_scrape_panos(data_chunk, panos, local_dir, remote_dir):
    t_start = perf_counter()

    pano_set = set()

    # accumulate list of pano ids to gather from sftp
    df_dict = data_chunk.to_dict('records')
    for row in df_dict:
        print(row['gsv_panorama_id'])
        pano_id = row['gsv_panorama_id']
        pano_set.add(pano_id)
        if pano_id != 'tutorial':
            if not pano_id in panos:
                panos[pano_id] = Panorama()
            panos[pano_id].add_feature(list(row.values()))
    
    # get available cpu_count
    cpu_count = mp.cpu_count() if mp.cpu_count() <= 8 else 8

    # split pano set into chunks for multithreading
    # pano_set = panos.keys()
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
# TODO: update with new label structure
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
    with open(thread_batch_txt, 'w', newline='') as sftp_file:
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
    cpu_count = mp.cpu_count()

    # split pano set into chunks for multithreading
    pano_set_size = len(panos)
    i = 0
    processes = []
    while i < pano_set_size:
        chunk_size = (pano_set_size - i) // cpu_count
        pano_ids = set(islice(panos, i, i + chunk_size))
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
        with Image.open(pano_path) as p:
            original_size = p.size
            if original_size != (GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT):
                # check if pano needs cleaning by looking for black space
                try:
                    pix = p.load()
                    if pix[GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT] <= BLACK_THRESHOLD and pix[original_size[0] - 1, original_size[1] - 1] <= BLACK_THRESHOLD:
                        print("resizing ", pano_path)
                        im = p.crop((0, 0, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT))
                        im = im.resize(original_size)
                        im.save(pano_path)
                except Exception as e:
                    print("error on ", p)
                    print(p.size)
                    print(e)
