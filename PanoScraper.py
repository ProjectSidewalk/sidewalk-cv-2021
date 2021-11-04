import csv
import os
import subprocess
from time import perf_counter
from itertools import islice
from datatypes.label import Label
from datatypes.panorama import Panorama
from datatypes.point import Point
import random

GSV_IMAGE_WIDTH  = 13312
GSV_IMAGE_HEIGHT = 6656

# null crops per pano
NULLS_PER_PANO = 3

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

    # create collection of commands
    # set remote and local working directories
    sftp_command_list = ['cd {}'.format(remote_dir), 'lcd {}'.format(local_dir)]
    for pano_id in panos.keys():
        # get first two characters of pano id
        two_chars = pano_id[:2]

        # get jpg for pano id
        sftp_command_list.append('-get {prefix}/{full_id}.jpg'.format(prefix=two_chars, full_id=pano_id))

    bash_command = "sftp -b batch.txt -P 9000 -i alphie-sftp/alphie_pano ml-sftp@sftp.cs.washington.edu"
    with open('batch.txt', 'w') as sftp_file:
        for sftp_command in sftp_command_list:
            sftp_file.write("%s\n" % sftp_command)
        sftp_file.write('quit\n')

    sftp = subprocess.Popen(bash_command.split(), shell=False)
    result = sftp.communicate()
    print(result)
    if sftp.returncode != 0:
        print("sftp failed on one or more commands: {0}".format(sftp_command_list))

    t_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:",
                                            t_stop-t_start)

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
            row = [pano.pano_id, x, y, 0, pano.photog_heading, None, None, None]
            null_rows.append(row)
    return null_rows
