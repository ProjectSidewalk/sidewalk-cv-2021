import csv
import os
import subprocess
from time import perf_counter
from itertools import islice

def bulk_scrape_panos(n, path_to_labeldata_csv, local_dir, remote_dir, output_csv_name):
    # TODO: find way to clear to pano_downloads folder and batch.txt file
    # on execution.
    t_start = perf_counter()
    panos_to_acquire = set()
    row_count = n
    start_row = 1 # 1-indexed, ignore the header row

    # create a csv reader to read input csv
    csv_file = open(path_to_labeldata_csv)
    csv_f = csv.reader(csv_file)

    # create a csv writer to write the output csvs
    path_to_output_csv = os.path.join(local_dir, output_csv_name)
    csv_output = open(path_to_output_csv, 'w')
    csv_w = csv.writer(csv_output)
    fields = ['gsv_panorama_id', 'sv_image_x', 'sv_image_y', 'label_type_id', 'photographer_heading', 'heading', 'pitch', 'label_id']
    csv_w.writerow(fields)
    # accumulate list of pano ids to gather from sftp
    for row in islice(csv_f, start_row, start_row + row_count):
        print(row[0])
        pano_id = row[0]
        if pano_id != 'tutorial':
            panos_to_acquire.add(pano_id)
            csv_w.writerow(row)

    print(len(panos_to_acquire))

    # create collection of commands
    # set remote and local working directories
    sftp_command_list = ['cd {}'.format(remote_dir), 'lcd {}'.format(local_dir)]
    for pano_id in panos_to_acquire:
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
