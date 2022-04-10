import multiprocessing as mp
import os
import pandas as pd
import subprocess

from time import perf_counter
from xml.etree import ElementTree as ET

# current city we are gathering xml metadata for
CITY = "spgg"

# unretrievable pano ids csv
PATH_TO_PANO_ID_CSV = 'rawdata/test.csv' #f'rawdata/{CITY}_unretrievable_panos.csv'

# the local directory xml metadat will be downloaded to
LOCAL_DIR = 'xml-metadata/'

# the remote directory panos will be scraped from
REMOTE_DIR = f'sidewalk_panos/Panoramas/scrapes_dump_{CITY}'

# pano metadata csv
PANO_METADATA_CSV = f'{CITY}_pano_metadata.csv'

def bulk_scrape_xml_metadata(data_chunk, local_dir, remote_dir):
    t_start = perf_counter()
    row_count = len(data_chunk)
    
    # get available cpu_count
    cpu_count = mp.cpu_count() if mp.cpu_count() <= 8 else 8

    # split pano set into chunks for multithreading
    i = 0
    processes = []
    while i < row_count:
        chunk_size = (row_count - i) // cpu_count
        pano_ids = data_chunk[i : i + chunk_size]
        print(len(pano_ids))
        process = mp.Process(target=acquire_n_xml_files, args=(remote_dir, local_dir, pano_ids, cpu_count))
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

    return execution_time

def acquire_n_xml_files(remote_dir, local_dir, pano_ids, thread_id):
    sftp_command_list = ['cd {}'.format(remote_dir), 'lcd {}'.format(local_dir)]

    # create collection of commands
    for row in pano_ids:
        pano_id = row['gsv_panorama_id']

        # get first two characters of pano id
        two_chars = pano_id[:2]

        # get jpg for pano id
        sftp_command_list.append('-get ./{prefix}/{full_id}.xml'.format(prefix=two_chars, full_id=pano_id))
    
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

def extract_pano_metadata_from_xml(path_to_metadata_xml):
    with open(path_to_metadata_xml, 'rb') as pano_xml:
        tree = ET.parse(pano_xml)
        root = tree.getroot()
        # gsv_panorama_id,image_width,image_height,tile_width,tile_height,copyright,center_heading,origin_heading,origin_pitch
        row = {}
        for child in root:
            # TODO: investigate if we can get center_heading, origin_heading, origin_pitch
            if child.tag == 'data_properties':
                # get gsv_panorama_id, image_width, image_height, tile_width, tile_height
                row['gsv_panorama_id'] = child.attrib['pano_id']
                row['image_width'] = child.attrib['image_width']
                row['image_height'] = child.attrib['image_height']
                row['tile_width'] = child.attrib['tile_width']
                row['tile_height'] = child.attrib['tile_height']
            elif child.tag == 'copyright':
                # get copyright
                row['copyright'] = child.text

        return row

if __name__ ==  '__main__':
    print("CPU count: ", mp.cpu_count())

    # local directory to write to (relative to shell root)
    if not os.path.isdir(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    pano_metadatas = []

    total_successful_extractions = 0
    total_failed_extractions = 0

    t_start = perf_counter()
    for chunk in pd.read_csv(PATH_TO_PANO_ID_CSV, chunksize=10000):
        pano_id_list = chunk.to_dict('records')

        # scrape xml metadata for each pano id in current batch from SFTP server
        scraper_exec_time = bulk_scrape_xml_metadata(pano_id_list, LOCAL_DIR, REMOTE_DIR)

        # output execution metrics
        print("====================================================================================================")
        print("XML scraping metrics:")
        print("Elapsed time scraping XML files for {} pano_ids in seconds:".format(len(chunk)),
                                                scraper_exec_time)
        print()

        # update pano id image metadata (particularly pano size)
        successful_extractions = 0
        for row in pano_id_list:
            pano_id = row['gsv_panorama_id']

            xml_path = os.path.join(LOCAL_DIR, f'{pano_id}.xml')
            if os.path.exists(xml_path):
                pano_metadata = extract_pano_metadata_from_xml(xml_path)
                pano_metadatas.append(pano_metadata)
                successful_extractions += 1


        total_successful_extractions += successful_extractions
        total_failed_extractions += len(chunk) - successful_extractions

        # delete xml metadata downloads from current batch
        for file in os.scandir(LOCAL_DIR):
            os.remove(file.path)

    # write pano_metadatas to CSV
    pano_metadatas_df = pd.DataFrame(pano_metadatas)
    pano_metadatas_df.to_csv(PANO_METADATA_CSV, index=False)

    t_stop = perf_counter()
    total_execution_time = t_stop - t_start

    print()
    print("====================================================================================================")
    print(f'Total successful xml downloads: {total_successful_extractions}')
    print(f'Total failed xml downloads: {total_failed_extractions}')
    print(f'Total execution time in seconds: {total_execution_time}')
