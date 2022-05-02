#!/bin/bash
CITIES=('amsterdam' 'cdmx' 'chicago' 'columbus' 'la-piedad' 'newberg' 'oradell' 'pittsburgh' 'seattle' 'spgg')

for city in "${CITIES[@]}"
do
    echo ./scrape_and_crop_labels.py pano-downloads/ crops/ "$city"
    echo Finished "$city"
done
