# Preparing Data
Our architecture trains and evaluates on `CROP_WIDTH x CROP_HEIGHT` crops of Google Street View (GSV) panoramas. Currently, `CROP_WIDTH = CROP_HEIGHT = 1500` as defined in `CropRunner.py`.

To create the data, we provide the `scrape_and_crop_labels.py` script, which, at a high level, downloads relevant panoramas from the Makeability Lab SFTP server to take crops from and then proceeds to create the crops given specific label metadata.

## Using `scrape_and_crop_labels.py`
The `scrape_and_crop_labels.py` script will make crops for label metadata from a specified city provided in the `city` command line argument (i.e., `seattle`).

Label metadata should be structured as follows:
```    
[
    {
        "label_id":47614,
        "gsv_panorama_id":"sHMY67LdNX48BFwpbGMD3A",
        "label_type_id":2,
        "agree_count":1,
        "disagree_count":0,
        "notsure_count":0,
        "image_width":16384,
        "image_height":8192,
        "sv_image_x":6538,
        "sv_image_y":-731,
        "canvas_width":720,
        "canvas_height":480,
        "canvas_x":275,
        "canvas_y":152,
        "zoom":1,
        "heading":190.25,
        "pitch":-34.4375,
        "photographer_heading":292.4190368652344,
        "photographer_pitch":-3.3052749633789062
    },
    ...
 ]
 ```
There are two ways one can provide label metadata for the cropper.
1. (Recommended) With the `-d` flag, one can provide a FDQN of a SidewalkWebpage server (i.e., `sidewalk-sea.cs.washington.edu`) to fetch metadata from the `cvMetadata` endpoint. 
2. With the `-c` flag, one can provide a relative path to a CSV that contains metadata structured as above per row. Such a file can be acquired from Mikey.

Note that using the `-d` flag will attempt to create crops for all applicable crops in that cities dataset. If only a subset is desired, one can uncomment [this line](https://github.com/michaelduan8/sidewalk-cv-2021/blob/df4db187f48b49eea3002259cafa58d564bc660e/scrape_and_crop_labels.py#L141) and specify a count. This method will still end up acquiring the metadata for all labels in the dataset. In the future, options such as paging maybe added to the endpoint for optimization.

### Downloading Panoramas
The script will download relevant panoramas for the label metadata from the city specified from the Makeability SFTP server. As a result, one will need to provide a path the SFTP key in the `Panoscraper.py` file [here](https://github.com/michaelduan8/sidewalk-cv-2021/blob/df4db187f48b49eea3002259cafa58d564bc660e/PanoScraper.py#L20). One can acquire the key by asking Mikey.

Furthermore, since panoramas will be downloaded to disk, metadata for crops are processed in batches. By default, after each batch of crops is created, downloaded panoramas will be deleted to make room for panoramas in the next batch. If you'd like to keep all downloaded panoramas, simply comment out [this block of code](https://github.com/michaelduan8/sidewalk-cv-2021/blob/df4db187f48b49eea3002259cafa58d564bc660e/scrape_and_crop_labels.py#L254)

### Crop Inaccuracies
Currently, the 'x, y' coordinates of labels on the 2d GSV panoramas are reconstructed approximately from the 3D fov coordinates provided by GSV metadata. As a result, crops may appear to be centered slightly offset. The refinement of the translation algorithm is a work in progress and may not be fully solveable given the metadata Google provides.
