#!/usr/bin/env python

from optparse import OptionParser
import tqdm
import glob
import os

#import ntpath

import clip_image
cancer_site_set = {"COAD", "UCEC"}

source_dir = "../data/Slide"
for cancer_site in cancer_site_set:
    for stage in ["train", "val"]:
        for file_path in tqdm.tqdm(glob.glob(source_dir + "/" + stage + "/" + cancer_site + "/*.svs")):
            out_dir = "../data/Tile/" + stage + "/" + cancer_site
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            clip_image.generate_tile(file_path, cancer_site, out_dir)
