#!/usr/bin/env python

from optparse import OptionParser
import csv
import sys
import os
import tqdm
import glob
import ntpath
import clip_image
cancer_site_set = {"COAD", "UCEC"}

source_dir = "../data/Slide"
for cancer_site in cancer_site_set:
    for stage in ["train", "val"]:
        for file_path in tqdm.tqdm(glob.glob(source_dir + "/" + stage + "/" + cancer_site + "/*.svs")):
            #print(file_path)
            #comd = "python clip_image.py " + file_path + " " + cancer_site + " " + "../data/Tile/" + stage + "/" + cancer_site 
            #os.system(comd)
            out_dir = "../data/Tile/" + stage + "/" + cancer_site
            clip_image.generate_tile(file_path, cancer_site, out_dir)
