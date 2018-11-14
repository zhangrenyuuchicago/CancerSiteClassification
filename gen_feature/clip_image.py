import openslide
import PIL
import sys
import numpy as np
import os
import ntpath


def generate_tile(slidefile, cancer_site, output_folder):
    # grid size in 20X magnification slide; double it when magnification is 40X
    grid_size = 1000
    # white cutoff; when < hi_cutoff, keep it 
    hi_cutoff = 0.85
    # keep only those patches within $inter_intensity$ of grayscale intensity distribution
    inter_intensity = 1.0
    # resize size; this size is used by Inception-V3
    re_size = (299, 299)

    if output_folder.endswith('/'):
        output_folder = output_folder[:-1]

    slide = openslide.open_slide(slidefile)
    level = slide.level_count - 1
    width, height = slide.level_dimensions[level]

    image = slide.read_region((0, 0), level, (width, height))
    img_gray = image.convert(mode="L")

    # region of interest; 
    # currently set all the slide as ROI
    ROI = []
    ROI.append([0, width, 0, height])

    top_width, top_height = slide.level_dimensions[0]
    thb_width, thb_height = slide.level_dimensions[level]

    # chop each ROI into 299x299 squares
    if slide.properties['openslide.objective-power'] == '40':
        size = (2*grid_size,2*grid_size)
    elif slide.properties['openslide.objective-power'] == '20':
        size = (grid_size,grid_size)
    else:
        print("The slide objective-powder")
        print(slide.properties['openslide.objective-power'])
        print('This magnification is not specified here')
        sys.exit(0)

    sc_ratio =  top_width * 1.0 / thb_width  # scale ratio between full and thumb images
    sc_size = int(size[0] / sc_ratio)

    XX = []
    YY = []

    for c in range(len(ROI)):
        XX.append(np.arange(ROI[c][0],ROI[c][1],sc_size))
        YY.append(np.arange(ROI[c][2],ROI[c][3],sc_size))

    intensities = [] # 1D list of normalized intensities
    xy_patches = [] # 1D list of all xy pairs

    img_gray = np.array(img_gray).T.astype(np.float)/256

    for c in range(len(ROI)):
        i_temp = [] # intensities will be concatenated across all x and y into a 1D list
        xy_temp = []
        for x in range(len(XX[c])-1):
            for y in range(len(YY[c])-1):
                npix = (int(YY[c][y+1]) - int(YY[c][y]))*(int(XX[c][x+1]) - int(XX[c][x]))
                i_temp.append(np.sum(img_gray[int(XX[c][x]):int(XX[c][x+1]),int(YY[c][y]):int(YY[c][y+1])])/npix)
                xy_temp.append([XX[c][x],YY[c][y]])
        intensities.append(np.array(i_temp))
        xy_patches.append(np.array(xy_temp))

    xy_keep = [] # this will store which xy pairs to keep based on intensity thresholding

    lower_bound = int(100 * (1 - inter_intensity) / 2)
    upper_bound = 100 - lower_bound

    for c in range(len(ROI)):
        xy_keep_temp = xy_patches[c][intensities[c] < hi_cutoff] # throw away anything > 0.9
        if len(xy_keep_temp) == 0:
            continue
        isubset = intensities[c][intensities[c] < hi_cutoff]
        lobound = np.percentile(isubset,lower_bound)
        hibound = np.percentile(isubset,upper_bound)
        xy_keep_temp = [xy_keep_temp[i] for i in np.where(np.logical_and(isubset >= lobound, isubset <= hibound))[0]]
        xy_keep.append(xy_keep_temp)

    file_name = ntpath.basename(slidefile)
    base_name = os.path.splitext(file_name)[0]

    for c in range(len(xy_keep)):
        for i in range(len(xy_keep[c])):
            x_pos, y_pos = sc_ratio * xy_keep[c][i]
            x_pos = int(x_pos)
            y_pos = int(y_pos)
            crop_image = slide.read_region((x_pos, y_pos), 0, size) 
            outfile_name = output_folder + "/" +  base_name + "_site_" + cancer_site + "_t" + str(c) + "_xpos" + str(x_pos) + "_ypos" + str(y_pos) + ".png"
            crop_image = crop_image.resize(re_size, resample=PIL.Image.ANTIALIAS)
            crop_image.save( outfile_name )
        

