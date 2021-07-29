import os
from glob import glob
import cv2
import numpy as np
from typing import Tuple
from imageio import imread, imsave
from tqdm import tqdm
from typing import Tuple
import PAGE

# Converts the ground truth of baseline detection datasets (XMLs) to image masks and binarizes them
# Based on https://github.com/dhlab-epfl/dhSegment

# Constant definitions
TARGET_HEIGHT = 1100
DRAWING_COLOR_BASELINES = (255, 255, 255) # somehow white doesnt work; does not matter because we binarize it afterwards anyway
DRAWING_COLOR_LINES = (0, 255, 0)
DRAWING_COLOR_POINTS = (0, 0, 255)

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# Folders 
IMAGE_FOLDER = '/cbad_2017_complex_val_images/'
MASK_FOLDER = '/cbad_2017_complex_val_masks/' # =output folder
XML_FOLDER = 'cbad_2017_complex_val_xml' #without "/"

def annotate_one_page(image_filename: str,
                      output_dir: str,
                      output_image_dir: str,
                      size: int=None,
                      draw_baselines: bool=True,
                      draw_lines: bool=False,
                      draw_endpoints: bool=False,
                      baseline_thickness: float=0.2,
                      diameter_endpoint: int=20) -> Tuple[str, str]:
    """
    Creates an annotated mask and corresponding original image and saves it in 'labels' and 'images' folders.
    Also copies the corresponding .xml file into 'gt' folder.

    :param image_filename: filename of the image to process
    :param output_dir: directory to output the annotated label image
    :param size: Size of the resized image (# pixels)
    :param draw_baselines: Draws the baselines (boolean)
    :param draw_lines: Draws the polygon's lines (boolean)
    :param draw_endpoints: Predict beginning and end of baselines (True, False)
    :param baseline_thickness: Thickness of annotated baseline (percentage of the line's height)
    :param diameter_endpoint: Diameter of annotated start/end points
    :return: (output_image_path, output_label_path)
    """
    page_filename = get_page_filename(image_filename)
    # Parse xml file and get TextLines
    page = PAGE.parse_file(page_filename)
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    img = imread(image_filename, pilmode='RGB')
    # Create empty mask
    gt = np.zeros_like(img)

    if text_lines:
        if draw_baselines:
            # Thickness : should be a percentage of the line height, for example 0.2
            # First, get the mean line height.
            mean_line_height, _, _ = _compute_statistics_line_height(page)
            absolute_baseline_thickness = int(max(gt.shape[0]*0.002, baseline_thickness*mean_line_height))

            # Draw the baselines
            gt_baselines = np.zeros_like(img[:, :, 0])
            gt_baselines = cv2.polylines(gt_baselines,
                                         [PAGE.Point.list_to_cv2poly(tl.baseline) for tl in
                                          text_lines],
                                         isClosed=False, color=255,
                                         thickness=absolute_baseline_thickness)
            gt[:, :, np.argmax(DRAWING_COLOR_BASELINES)] = gt_baselines

        if draw_lines:
            # Draw the lines
            gt_lines = np.zeros_like(img[:, :, 0])
            for tl in text_lines:
                gt_lines = cv2.fillPoly(gt_lines,
                                        [PAGE.Point.list_to_cv2poly(tl.coords)],
                                        color=255)
            gt[:, :, np.argmax(DRAWING_COLOR_LINES)] = gt_lines

        if draw_endpoints:
            # Draw endpoints of baselines
            gt_points = np.zeros_like(img[:, :, 0])
            for tl in text_lines:
                try:
                    gt_points = cv2.circle(gt_points, (tl.baseline[0].x, tl.baseline[0].y),
                                           radius=int((diameter_endpoint / 2 * (gt_points.shape[0] / TARGET_HEIGHT))),
                                           color=255, thickness=-1)
                    gt_points = cv2.circle(gt_points, (tl.baseline[-1].x, tl.baseline[-1].y),
                                           radius=int((diameter_endpoint / 2 * (gt_points.shape[0] / TARGET_HEIGHT))),
                                           color=255, thickness=-1)
                except IndexError:
                    print('Length of baseline is {}'.format(len(tl.baseline)))
            gt[:, :, np.argmax(DRAWING_COLOR_POINTS)] = gt_points

    # Make output filenames
    image_label_basename = get_image_label_basename(image_filename)
    output_image_path = os.path.join(output_image_dir, '{}.jpg'.format(image_label_basename))
    output_label_path = os.path.join(output_dir, '', '{}.png'.format(image_label_basename))
    # Resize (if necessary) and save image and label
    save_and_resize(img, output_image_path, size=size)
    save_and_resize(gt, output_label_path, size=size, nearest=True)
    # Copy XML file to 'gt' folder
    #shutil.copy(page_filename, os.path.join(output_dir, 'gt', '{}.xml'.format(image_label_basename)))

    #return os.path.abspath(output_image_path), os.path.abspath(output_label_path)

def get_image_label_basename(image_filename: str) -> str:
    """
    Creates a new filename composed of the begining of the folder/collection (ex. EPFL, ABP) and the original filename

    :param image_filename: path of the image filename
    :return:
    """
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    acronym = directory.split(os.path.sep)[-1].split('_')[0]
    return '{}{}'.format('', basename.split('.')[0])

def get_page_filename(image_filename: str) -> str:
    """
    Given an path to a .jpg or .png file, get the corresponding .xml file.

    :param image_filename: filename of the image
    :return: the filename of the corresponding .xml file, raises exception if .xml file does not exist
    """
    page_filename = os.path.join(os.path.dirname(os.path.dirname(image_filename)),
                                 XML_FOLDER,
                                 '{}.xml'.format(os.path.basename(image_filename)[:-4]))

    if os.path.exists(page_filename):
        return page_filename
    else:
        raise FileNotFoundError

def _compute_statistics_line_height(page_class: PAGE.Page, verbose: bool=False) -> Tuple[float, float, float]:
    """
    Function to compute mean and std of line height in a page.

    :param page_class: PAGE.Page object
    :param verbose: either to print computational info or not
    :return: tuple (mean, standard deviation, median)
    """
    y_lines_coords = [[c.y for c in tl.coords] for tr in page_class.text_regions for tl in tr.text_lines if tl.coords]
    line_heights = np.array([np.max(y_line_coord) - np.min(y_line_coord) for y_line_coord in y_lines_coords])

    # Remove outliers
    if len(line_heights) > 3:
        outliers = _is_outlier(np.array(line_heights))
        line_heights_filtered = line_heights[~outliers]
    else:
        line_heights_filtered = line_heights
    if verbose:
        print('Considering {}/{} lines to compute line height statistics'.format(len(line_heights_filtered),
                                                                                 len(line_heights)))

    # Compute mean, std, median
    mean = np.mean(line_heights_filtered)
    median = np.median(line_heights_filtered)
    standard_deviation = np.std(line_heights_filtered)

    return mean, standard_deviation, median


def save_and_resize(img: np.array,
                    filename: str,
                    size=None,
                    nearest: bool=False) -> None:
    """
    Resizes the image if necessary and saves it. The resizing will keep the image ratio

    :param img: the image to resize and save (numpy array)
    :param filename: filename of the saved image
    :param size: size of the image after resizing (in pixels). The ratio of the original image will be kept
    :param nearest: whether to use nearest interpolation method (default to False)
    :return:
    """
    if size is not None:
        h, w = img.shape[:2]
        ratio = float(np.sqrt(size/(h*w)))
        resized = cv2.resize(img, (int(w*ratio), int(h*ratio)),
                             interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        imsave(filename, resized)
    else:
        # b,g,r = cv2.split(img)
        # img = cv2.merge((b,g,r))
        imsave(filename, img)
    
def save_and_resize2(img: np.array,
                filename: str,
                size=None,
                nearest: bool=False) -> None:
    """
    Resizes the image if necessary and saves it. The resizing will keep the image ratio

    :param img: the image to resize and save (numpy array)
    :param filename: filename of the saved image
    :param size: size as tuple. exp (1536, 1536)
    :param nearest: whether to use nearest interpolation method (default to False)
    :return:
    """
    if size is not None:
        resized = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        imsave(filename, resized)
    else:
        # b,g,r = cv2.split(img)
        # img = cv2.merge((b,g,r))
        imsave(filename, img)

        

def _is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise. Used to find outliers in 1D data.
    https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data

    References:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    :param points : An numobservations by numdimensions array of observations
    :param thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    :return: mask : A num_observations-length boolean array.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    # Replace zero values by epsilon
    if not isinstance(med_abs_deviation, float):
        med_abs_deviation = np.maximum(med_abs_deviation, len(med_abs_deviation)*[1e-10])
    else:
        med_abs_deviation = np.maximum(med_abs_deviation, 1e-10)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


if __name__ == "__main__":

    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    input_folder_path = my_absolute_dirpath + IMAGE_FOLDER
    dirs = os.listdir(input_folder_path)

    output_folder_path = my_absolute_dirpath + IMAGE_FOLDER
    print("Starting conversion from XML to Image mask")
    for file in tqdm(dirs):
        annotate_one_page(image_filename=input_folder_path + file, output_dir=my_absolute_dirpath + MASK_FOLDER, output_image_dir=input_folder_path)
    
    # binarize
    print("Starting binarizing")
    dirs = os.listdir(my_absolute_dirpath + MASK_FOLDER + '/')
    for file in tqdm(dirs):
        if file.endswith('.png'):
            im_gray = cv2.imread(my_absolute_dirpath + MASK_FOLDER + file, 0)
            im_bw = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY)[1]
            #save image
            if not cv2.imwrite(my_absolute_dirpath + MASK_FOLDER + file, im_bw):
                raise Exception("Could not write image")


