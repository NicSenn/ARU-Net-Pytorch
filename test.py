import torch
from model import create_aru_net
from utils import(
    evaluate,
    load_checkpoint,
    get_test_loaders,
    save_test_predictions_as_imgs,
)

# Params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
# Set the correct parameters according to the model
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PADDING = True # True when the model was trained with data augmentation / random downsampling
PIN_MEMORY = True

# ARU-NET Params
SCALE_SPACE_NUM = 6
RES_DEPTH = 3
FEAT_ROOT = 8 # starting root for features
FILTER_SIZE = 3 # size of kernel
POOL_SIZE = 2 # size of pooling
ACTIVATION_NAME = "relu" # choose "relu" or "elu"
MODEL = "aru" # choose "aru", "ru", or "u"
NUM_SCALES = 5 # amount of scaled images you want to use you (e.g. 3: original image and two downscaled versions)

# Model
CHECKPOINT = "models\cbad_2019.tar"

# Paths
TEST_IMG_DIR = "data\cbad_2017_less_val/cbad_2017_simple_val_images/"
TEST_OUTPUT_DIR = "saved_test_images/"
TEST_OUTPUT_XML_DIR = "saved_test_images_xml/"

# Evaluation 
EVALUATE = True # if you want to evaluate with Transkribus Baseline Evaluation after inference
EVALUATION_DIR = "./evaluation/" # folder with Transkribus jar (trans.jar)
TRUTH_XML_DIR = "data\cbad_2017_less_val/cbad_2017_simple_val_xml/" # ground truth of test images

def main():

    model_kwargs = dict(
        scale_space_num = SCALE_SPACE_NUM,
        res_depth = RES_DEPTH,
        feat_root = FEAT_ROOT,
        filter_size = FILTER_SIZE,
        pool_size = POOL_SIZE,
        activation_name = ACTIVATION_NAME,
        model = MODEL,
        num_scales = NUM_SCALES,
    )

    model = create_aru_net(in_channels = 1, out_channels=1, model_kwargs = model_kwargs).to(DEVICE)
    test_loader = get_test_loaders(TEST_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, PADDING, NUM_WORKERS, PIN_MEMORY)
    load_checkpoint(torch.load(CHECKPOINT), model)

    save_test_predictions_as_imgs(test_loader, model, image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH, padding = PADDING, output_dir=TEST_OUTPUT_DIR, device=DEVICE)
    if EVALUATE:
        evaluate(predicted_xml_dir=TEST_OUTPUT_XML_DIR, truth_xml_dir=TRUTH_XML_DIR, evaluation_dir = EVALUATION_DIR)




if __name__ == "__main__":
    main()