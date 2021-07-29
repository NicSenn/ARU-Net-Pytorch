import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#from wandb.env import MODE
from model import create_aru_net
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    evaluate,
)
#import wandb

# Params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
BATCH_SIZE = 1
NUM_EPOCHS = 250
NUM_WORKERS = 0
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
RANDOM_DOWNSAMPLING = False
PIN_MEMORY = True
LOAD_MODEL = False

# ARU-NET Params
SCALE_SPACE_NUM = 6
RES_DEPTH = 3
FEAT_ROOT = 8 # starting root for features
FILTER_SIZE = 3 # size of kernel
POOL_SIZE = 2 # size of pooling
ACTIVATION_NAME = "relu" # choose "relu" or "elu"
MODEL = "aru" # choose "aru", "ru", or "u"
NUM_SCALES = 5 # amount of scaled images you want to use you (e.g. 3: original image and two downscaled versions);

# Dataset paths
TRAIN_IMG_DIR = "data/cbad_2017_less_val/cbad_2017_mixed_train_images/"
TRAIN_MASK_DIR = "data/cbad_2017_less_val/cbad_2017_mixed_train_masks/"
VAL_IMG_DIR = "data/cbad_2017_less_val/cbad_2017_simple_val_images/"
VAL_MASK_DIR = "data/cbad_2017_less_val/cbad_2017_simple_val_masks/"
SAVED_IMAGES = "saved_images/"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"

# Evaluation (only works without random downsampling)
EVALUATE = True # if you want to evaluate with Transkribus after every epoch
EVALUATION_DIR = "./evaluation/" # folder with Transkribus jar (trans.jar)
VAL_XML_DIR = "data/cbad_2017_less_val/cbad_2017_simple_val_xml/" # ground truth of validation images
OUTPUT_XML_DIR = "saved_images_xml/"


# trains one epoch
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) # progress bar

    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast(enabled=True):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log wandb
        #wandb.log({"loss": loss})

        # update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():

    print("Device: " + DEVICE)
    # config for W&B
    # config = dict(
    #     # Training information
    #     train_img_dir = TRAIN_IMG_DIR,
    #     train_mask_dir = TRAIN_MASK_DIR,
    #     val_img_dir = VAL_IMG_DIR,
    #     val_mask_dir = VAL_MASK_DIR,

    #     # HYPER PARAMS
    #     batch_size = BATCH_SIZE,
    #     num_epochs = NUM_EPOCHS,
    #     image_height = IMAGE_HEIGHT,
    #     image_width = IMAGE_WIDTH,
    #     learning_rate = LEARNING_RATE,

    #     # ARU-NET PARAMS
    #     scale_space_num = SCALE_SPACE_NUM,
    #     res_depth = RES_DEPTH,
    #     feat_root = FEAT_ROOT,
    #     filter_size = FILTER_SIZE,
    #     pool_size = POOL_SIZE,
    #     activation_name = ACTIVATION_NAME,
    #     model = MODEL,
    #     num_scales = NUM_SCALES,

    # )

    # Configure wandb (Weights & Biases)
    #wandb.init(project='aru-self', config = config)

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
    # detect errors
    torch.autograd.set_detect_anomaly(True)

    model = create_aru_net(in_channels=1, out_channels=1, model_kwargs = model_kwargs).to(DEVICE)

    # using this because we didnt use torch.sigmoid() on self.final_conv(x)
    # with multiple out channels, use cross entropy loss instead
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        RANDOM_DOWNSAMPLING,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    # wandb watch
    #wandb.watch(model, log="all")

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    best_dice_score = 0.0
    best_f_measure = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print("Epoch: " + str(epoch) + "/" + str(NUM_EPOCHS))
        #wandb.log({"epoch": epoch})
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=CHECKPOINT_PATH)

        # check accuracy
        current_dice_score = check_accuracy(val_loader, model, device=DEVICE)
        if best_dice_score < current_dice_score:
            best_dice_score = current_dice_score
            print("New best dice_score: " + str(best_dice_score))
            save_checkpoint(checkpoint, filename=(CHECKPOINT_PATH[:-8] + "_best_dice_score.pth.tar"))

        # print some examples to a folder and evaluate
        if epoch != 0 and epoch % 10 == 0:
            if EVALUATE:
                save_predictions_as_imgs(val_loader, model, output_dir=SAVED_IMAGES, output_xml_dir = OUTPUT_XML_DIR, device=DEVICE, evaluate=True, batch_size = BATCH_SIZE)
                evaluation = evaluate(predicted_xml_dir=OUTPUT_XML_DIR, truth_xml_dir=VAL_XML_DIR, evaluation_dir=EVALUATION_DIR)
                #wandb.log(evaluation)
                if best_f_measure < evaluation['f_measure']:
                    best_f_measure = evaluation['f_measure']
                    print("New best F-Score: " + str(best_f_measure))
                    save_checkpoint(checkpoint, filename=(CHECKPOINT_PATH[:-8] + "_best_f_score.pth.tar"))
            else:
                save_predictions_as_imgs(val_loader, model, output_dir=SAVED_IMAGES, device=DEVICE, evaluate=False, batch_size = BATCH_SIZE)
        print("Best dice score yet: " + str(best_dice_score))
        if EVALUATE:
            print("Best F-Score yet: " + str(best_f_measure))

if __name__ == "__main__":
    main()