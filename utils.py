import torch
import torchvision 
import torchvision.transforms as transforms
from dataset import NewDataset
from dataset_test import TestDataset
from torch.utils.data import DataLoader
import cv2
import random
from page_xml.xmlPAGE import pageData
from shapely.geometry import LineString
from additional_utils import polyapprox as pa
import numpy as np
import os
import PIL as PIL
import subprocess
import pandas as pd
import io
from tqdm import tqdm
import albumentations as A
#import wandb


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint " + filename)
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    random_downsampling,
    image_height,
    image_width,
    num_workers=4,
    pin_memory=True,
):
    train_ds = NewDataset(
       image_dir=train_dir,
        mask_dir=train_maskdir,
        random_downsampling=random_downsampling,
        image_height=image_height,
        image_width=image_width
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = NewDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        random_downsampling=random_downsampling,
        image_height=image_height,
        image_width=image_width
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def get_test_loaders(
    test_dir,
    image_height,
    image_width,
    padding,
    num_workers=4,
    pin_memory=True,
):

    test_ds = TestDataset(
        image_dir=test_dir,
        image_height=image_height,
        image_width = image_width,
        padding=padding,
    )

    test_loader = DataLoader(
        test_ds,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return test_loader

def check_accuracy(loader, model, device="cuda"):
    """Calculates accuracy and dice score of validation images."""
    print("Checking accuracy..")
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # unsqueeze because label doesn't have a channel
            preds = torch.sigmoid(model(x)) # change this if you use more than one class
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds) # numel = number of elements
            # calculate the similarity (only binary)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            
        
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    #wandb.log({"dice_score": dice_score/len(loader)})
    model.train()
    return float(dice_score/len(loader))

def save_predictions_as_imgs(loader, model, batch_size, output_dir="saved_images/", output_xml_dir= "saved_test_images_xml/", device="cuda", evaluate = False):
    """Saves the current predictions of the training model as images. If random downsampling is disabled, evaluation can be run."""
    print("Saving predicted images..")
    model.eval()
    max = len(loader.dataset)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # In preds every image inside one batch is saved; for evaluation we need every prediction as a standalone image
        splitSize = 1
        if max - idx * batch_size >= batch_size:
            splitSize = batch_size
        else:
            splitSize = max - idx * batch_size

        preds = np.array_split(preds, splitSize)
        y = y.unsqueeze(1)
        y = torch.tensor_split(y, splitSize)
        # save predicted baselines as img
        for i in range(len(preds)):
            image_name = loader.dataset.images[idx * batch_size + i] # name of original image

            # delete data endings
            if str.endswith(image_name, '.jpg') or str.endswith(image_name, '.JPG') or str.endswith(image_name, '.png'):
                image_name = image_name[:len(image_name) - 4]
            elif str.endswith(image_name, '.jpeg'):
                image_name = image_name[:len(image_name) - 5]

            aImgPath = f"{output_dir}/{image_name}.png"
            torchvision.utils.save_image(
                preds[i], aImgPath
            )

            # original mask
            torchvision.utils.save_image(y[i], f"{output_dir}/{image_name}_original.png")
        
        # Generate XMLs for evaluation
        if evaluate:
            for i in range(len(preds)):
                # get size of original image
                image_name = loader.dataset.images[idx * batch_size + i]
                image_path = loader.dataset.image_dir + image_name
                width, height = PIL.Image.open(image_path).size

                # delete data endings
                if str.endswith(image_name, '.jpg') or str.endswith(image_name, '.JPG') or str.endswith(image_name, '.png'):
                    image_name = image_name[:len(image_name) - 4]
                elif str.endswith(image_name, '.jpeg'):
                    image_name = image_name[:len(image_name) - 5]

                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=(height, width)),
                    transforms.ToTensor()
                ])
            
                preds[i] = transform(preds[i][0])
            
                aImgPath_upscaled = f"{output_dir}/{image_name}_upscaled.png"
                torchvision.utils.save_image(preds[i], aImgPath_upscaled)
                # combine baselines with original image
                combineImages(original_image_path=image_path, baseline_image_path=aImgPath_upscaled, folder=output_dir, image_name=image_name)

                # generate xml
                gen_page(in_img_path=image_path, line_mask = preds[i][0,:,:], id=image_name, output_dir=output_xml_dir)
    model.train()

def save_test_predictions_as_imgs(loader, model, image_height, image_width, padding, output_dir ="saved_test_images/", output_xml_dir= "saved_test_images_xml/", device="cuda"):
    """Saves images predicted by the testing model. If the model has been trained with random downsampling, padding must be enabled."""
    model.eval()
    loop = tqdm(loader)
    for idx, (x) in tqdm(enumerate(loop)):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        image_name = loader.dataset.images[idx] # name of original image

        # get size of original image
        image_path = loader.dataset.image_dir + image_name
        original_image = PIL.Image.open(image_path)
        width, height = original_image.size

        # delete data endings
        if str.endswith(image_name, '.jpg') or str.endswith(image_name, '.JPG') or str.endswith(image_name, '.png'):
            image_name = image_name[:len(image_name) - 4]
        elif str.endswith(image_name, '.jpeg'):
            image_name = image_name[:len(image_name) - 5]

        aImgPath = f"{output_dir}/{image_name}.png"
        torchvision.utils.save_image(
            preds, aImgPath
        )
        # remove padding
        if padding:

            transform = transforms.Compose([
                transforms.ToPILImage(),
            ])

            preds = transform(preds[0])
            
            preds = np.array(preds)
            
            # downscale like in train to calculate padding and unpad afterwards
            max_size = max(image_height, image_width)
            resize = A.LongestMaxSize(max_size=max_size, p=1)
            downscaled_version = resize(image=np.array(original_image))
            downscaled_image = downscaled_version["image"]

            # calculated used padding
            maxH = image_height
            maxW = image_width
            padH = maxH - downscaled_image.shape[0]
            padW = maxW - downscaled_image.shape[1]

            # now unpad the created image
            preds = np.array(preds)
            preds = preds[0:image_height - padH, 0:image_width - padW]
            #preds = transforms.functional.crop(preds, top = 0, left = 0, height = padH, width=padW),
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(height, width)),
                transforms.ToTensor()
            ])

            preds = transform(preds)
        
        else:

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(height, width)),
                transforms.ToTensor()
            ])

            preds = transform(preds[0])

        aImgPath_upscaled = f"{output_dir}/{image_name}_upscaled.png"
        torchvision.utils.save_image(
            preds, aImgPath_upscaled
        )

        # combine baselines with original image
        combineImages(original_image_path=image_path, baseline_image_path=aImgPath_upscaled, folder=output_dir, image_name=image_name)

        # generate xml
        gen_page(in_img_path=image_path, line_mask = preds[0,:,:], id=image_name, output_dir=output_xml_dir)


    model.train()

def evaluate(predicted_xml_dir, truth_xml_dir, evaluation_dir):
    """Code from: https://github.com/dhlab-epfl/dhSegment"""
    cbad_jar = evaluation_dir + 'trans.jar'

    predicted_pages_list = os.path.join(evaluation_dir, 'predicted_xml.lst')
    truth_pages_list = os.path.join(evaluation_dir, 'truth_xml.lst')

    # Create .lst files containing paths to xmls
    files = os.listdir(predicted_xml_dir)
    with open(predicted_pages_list, 'wb') as f:
        for xml in files:
            xml = ("./" + predicted_xml_dir + xml + "\n")
            f.write(xml.encode('utf-8'))

    with open(truth_pages_list, 'wb') as f:
        for xml in files:
            xml = ("./" + truth_xml_dir + xml + "\n")
            f.write(xml.encode('utf-8'))

    # Evaluation with java tool
    cmd = 'java -jar {} {} {}'.format(cbad_jar, truth_pages_list, predicted_pages_list)
    result = subprocess.check_output(cmd, shell=True).decode()
    with open(os.path.join(evaluation_dir, 'scores.txt'), 'w') as f:
        f.write(result)
    parse_score_txt(result, os.path.join(evaluation_dir, 'scores.csv'))

    # Parse results from output of tool
    lines = result.splitlines()
    avg_precision = float(next(filter(lambda l: 'Avg (over pages) P value:' in l, lines)).split()[-1])
    avg_recall = float(next(filter(lambda l: 'Avg (over pages) R value:' in l, lines)).split()[-1])
    f_measure = float(next(filter(lambda l: 'Resulting F_1 value:' in l, lines)).split()[-1])

    print('P {}, R {}, F {}'.format(avg_precision, avg_recall, f_measure))
    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'f_measure': f_measure
    }

def parse_score_txt(score_txt: str, output_csv: str):
    """Code from: https://github.com/dhlab-epfl/dhSegment"""
    lines = score_txt.splitlines()
    header_ind = next((i for i, l in enumerate(lines)
                       if l == '#P value, #R value, #F_1 value, #TruthFileName, #HypoFileName'))
    final_line = next((i for i, l in enumerate(lines) if i > header_ind and l == ''))
    csv_data = '\n'.join(lines[header_ind:final_line])
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.rename(columns={k: k.strip() for k in df.columns})
    df['#HypoFileName'] = [os.path.basename(f).split('.')[0] for f in df['#HypoFileName']]
    del df['#TruthFileName']
    df = df.rename(columns={'#P value': 'P', '#R value': 'R', '#F_1 value': 'F_1', '#HypoFileName': 'basename'})
    df = df.reindex(columns=['basename', 'F_1', 'P', 'R'])
    df = df.sort_values('F_1', ascending=True)
    df.to_csv(output_csv, index=False)
    

def combineImages(original_image_path, baseline_image_path, folder, image_name):
    """Combine result image with input image so baselines are drawn on the original image."""

    aImgPath = original_image_path
    
    #get the original image
    input_image = cv2.imread(aImgPath)

    #get the output image (baselines)
    output_image = cv2.imread(baseline_image_path)

    #values between 0 and 1
    image_mask = output_image/255

    #get a red picture with the size of the output image
    color = np.zeros_like(output_image)
    color[:,:, 2] += 255

    combined_image =  (1-image_mask) * input_image + image_mask * color
    combined_image = np.uint8(combined_image)

    save_location = os.path.join(folder, image_name + '_combined.png')
    cv2.imwrite(save_location, combined_image)

def gen_page(in_img_path, line_mask, id, output_dir="saved_test_images_xml/"):
    """Code from: https://github.com/imagine5am/ARU-Net"""
    in_img = cv2.imread(in_img_path)
    (in_img_rows, in_img_cols, _) = in_img.shape
    # print('line_mask.shape:', line_mask.shape)
    
    cScale = np.array(
        [in_img_cols / line_mask.shape[1], in_img_rows / line_mask.shape[0]]
    )
    id = str(id)
    page = pageData(os.path.join(output_dir[:-1], id + ".xml"), creator="ARU-Net PyTorch") # remove "/" from output_dir
    page.new_page(os.path.basename(in_img_path), str(in_img_rows), str(in_img_cols))
    
    kernel = np.ones((5, 5), np.uint8)
    validValues = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    
    #lines = line_mask.copy()
    lines = torch.clone(line_mask)
    lines[line_mask > 0.1] = 1
    #lines = lines.astype(np.uint8)
    lines = lines.to(torch.uint8)
    lines = lines.cpu().numpy()
    
    # plt.axis("off")
    # plt.imshow(lines, cmap='gray')
    # plt.show()

    r_id = 0
    lin_mask = np.zeros(line_mask.shape, dtype="uint8")
    
    reg_mask = np.ones(line_mask.shape, dtype="uint8")
    res_ = cv2.findContours(
        np.uint8(reg_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(res_) == 2:
        contours, hierarchy = res_
    else:
        _, contours, hierarchy = res_
        
    for cnt in contours:
        min_area = 0.01
        # --- remove small objects
        if cnt.shape[0] < 4:
            continue
        if cv2.contourArea(cnt) < min_area * line_mask.shape[0]:
            continue

        rect = cv2.minAreaRect(cnt)
        # --- soft a bit the region to prevent spikes
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # box = np.array((rect[0][0], rect[0][1], rect[1][0], rect[1][1])).astype(int)
        r_id = r_id + 1
        approx = (approx * cScale).astype("int32")
        reg_coords = ""
        for x in approx.reshape(-1, 2):
            reg_coords = reg_coords + " {},{}".format(x[0], x[1])
            
        cv2.fillConvexPoly(lin_mask, points=cnt, color=(1, 1, 1))
        lin_mask = cv2.erode(lin_mask, kernel, iterations=1)
        lin_mask = cv2.dilate(lin_mask, kernel, iterations=1)
        reg_lines = lines * lin_mask
    
        resl_ = cv2.findContours(
            reg_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(resl_) == 2:
            l_cont, l_hier = resl_
        else:
            _, l_cont, l_hier = resl_
        
        # IMPORTANT l_cont, l_hier
        if len(l_cont) == 0:
            continue
            
        # --- Add region to XML only is there is some line
        uuid = ''.join(random.choice(validValues) for _ in range(4))
        text_reg = page.add_element(
            'TextRegion', "r" + uuid + "_" +str(r_id), 'full_page', reg_coords.strip()
        )
        n_lines = 0
        for l_id, l_cnt in enumerate(l_cont):
            if l_cnt.shape[0] < 4:
                continue
            if cv2.contourArea(l_cnt) < 0.01 * line_mask.shape[0]:
                continue
            # --- convert to convexHull if poly is not convex
            if not cv2.isContourConvex(l_cnt):
                l_cnt = cv2.convexHull(l_cnt)
            lin_coords = ""
            l_cnt = (l_cnt * cScale).astype("int32")
            # IMPORTANT
            (is_line, approx_lin) = get_baseline(in_img, l_cnt)
            
            if is_line == False:
                continue
            
            is_line, l_cnt = build_baseline_offset(approx_lin, offset=50)
            if is_line == False:
                continue
            for l_x in l_cnt.reshape(-1, 2):
                lin_coords = lin_coords + " {},{}".format(
                    l_x[0], l_x[1]
                )
            uuid = ''.join(random.choice(validValues) for _ in range(4))
            text_line = page.add_element(
                "TextLine",
                "l" + uuid + "_" + str(l_id),
                'full_page',
                lin_coords.strip(),
                parent=text_reg,
            )
            # IMPORTANT
            baseline = pa.points_to_str(approx_lin)
            page.add_baseline(baseline, text_line)
            n_lines += 1
    page.save_xml()

def get_baseline(Oimg, Lpoly):
    """Code from: https://github.com/imagine5am/ARU-Net"""
    # --- Oimg = image to find the line
    # --- Lpoly polygon where the line is expected to be
    minX = Lpoly[:, :, 0].min()
    maxX = Lpoly[:, :, 0].max()
    minY = Lpoly[:, :, 1].min()
    maxY = Lpoly[:, :, 1].max()
    mask = np.zeros(Oimg.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, Lpoly, (255, 255, 255))
    res = cv2.bitwise_and(Oimg, mask)
    bRes = Oimg[minY:maxY, minX:maxX]
    bMsk = mask[minY:maxY, minX:maxX]
    bRes = cv2.cvtColor(bRes, cv2.COLOR_RGB2GRAY)
    _, bImg = cv2.threshold(bRes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cols = bImg.shape
    # --- remove black halo around the image
    bImg[bMsk[:, :, 0] == 0] = 255
    Cs = np.cumsum(abs(bImg - 255), axis=0)
    maxPoints = np.argmax(Cs, axis=0)
    Lmsk = np.zeros(bImg.shape)
    points = np.zeros((cols, 2), dtype="int")
    # --- gen a 2D list of points
    for i, j in enumerate(maxPoints):
        points[i, :] = [i, j]
    # --- remove points at post 0, those are very probable to be blank columns
    points2D = points[points[:, 1] > 0]
    if points2D.size <= 15:
        # --- there is no real line
        return (False, [[0, 0]])
    
    # --- take only 100 points to build the baseline
    max_vertex = 30
    num_segments = 4
    if points2D.shape[0] > max_vertex:
        points2D = points2D[
            np.linspace(
                0, points2D.shape[0] - 1, max_vertex, dtype=np.int
            )
        ]
    (approxError, approxLin) = pa.poly_approx(
        points2D, num_segments, pa.one_axis_delta
    )
    
    approxLin[:, 0] = approxLin[:, 0] + minX
    approxLin[:, 1] = approxLin[:, 1] + minY
    return (True, approxLin)

def build_baseline_offset(baseline, offset=50):
    """
    build a simple polygon of width $offset around the
    provided baseline, 75% over the baseline and 25% below.
    Code from: https://github.com/imagine5am/ARU-Net
    """
    try:
        line = LineString(baseline)
        up_offset = line.parallel_offset(offset * 0.75, "right", join_style=2)
        bot_offset = line.parallel_offset(offset * 0.25, "left", join_style=2)
    except:
        #--- TODO: check if this baselines can be saved
        return False, None
    if (
        up_offset.type != "LineString"
        or up_offset.is_empty == True
        or bot_offset.type != "LineString"
        or bot_offset.is_empty == True
    ):
        return False, None
    else:
        up_offset = np.array(up_offset.coords).astype(np.int)
        bot_offset = np.array(bot_offset.coords).astype(np.int)
        return True, np.vstack((up_offset, bot_offset)) 


