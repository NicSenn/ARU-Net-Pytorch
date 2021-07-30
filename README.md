# ARU-Net Pytorch
 
This framework includes the PyTorch version of the [ARU-Net](https://github.com/TobiasGruening/ARU-Net). Stage 2 is not implemented like in the paper but uses the code from [P2PaLA](https://github.com/lquirosd/P2PaLA) to generate the dedicated baseline XMLs.

## Installation
1. Use Python >= 3.8.8 and PyTorch >= 1.9
2. Install needed packages in requirements.txt using a virtual environment (e.g. conda). If there are problems installing "albumentations" while using conda, try "conda install -c conda-forge albumentations". 
3. Install https://github.com/CyberZHG/torch-same-pad (pip install git+https://github.com/CyberZHG/torch-same-pad.git).
4. If needed, set the PYTHONPATH to the project location.

## Training
1. Put your training dataset inside the "/data" folder: Split images and ground truth of the training and validation part into seperate folders. (e.g. data/dataset/training_images and data/dataset/training_masks; same for validation). Make sure your image masks are binarized.
2. Set the training params on the top of train.py. Verify that the correct paths are set. Evaluation after every epochs does only work without random downsampling.
3. Run train.py. After every epoch, results of validation images are saved into "/saved_images". If evaluation is enabled, xmls are saved in "/saved_images_xml".


## Testing / Inference 
1. Put your testing dataset inside the "/data" folder (e.g. data/dataset/testing_images). For optional evaluation, also add your xml ground truth to the "/data" folder (e.g. data/dataset/testing_xml).
2. Set the testing params and the correct model on the top of test.py. Example models and their correct params are given inside "/models". Verify that the correct paths are set. Set "EVALUATION" to True if you want to evaluate the generated xmls with the ground truth.
3. Run test.py. Images are saved into "/saved_test_images", xmls in "/saved_test_images_xml". Evaluation scores are printed out and saved inside an evaluation[...].txt file in the project folder.
