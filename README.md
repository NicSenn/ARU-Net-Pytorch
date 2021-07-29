# ARU-Net Pytorch
 
This framework includes the PyTorch version of the [ARU-Net](https://github.com/TobiasGruening/ARU-Net). Stage 2 is not implemented like in the paper but uses the code from [P2PaLA](https://github.com/lquirosd/P2PaLA) to generate the dedicated baseline XMLs.

## Installation
1. Use Python >= 3.8.8 and PyTorch >= 1.9
2. Install needed packages in requirements.txt using a virtual environment (e.g. conda).
3. Install https://github.com/CyberZHG/torch-same-pad (pip install git+https://github.com/CyberZHG/torch-same-pad.git).
4. Set PYTHONPATH to the project location if needed.

## Training
1. Put your training dataset inside the "/data" folder: Split images and ground truth into seperate folders. (e.g. data/dataset/training_images and data/dataset/training_masks). Make sure your image masks are binarized.
2. Set the training params inside train.py Verify that the correct paths are set. Evaluation after every epochs does only work without random downsampling.
3. Run train.py. After every epoch, results of validation images are saved into "/saved_images". If evaluation is enabled, xmls are saved in "/saved_images_xml".


## Testing / Inference 
1. Put your testing dataset inside the "/data" folder (e.g. data/dataset/testing_images).
2. Set the testing params and the correct model inside test.py. Example models are given inside "/models". Verify that the correct paths are set. Set EVALUATION to True if you want to evaluate the generated xmls with the ground truth.
3. Run test.py. Results are saved into "/saved_test_images". If evaluation is enabled, xmls are saved in "/saved_test_images_xml". Scores are printed out and saved inside an evaluation[...].txt file in the project folder.
