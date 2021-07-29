# This file generates a .lst file for truth and reco xmls.
# .lst is needed to run the evaluation.
# All files need to be inside the reco_xml and truth_xml folders.

import os

#This needs to be done for Training and Validation
truth_path_xml = os.getcwd() + '\\evaluation\\truth_xml' # Source Folder reco
dstpath = os.getcwd()

# Folder won't used
files = os.listdir(truth_path_xml)

reco_list = []

with open("evaluation/reco_xml.lst", 'wb') as f:
    for xml in files:
        xml = ("./reco_xml/" + xml + "\n")
        f.write(xml.encode('utf-8'))

with open("evaluation/truth_xml.lst", 'wb') as f:
    for xml in files:
        xml = ("./truth_xml/" + xml + "\n")
        f.write(xml.encode('utf-8'))


