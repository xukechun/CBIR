# CBIR

## Introduction

- This is a project of image retrieval for course CVML2020 in ZJU/CAML2020 in Cambridge.
- complement three methods（BOF/VLAD/BOF_SVM） for image retrieval.

## Installation

- Python3

- You can quickly install/update these dependencies by running the following (replace `pip` with `pip3` for Python 3):

  ```
  pip install pickle sqlite3 opencv-python sklearn numpy scipy PCV PIL matplotlib argparse glob heapq itertools
  ```

## Instructions

1. generating vocabulary dictionaries

- run  `bof.py` to generate vocabulary dictionary for `BOF`
- run  `vlad.py` to generate vocabulary dictionary for `VLAD`
- run  `bof_svm.py` to generate vocabulary dictionary for `BOF_SVM`

2. testing with different methods

   User need to choose the ID of tested image`(BOF and VLAD within 500 while BOF_SVM within 300)`, number of images to retrieve and method to use`(0:BOF,1:VLAD,2:BOF_SVM)`  

   Then run the following:

   ```
   python Search_images.py -q [Index of a query image] -r [number of images to retrieve] -m [method to use]
   ```



