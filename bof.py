# -*- coding: utf-8 -*-
# This script generates bof vocabulary and database
# Created on June 14th, 2020

import pickle
from imagesearch import vocabulary
from tools.imtools import get_imlist
from localdescriptors import sift
from imagesearch import imagesearch
from sqlite3 import dbapi2 as sqlite

IMAGE_PATH = 'ukbench500/'

# get image list
image_list = get_imlist(IMAGE_PATH)
images_num = len(image_list)
# get feature
feature_list = [image_list[i][:-3] + 'sift' for i in range(images_num)]


class Bof:
    def sift_process(self):
        for i in range(images_num):
            sift.process_image(image_list[i], feature_list[i])

    def get_vocabulary(self):
        # generate vocabularies
        voc = vocabulary.Vocabulary('ukbenchtest')
        voc.train(feature_list, 1000, 10)

        # saving vocabulary
        with open('vocabulary_bof.pkl', 'wb') as f:
            pickle.dump(voc, f)
        print('vocabulary is:', voc.name, voc.nbr_words)

    def get_database(self):
        # load vocabulary
        with open('vocabulary_bof.pkl', 'rb') as f:
            voc = pickle.load(f)
        # set index
        index = imagesearch.Indexer('index_bof.db', voc)
        index.create_tables()
        # project features on vocabulary
        for i in range(images_num)[:1000]:
            locs, descr = sift.read_features_from_file(feature_list[i])
            index.add_to_index(image_list[i], descr)
        # commit to database
        index.db_commit()
        con = sqlite.connect('index_bof.db')
        print(con.execute('select count (filename) from imlist').fetchone())
        print(con.execute('select * from imlist').fetchone())

    def generate_bof_dict(self):
        self.sift_process()
        self.get_vocabulary()
        self.get_database()


if __name__ == '__main__':
   bof = Bof()
   bof.generate_bof_dict()


