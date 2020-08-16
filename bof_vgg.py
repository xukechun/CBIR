from cnn_cbir import SearchEngine, FeatureExtractor
from utils import load_image_and_bbs, draw_bbs_to_img
import cv2
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

data_dir = '/home/xukechun/data1/xukechun/pg_data/Images'
query_dir = '/home/xukechun/data1/xukechun/pg_data/Queries'


def create_SearchEngine():
    fea_extractor = FeatureExtractor(cache_dir='largedata_cache')
    search_engine = SearchEngine(data_dir, fea_extractor)
    search_engine.build()
    return search_engine


def catch_BoundingBox(q_ind):
    im_bb_path_tuples = [(os.path.join(query_dir, '{:02d}.jpg'.format(i)),
                          os.path.join(query_dir, '{:02d}.txt'.format(i))
                          ) for i in range(1, 51)
                         ]
    demo_im_path, demo_bb_path = im_bb_path_tuples[q_ind]
    print(demo_im_path)

    demo_im, demo_bb = load_image_and_bbs(demo_im_path, demo_bb_path)
    draw_bbs_to_img(demo_im, demo_bb)
    return demo_im, demo_bb


def retrieve_object(search_engine, demo_im, demo_bb, nbr_results):
    similar_imgs = search_engine.retrieve_object(demo_im, demo_bb, top_k=nbr_results)
    match_scores = []
    match_images = []
    # match_img_paths = []
    for img_path, score, bbs in similar_imgs:
        print(score, img_path)
        match_scores.append(score)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        match_images.append(img)
        # match_img_paths.append(img_path)
        # draw_bbs_to_img(img, bbs, 'ijhw_mat')
    # show the top match images
    plt.figure()
    plt.suptitle('Search Results with BOF_VGG')
    nbr_results = len(match_images)
    ax = plt.subplot(3, 3, 1)
    plt.imshow(demo_im)
    plt.xticks([])
    plt.yticks([])
    ax.set_title('queried image', fontsize=10)
    for i in range(nbr_results):
        ax = plt.subplot(3, 3, i + 4)
        plt.imshow(match_images[i])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(str(match_scores[i])[0:5], fontsize=10)
    plt.show()


if __name__ == '__main__':
    search_engine = create_SearchEngine()
    # demo_im, demo_bb = catch_BoundingBox(13)
    # retrieve_object(search_engine, demo_im, demo_bb, 6)
