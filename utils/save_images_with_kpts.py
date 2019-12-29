import sys
sys.path.append("..")
import wingnet as wn
import argparse
import os
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


def load_project(path):
    print("loading from {}".format(path))
    if path and os.path.exists(path):
        project = pd.read_pickle(path)
        for img_path in project["path"]:
            if not os.path.exists(img_path):
                print("Image {} does not exist!".format(img_path))
                return
        return project
    else:
        print("project does not exist: {}".format(path))


def load_and_save_images(in_path, output_path):
    project = load_project(in_path)
    print(project)
    for image_pth, kpts in zip(project['path'], project['keypoints']):
        image = cv.imread(image_pth)
        xs = [i * image.shape[1] for i in kpts[0::2]]
        ys = [i * image.shape[0] for i in kpts[1::2]]
        plt.imshow(image)
        plt.scatter(xs, ys, marker='x', c='r')
        plt.axis('off')
        plt.grid(b=None)
        # plt.show()
        fname = os.path.splitext(os.path.basename(image_pth))[0]
        print("save to {}/{}.jpg".format(output_path, fname))
        plt.savefig("{}/{}.jpg".format(output_path, fname))
        plt.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Save images and keypoints')
    args.add_argument('--path', default=None, type=str, help='path to project file')
    args.add_argument('-output_path', default=None, type=str, help='path to save images')

    args = args.parse_args()
    filepath = args.path
    output_path = args.output_path
    if output_path is None:
        # fname = os.path.splitext(filepath)[0]
        fname = os.path.splitext(os.path.basename(filepath))[0]
        output_path = '/tmp/out_{}'.format(fname)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    load_and_save_images(filepath, output_path)