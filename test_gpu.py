import cv2 as cv
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import deploy_network as wing_net
import data_loader.data_loaders as module_data
import glob
import torch


def square_distance_centroid(points, norm_factor):
    points[0::2] *= norm_factor[0]
    points[1::2] *= norm_factor[1]
    pts = [np.array([x, y]) for x, y in zip(points[0::2], points[1::2])]
    centroid = np.array([sum(points[0::2])/len(pts), sum(points[1::2])/len(pts)])

    distances = np.array([math.sqrt(sum(x)) for x in ((pts[:]-centroid)**2)[:]])
    distances = distances**2
    metric = math.sqrt(sum(distances))
    return metric

def get_roi(coords, roi_width, max_size):
    roi = [coords[1]-roi_width, coords[1]+roi_width, coords[0]-roi_width, coords[0]+roi_width]
    roi[0] = max(0, roi[0])
    roi[1] = min(max_size[0], roi[1])
    roi[2] = max(0, roi[2])
    roi[3] = min(max_size[1], roi[3])
    roi_shape = [roi[1]-roi[0], roi[3]-roi[2]]
    return roi, roi_shape

def auto_canny(image, sigma=0.33):
    image = cv.GaussianBlur(image, (3, 3), 0)
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    print("median={}".format(v))
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    #edged = cv.Canny(image, lower, upper)
    edged = cv.Canny(image, 15, 40)

    # return the edged image
    return edged

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def binarize(image, dilations=2):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    th2 = cv.dilate(th2,kernel,iterations = 2)
    return th2

#Returns the max corner as x,y coord (not image coord)
def get_max_corner(binary_image, resize):
    gray = cv.resize(binary_image, resize)
    dst = cv.cornerHarris(gray, 40, 9, 0.04)
    gaussian = (gkern(resize[0], 3))
    dst = np.multiply(dst, gaussian)
    plt.imshow(dst, cmap='gray')
    plt.show()

    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(dst)
    return maxLoc


def show_roi_of_keypoints(image, kpts, roi_size=20):
    print("keypoints={}".format(kpts))
    print("kpts*img={}*{}".format(kpts, image.shape))
    kpts[0::2] = kpts[0::2]*image.shape[1]
    kpts[1::2] = kpts[1::2]*image.shape[0]
    print(kpts)
    roi_width = roi_size//2
    plt.imshow(image, cmap='gray')
    plt.scatter(kpts[0::2], kpts[1::2])
    plt.show()
    refined_x = []
    refined_y = []
    for i in range(0, len(kpts)//2, 1):
        x = int(kpts[2*i])
        y = int(kpts[2*i+1])

        rc, rs = get_roi([x,y], roi_width, image.shape[0:2])
        roi = image[rc[0]:rc[1], rc[2]:rc[3]] 
        print("ROI shape={}".format(rs))
        plt.imshow(roi, cmap='gray')
        plt.show()

        img = binarize(roi)
        resize = (roi_size, roi_size)
        pt = get_max_corner(img, resize)
        print("pt at {}".format(pt))
        print(rs)
        pt1 = pt * np.array([rs[1]/roi_size, rs[0]/roi_size])
        print("{}*{}={}".format(pt, np.array([rs[1]/roi_size, rs[0]/roi_size]), pt1))
        refined_x.append(pt1[1]-roi_width+x)
        refined_y.append(pt1[0]-roi_width+y)
    plt.imshow(image, cmap='gray')
    plt.scatter(refined_x, refined_y)
    plt.show()




def test_data_loader(image_list, BATCH_SIZE=8, RESIZE=(256, 256)):
    model = wing_net.WingKeypointsGenerator(image_list)
    data_loader = module_data.WingsInferenceDataLoader(image_list, BATCH_SIZE,
                                                       resize_dims=RESIZE, shuffle=False,
                                                       validation_split=0.0, num_workers=2)
    for batch_idx, (data, names) in enumerate(data_loader):
        print(data.shape)
        print(len(names))

def test_forward(image_list, model_path):
    print("test GPU cuda:{}".format(torch.cuda.is_available()))
    # print(wings)
    model = wing_net.WingKeypointsGenerator(image_list, model_path=model_path)
    out = model.process_images()
    return out
    # print(model.model)

if __name__ == "__main__":
    model_path = '/home/timo/Data2/wingNet/wingNet_models/wings_resnet34_weights'
    wing_path = '/home/timo/Data2/wingNet/wings/No_TPS/avi_wings/0_wings'
    wings = sorted(glob.glob("{}/*.jpg".format(wing_path)))

#    kpts = test_forward(wings, model_path)
    w1_kpts = np.array([0.65600586, 0.33804876, 0.67442197, 0.37681636, 0.5224095 ,
       0.5495855 , 0.56550914, 0.6327244 , 0.5132236 , 0.84042585,
       0.1004951 , 0.88142204, 0.02135012, 0.7850952 , 0.12156321,
       0.4940002 ])
    image = cv.imread(wings[0])
#    print("raw_pts={}".format(kpts[0]))
    print("wings={}".format(wings[0]))
#    show_roi_of_keypoints(image, kpts[0][1], roi_size=200)
    show_roi_of_keypoints(image, w1_kpts, roi_size=250)
