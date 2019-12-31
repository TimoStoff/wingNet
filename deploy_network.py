import torch
import data_loader.data_loaders as module_data
import model.model as module_arch
import numpy as np
import pandas as pd


class WingKeypointsGenerator():

    def __init__(self, image_list, model_path="/home/timo/Data2/wingNet_models/wings_resnet34_weights"):
        print("initializing keypoint generator...")
        self.RESIZE = (256, 256)
        self.BATCH_SIZE = 2
        self.KPT_DIV = np.array([self.RESIZE[0], self.RESIZE[1], self.RESIZE[0], self.RESIZE[1],
                                 self.RESIZE[0], self.RESIZE[1], self.RESIZE[0], self.RESIZE[1],
                                 self.RESIZE[0], self.RESIZE[1], self.RESIZE[0], self.RESIZE[1],
                                 self.RESIZE[0], self.RESIZE[1], self.RESIZE[0], self.RESIZE[1]])
        self.use_gpu = True
        self.device = self.get_device(self.use_gpu)
        self.model = self.load_model(model_path)
        if self.use_gpu:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.image_list = image_list
        self.data_loader = module_data.WingsInferenceDataLoader(image_list, self.BATCH_SIZE, resize_dims=self.RESIZE,
                                                                shuffle=False, validation_split=0.0, num_workers=2)
        print('... done')

    def __del__(self):
        torch.cuda.empty_cache()

    def load_model(self, path_to_model):
        device = 'cuda:0' if (torch.cuda.is_available() and self.use_gpu) else 'cpu'
        print('Loading model to {}...'.format(device))
        w_net_model = module_arch.wingnet()
        w_net_model.load_state_dict(torch.load(path_to_model))#, map_location={'cuda:0': 'cpu'}
        print('...done')
        return w_net_model

    def get_device(self, use_gpu):
        if use_gpu and torch.cuda.is_available() and self.use_gpu:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        print('Device:', device)
        return device

    def pass_forward(self, data, model, device):
        output_valid = model(data.to(device, dtype=torch.float)).cpu().detach().numpy()
        output_valid = np.squeeze(output_valid)
        return output_valid

    def process_images(self):
        print("Processing...")
        results = []
        cnt = 0
        for batch_idx, (data, names) in enumerate(self.data_loader):
            print("Loading {} to {}".format(cnt, cnt+len(names)))
            keypoints = self.pass_forward(data, self.model, self.device)
            if keypoints.ndim == 1:
                results.append([names[0], keypoints])
                # print("result={}".format(results[-1]))
            else:
                for name, keypoint in zip(names, keypoints):
                    results.append([name, keypoint])
                    # print("result={}".format(results[-1]))
            cnt += len(names)
            print("Processed {} images".format(cnt))
        print(results)
        return results

# def area_triangle(triangle_points):
#     ax, ay, bx, by, cx, cy = triangle_points
#     area = np.abs((ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))/2)
#     return area
#
# def polygon_area(points):
#     x = np.array(points[0::2])
#     y = np.array(points[1::2])
#     return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
#
# points = [15, 15, 50, 25, 23, 30]
# area_triangle(points)
# print(points[0::2])
# print(points[1::2])
#
# points = [3, 4, 12, 8, 5, 6, 9, 5, 5, 11]
# print(polygon_area(points))


# folder_path = ["/home/timo/Data2/wings/clem_wings/clem_wings/A.F1_NS.xch"]
# image_list = module_data.get_image_paths(folder_path)
# wkptg = WingKeypointsGenerator(image_list)
# output = wkptg.process_images()
# print(output)



# out_list = [[image_list[i], []] for i in range(0, len(image_list))]
# results = pd.DataFrame(out_list, columns=['image_path', 'keypoints'])
# print(out_list)
# results = pd.DataFrame(output_list, columns=['image_path', 'keypoints'])
# print(image_list)
