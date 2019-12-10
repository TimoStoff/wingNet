import deploy_network as wing_net
import data_loader.data_loaders as module_data
import glob
import torch

def test_data_loader(image_list, BATCH_SIZE=8, RESIZE=(256, 256)):
    data_loader = module_data.WingsInferenceDataLoader(image_list, BATCH_SIZE,
                                                       resize_dims=RESIZE, shuffle=False,
                                                       validation_split=0.0, num_workers=2)
    for batch_idx, (data, names) in enumerate(data_loader):
        print(data.shape)
        print(len(names))

def test_forward(image_list):
    print("test GPU cuda:{}".format(torch.cuda.is_available()))
    # print(wings)
    model = wing_net.WingKeypointsGenerator(image_list)
    out = model.process_images()
    # print(model.model)

if __name__ == "__main__":
    wing_path = '/home/timo/Data2/wings/ExpEvo/test'
    wings = sorted(glob.glob("{}/*.jpg".format(wing_path)))

    test_forward(wings)
    # test_data_loader(wings)