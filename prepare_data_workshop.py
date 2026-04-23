import os
import rasterio
from preprocessing import normalize
import numpy as np

data4workshop = "Dataset/test0"
save_path = "Dataset/testing_dataset4workshop"

lst = os.listdir(data4workshop)
img_train_all = []
img_train_names = []
i = 0
for dir in lst:
    img_path = os.path.join(data4workshop, dir )
    save_img_dir = os.path.join(save_path, dir[:-4] )
    os.makedirs(save_img_dir, exist_ok=True)
    # print(img_path)
    with rasterio.open(img_path) as src:
        img_test = src.read()
        img_test_meta = src.meta

    hr_sentinel_img = img_test[:10,:,:]
    hr_sentinel_img_meta = img_test_meta.copy()
    hr_sentinel_img_meta.update({"count": 10 })
    hr_sentinel_imagename = os.path.join( save_img_dir, "hr_sentinel_" + dir)
    print(hr_sentinel_imagename)

    with rasterio.open(hr_sentinel_imagename, "w", **hr_sentinel_img_meta) as dst:
        dst.write(hr_sentinel_img)


    hr_cond_img = img_test[10:, :, :]
    hr_cond_img_meta = img_test_meta.copy()
    hr_cond_img_meta.update({"count": 4 })
    hr_cond_img_imagename = os.path.join( save_img_dir, "hr_guide_rgbnir_" + dir)
    print(hr_cond_img_imagename)

    with rasterio.open(hr_cond_img_imagename, "w", **hr_cond_img_meta) as dst:
        dst.write(hr_cond_img)






