import os
from utils import read_generate_data
main_dart_path ="/home/user/Bureau/Houda/Workspace/SentinelPleiadesFusion/Dataset/DART_Sentinel_Pleiades"
train_dart_path = os.path.join(main_dart_path, "train_Sentinel_Pleiades")
val_dart_path =  os.path.join(main_dart_path, "val_Sentinel_Pleiades")
test_dart_path =  os.path.join(main_dart_path, "test_Sentinel_Pleiades")


# read_generate_data(train_dart_path, "./Dataset/train/")
read_generate_data(val_dart_path, "./Dataset/val/")
read_generate_data(test_dart_path, "./Dataset/test/")