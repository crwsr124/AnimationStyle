import os
import shutil
from skimage import transform, io


origin_path = "/home/cr/Desktop/anime_data/"
img_dir = "./B_256/"
bg_dir = "./B_256_bg/"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(bg_dir, exist_ok=True)

i = 0
for root, ds, fs in os.walk(origin_path):
    if root.endswith('512'):
        file_list = os.listdir(root)
        for name in file_list:
            file_path = os.path.join(root, name)
            origin_img = io.imread(file_path)
            out_img = transform.resize(origin_img, (256, 256), anti_aliasing=True)
            out_file_path = os.path.join(img_dir, str(i)+".png")
            print(out_file_path)
            io.imsave(out_file_path, out_img)
            i = i+1

i = 0
for root, ds, fs in os.walk(origin_path):
    if root.endswith('BG256'):
        file_list = os.listdir(root)
        for name in file_list:
            file_path = os.path.join(root, name)
            out_file_path = os.path.join(bg_dir, str(i)+".png")
            shutil.copy(file_path, out_file_path)
            i = i+1
            