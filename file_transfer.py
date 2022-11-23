import shutil
import os 

count = 0

source_dir = "/scratch/sshrestha8/masterproject/NAN_CLIP_PAD512_NPY/FITS_PARTITION/PARTITION3"
# train_dest_dir = "/scratch/sshrestha8/masterproject/fake_train_2700_val_300/train/"
val_dest_dir = "/scratch/sshrestha8/masterproject/fake_train_2700_val_300/val/"

source_dir_list = os.listdir(source_dir)

for file in source_dir_list:
    count += 1
    if count<1344:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dest_dir))
    # elif count>=2700 and count<3000:
    #     shutil.copy(os.path.join(source_dir, file), os.path.join(val_dest_dir))
    else: 
        break

# import os
# train_location = os.listdir("/scratch/sshrestha8/masterproject/fake_train_2700_val_300/train")
# val_location = os.listdir("/scratch/sshrestha8/masterproject/fake_train_2700_val_300/val")

# for file in train_location:
#     if file in val_location:
#         print(file)