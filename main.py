import argparse
import os
import numpy as np
import math
import itertools
import sys
import csv
from math import log10
from tqdm import tqdm
import torchvision.utils as utils
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ssim import SSIM
from models import *
from datasets import *
import torch
from numba import cuda
import torch.nn as nn
import torch.nn.functional as F

import torch
import gc
import cProfile, pstats
from GPUtil import showUtilization as gpu_usage



saving_directory="/scratch/sshrestha8/super-resolution2"
job_id = os.getenv('SLURM_JOB_ID')


saving_directory = os.path.join(saving_directory, str(job_id))
os.makedirs(saving_directory, exist_ok=True)

result_fake_image_directory = os.path.join(saving_directory, "fake_images")
os.makedirs(result_fake_image_directory, exist_ok=True)

result_train_image_path = os.path.join(result_fake_image_directory, "train/")
os.makedirs(result_train_image_path, exist_ok=True)

result_val_image_path = os.path.join(result_fake_image_directory, "val/")
os.makedirs(result_val_image_path, exist_ok=True)

result_snapshot_image_path = os.path.join(saving_directory, "fake_snapshots/")
os.makedirs(result_snapshot_image_path, exist_ok=True)

result_csv_path = os.path.join(saving_directory, "result/")
os.makedirs(result_csv_path, exist_ok=True)

result_csv_path = os.path.join(result_csv_path, "result.csv")


def display_transform():
    return Compose([
        ToPILImage(),
        Pad(512),
        # Resize(512),
        CenterCrop(512),
        ToTensor()
    ])



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="true", help="name of the dataset")
parser.add_argument("--train_batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--gen_lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--dis_lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--train_sample_interval", type=int, default=1, help="interval between saving image samples")
# parser.add_argument("--val_sample_interval", type=int, default=1, help="interval between saving image samples")

# parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model checkpoints")
parser.add_argument("--crop_size", type=int, default=256, help="cropping size")
parser.add_argument("--upscale_factor", type=int, default=4, help="cropping size")
parser.add_argument("--total_image", type=int, default=15, help="cropping size")
parser.add_argument("--nrow", type=int, default=3, help="cropping size")
parser.add_argument("--train_partitions", type=str, default="1,2,4", help="training partitions")
parser.add_argument("--val_partitions", type=str, default="3", help="validation partitions")
parser.add_argument("--main_folder", type=str, default="/scratch/sshrestha8/masterproject/FAKE_NAN_CLIP_PAD512_NPY/FITS_PARTITION/PARTITION", help="main folder path")


# fake: 
#/scratch/sshrestha8/masterproject/FAKE_NAN_CLIP_PAD512_NPY/FITS_PARTITION/PARTITION
#true:
#/scratch/sshrestha8/masterproject/NAN_CLIP_PAD512_NPY/FITS_PARTITION/PARTITION



def unsigned_flux_metric(pred, hmidata):
    abs_difference = abs(pred - hmidata)
        
    return abs_difference.sum()


def range_normalization(image, minimum, maximum):
    return ((image - minimum) / (maximum - minimum))

def range_unnormalized(fitsfile):
    fitsfile = 3000 * fitsfile - 1500

    return fitsfile  # changes to -1500 to 1500

#/scratch/sshrestha8/masterproject/FAKE_NAN_CLIP_PAD512_NPY/train_val_test/TRAIN
#/scratch/sshrestha8/masterproject/fake_train_2700_val_300/train/

#not using crop size
def main(opt):


    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    
    if cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    
    train_set = DatasetFromFolder(main_folder=opt.main_folder,partition_list=opt.train_partitions, crop_size=opt.crop_size, upscale_factor=opt.upscale_factor, train=True) 

    val_set = DatasetFromFolder(main_folder = opt.main_folder, partition_list=opt.val_partitions, crop_size=opt.crop_size, upscale_factor=opt.upscale_factor, train=False)
    
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.train_batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
    
    hr_shape = (opt.hr_height, opt.hr_width)
 
    # Initialize generator and discriminator
    generator = GeneratorResNet() #new
    hr_shape = (opt.hr_height, opt.hr_width)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)) #new
    

        
    if opt.epoch != 0: #new
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth")) #new
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth")) #new
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.gen_lr, betas=(opt.b1, opt.b2)) #new
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.dis_lr, betas=(opt.b1, opt.b2)) #new
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=4)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=4)
    

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor #new

    
    ssim_metric = SSIM()
    
    feature_extractor = FeatureExtractor(device)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()


    if cuda:
        generator = generator.cuda() #new
        discriminator = discriminator.cuda()  #new
        feature_extractor = feature_extractor.cuda() #new
        criterion_GAN = criterion_GAN.cuda() #new
        criterion_content = criterion_content.cuda() #new
        
    discriminator_for_shape = discriminator

    # parallelize the models
    generator = torch.nn.DataParallel(generator, device_ids=[0, 1,2,3])
    discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1,2,3])
    feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=[0, 1,2,3])
    
    results = {'train_mse': [],'val_mse': [], 'train_psnr': [],'val_psnr': [], 'train_batch_unsigned':[], 'val_batch_unsigned':[], 'train_batch_sizes': [], 'val_batch_sizes': [], 'train_loss_G':[], 'val_loss_G':[], 'train_loss_D':[], 'val_loss_D':[]}
    for epoch in range(opt.epoch, opt.n_epochs):
        print("epoch", epoch)

        train_i = 0
        
        generator.train()
        discriminator.train()
        train_images = []
        
        
        train_results = {'mse': 0, 'psnr': 0, 'batch_unsigned':0, 'batch_sizes': 0, 'loss_G':0, 'loss_D':0, 'ssim':0}

        for i, batch  in enumerate(train_loader):

            imgs_lr= batch[0].to(device).float() #[1, 1, 128, 128]
            imgs_hr= batch[1].to(device).float() #new [1, 1, 512, 512]
            original = batch[2].to(device).float() #new [1, 1, 512, 512]
            
            batch_size = imgs_lr.size(0)
            train_results['batch_sizes'] += batch_size

            train_i += 1
            
            
            # Adversarial ground truths

            valid = torch.from_numpy(np.ones((imgs_lr.size(0), *discriminator_for_shape.output_shape))) #new [1, 1, 32, 32]
            valid = valid.to(device).float()

            fake = torch.from_numpy(np.zeros((imgs_lr.size(0), *discriminator_for_shape.output_shape))) #new [1, 1, 32, 32]
            fake = fake.to(device).float()

            optimizer_G.zero_grad() #new
            gen_hr = generator(imgs_lr) #new
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid) #new
            
            # Content loss
            gen_features = feature_extractor(gen_hr) #new
            real_features = feature_extractor(imgs_hr) #new
            loss_content = criterion_content(gen_features, real_features.detach())    #new
            
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2


            loss_D.backward()
            optimizer_D.step()

            batch_mse = ((gen_hr - imgs_hr) ** 2).data.mean()
            train_results["mse"] += batch_mse * batch_size

            un_norm = range_unnormalized(gen_hr)
            train_results["batch_unsigned"] += unsigned_flux_metric(un_norm, original)
            
            train_results["loss_G"] += loss_G
            train_results["loss_D"] += loss_D
            # train_results["ssim"] += ssim_metric(un_norm, original)
            train_results["ssim"] += 0
            


            psnr = 10 * log10((imgs_hr.max()**2) / (train_results["mse"]/ train_results['batch_sizes']))
            train_results["psnr"] += psnr 
            if epoch % 5 == 0 and train_i %100 == 0 :
                imgs_lr0 = imgs_lr[0]
                imgs_hr0 = imgs_hr[0]
                gen_hr0 = gen_hr[0]


                train_images.extend(
                [display_transform()(imgs_lr0.squeeze(0)), display_transform()(imgs_hr0.data.cpu().squeeze(0)),
                 display_transform()(gen_hr0.data.cpu().squeeze(0))])
            
            gc.collect()
            torch.cuda.empty_cache()
            
            del original, valid, fake, imgs_lr, imgs_hr, gen_hr, gen_features, real_features, loss_content, loss_real
            del loss_fake, loss_D, batch_mse, un_norm, psnr

        
        train_results["mse"] = train_results["mse"].item() / train_results['batch_sizes']
        train_results["batch_unsigned"] = train_results["batch_unsigned"].item()/ len(train_loader)
        train_results["loss_G"] = train_results["loss_G"].item() / len(train_loader)
        train_results["loss_D"] = train_results["loss_D"].item() / len(train_loader)
        train_results["psnr"] = train_results["psnr"] / len(train_loader)
        train_results["ssim"] = train_results["ssim"] / len(train_loader)

        
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            }, result_snapshot_image_path+'/generator__epoch_%d.pth' % (epoch))

        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict(),
            }, result_snapshot_image_path+'/discriminator_epoch_%d.pth' % (epoch))
        
        #val
        generator.eval()
        discriminator.eval()
        
        
        with torch.no_grad():

            val_results = {'mse': 0, 'psnr': 0, 'batch_unsigned':0, 'batch_sizes': 0,'loss_G':0, 'loss_D':0, 'ssim':0}
            
            batch_unsigned = 0
            corr2 = 0
            
            val_images = []
            val_i = 0
            for i, batch in enumerate(val_loader):
                
                imgs_lr= batch[0].to(device).float() #[1, 1, 128, 128]
                imgs_hr= batch[1].to(device).float() #new [1, 1, 512, 512]
                original = batch[2].to(device).float() #new [1, 1, 512, 512]
                
            
                val_i += 1
            
                batch_size = imgs_lr.size(0)
                val_results['batch_sizes'] += batch_size

                # Adversarial ground truths
                valid = torch.from_numpy(np.ones((imgs_lr.size(0), *discriminator_for_shape.output_shape))) #new [1, 1, 32, 32]
                valid = valid.to(device).float()

                fake = torch.from_numpy(np.zeros((imgs_lr.size(0), *discriminator_for_shape.output_shape))) #new [1, 1, 32, 32]
                fake = fake.to(device).float()

                gen_hr = generator(imgs_lr) #new
                
                loss_GAN = criterion_GAN(discriminator(gen_hr), valid) #new
            
                # Content loss
                gen_features = feature_extractor(gen_hr) #new
                real_features = feature_extractor(imgs_hr) #new
                loss_content = criterion_content(gen_features, real_features.detach())    #new

                # Total loss
                loss_G = loss_content + 1e-3 * loss_GAN
            
                loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2
                
                batch_mse = ((gen_hr - imgs_hr) ** 2).data.mean()
                val_results["mse"] += batch_mse * batch_size
                
                un_norm = range_unnormalized(gen_hr)
                val_results["batch_unsigned"] += unsigned_flux_metric(un_norm, original)
                # val_results["ssim"] += ssim_metric(un_norm, original)
                val_results["ssim"] += 0
                
                val_results["loss_G"] += loss_G
                val_results["loss_D"] += loss_D
                
                psnr = 10 * log10((imgs_hr.max()**2) / (val_results["mse"]/ val_results['batch_sizes']))
                val_results["psnr"] += psnr 
                
                if epoch % 5 == 0 and val_i %50 == 0 :#change
             
                    imgs_lr0 = imgs_lr[0]
                    imgs_hr0 = imgs_hr[0]
                    gen_hr0 = gen_hr[0]

                    val_images.extend(
                    [display_transform()(imgs_lr0.squeeze(0)), display_transform()(imgs_hr0.data.cpu().squeeze(0)),
                     display_transform()(gen_hr0.data.cpu().squeeze(0))])
                    
                gc.collect()
                torch.cuda.empty_cache()
            
            
                del original, valid, fake, imgs_lr, imgs_hr, gen_hr, gen_features, real_features, loss_content, loss_real
                del loss_fake, loss_D, batch_mse, un_norm, psnr
        
            val_results["mse"] = val_results["mse"].item() / val_results['batch_sizes']
            val_results["batch_unsigned"] = val_results["batch_unsigned"].item() / len(val_loader)
            val_results["loss_G"] = val_results["loss_G"].item() / len(val_loader)
            val_results["loss_D"] = val_results["loss_D"].item() / len(val_loader)
            val_results["ssim"] = val_results["ssim"] / len(val_loader)
            val_results["psnr"] = val_results["psnr"] / len(val_loader)
            
        scheduler_G.step(val_results["loss_G"])
        scheduler_D.step(val_results["loss_D"])
        
        if len(train_images) > 0:
            
            train_images = torch.stack(train_images) #len = 30
            train_images = torch.chunk(train_images, train_images.size(0) // opt.total_image)
            train_save_bar = tqdm(train_images, desc='[saving training results]')
            train_index = 1
            
            for image in train_save_bar: # 2
                
                image = utils.make_grid(image, nrow=opt.nrow, padding=5)
                utils.save_image(image, result_train_image_path + 'epoch_%d_trainindex_%d.png' % (epoch, train_index), padding=5)
                train_index += 1   
            
        if len(val_images) > 0:
            
            val_images = torch.stack(val_images) #len = 30
            val_images = torch.chunk(val_images, val_images.size(0) // opt.total_image)
            val_save_bar = tqdm(val_images, desc='[saving validation results]')
            val_index = 1

            for image in val_save_bar: # 2
                
                image = utils.make_grid(image, nrow=opt.nrow, padding=5)
                utils.save_image(image, result_val_image_path + 'epoch_%d_valindex_%d.png' % (epoch, val_index), padding=5)
                val_index += 1   

        result_data = [epoch, optimizer_G.param_groups[0]["lr"], optimizer_D.param_groups[0]["lr"], round(train_results["mse"],4), round(val_results["mse"],4), round(train_results["batch_unsigned"],4), round(val_results["batch_unsigned"],4), round(train_results["loss_G"],4), round(val_results["loss_G"],4), round(train_results["loss_D"],4), round(val_results["loss_D"],4), round(train_results["psnr"],4), round(val_results["psnr"],4)]
        
        headers = ['epoch', 'G_lr', 'D_lr', 'train mse','val mse', 'train difference of sign', 'val difference of sign', 'train loss_G', 'val loss_G', 'train loss_D', 'val loss_D', 'train psnr', 'val psnr']
        
        file_exists = os.path.isfile(result_csv_path)
        
        with open(result_csv_path, "a") as outfile:
            writer = csv.writer(outfile)
            
            if not file_exists:
                writer.writerow(headers)  # file doesn't exist yet, write a header
                
            writer.writerow(result_data)
                    
                    
opt = parser.parse_args()


profiler = cProfile.Profile()
profiler.enable()
main(opt)
profiler.disable()
stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
stats.print_stats(40)
    
