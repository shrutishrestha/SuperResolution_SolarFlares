import argparse
import os
import numpy as np
import math
import itertools
import sys
from math import log10
from tqdm import tqdm
import torchvision.utils as utils
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import cProfile, pstats

fake = False



result_train_image_path = "/scratch/sshrestha8/super-resolution2/images/train/"
result_val_image_path = "/scratch/sshrestha8/super-resolution2/images/val/"
result_snapshots_path = "/scratch/sshrestha8/super-resolution2/snapshots/"
os.makedirs(result_train_image_path, exist_ok=True)
os.makedirs(result_val_image_path, exist_ok=True)

os.makedirs(result_snapshots_path, exist_ok=True)


def display_transform():
    return Compose([
        ToPILImage(),
        Pad(512),
        # Resize(512),
        CenterCrop(512),
        ToTensor()
    ])


if not fake:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="true", help="name of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=512, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--train_sample_interval", type=int, default=1000, help="interval between saving image samples")
    parser.add_argument("--val_sample_interval", type=int, default=100, help="interval between saving image samples")

    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--crop_size", type=int, default=512, help="cropping size")
    parser.add_argument("--upscale_factor", type=int, default=4, help="cropping size")
    parser.add_argument("--total_image", type=int, default=15, help="cropping size")
    parser.add_argument("--nrow", type=int, default=3, help="cropping size")

if fake:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="fake", help="name of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=512, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--train_sample_interval", type=int, default=1, help="interval between saving image samples")
    parser.add_argument("--val_sample_interval", type=int, default=1, help="interval between saving image samples")

    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--crop_size", type=int, default=512, help="cropping size")
    parser.add_argument("--upscale_factor", type=int, default=4, help="cropping size")
    parser.add_argument("--total_image", type=int, default=10, help="cropping size")
    parser.add_argument("--nrow", type=int, default=5, help="cropping size")


def unsigned_flux_metric(pred, hmidata):
    batch = len(pred)
    total_difference = 0
    unsigned_hmidataflux_batch_list = []

    for i in range(batch):
        loop_hmidata = hmidata[i]
        loop_pred = pred[i]

        pred_unsigned = torch.abs(loop_pred)
        hmidata_unsigned = torch.abs(loop_hmidata)

        total_unsigned_predictedflux = torch.sum(pred_unsigned)
        total_unsigned_hmidataflux = torch.sum(hmidata_unsigned)
        
        diff = torch.subtract(total_unsigned_predictedflux, total_unsigned_hmidataflux)
        
        total_difference += diff
        
    return total_difference


def range_normalization(image, minimum, maximum):
    return ((image - minimum) / (maximum - minimum))

def range_unnormalized(fitsfile):
    fitsfile = 3000 * fitsfile - 1500

    return fitsfile  # changes to -1500 to 1500

#not using crop size
def main(opt):

    
    #CHANGE
    cuda = torch.cuda.is_available()
    train_set = TrainDatasetFromFolder('/scratch/sshrestha8/masterproject/NAN_CLIP_PAD512_NPY/train_val_test/TRAIN', crop_size=opt.crop_size, upscale_factor=opt.upscale_factor) 

    val_set = ValDatasetFromFolder('/scratch/sshrestha8/masterproject/NAN_CLIP_PAD512_NPY/train_val_test/VAL', upscale_factor=opt.upscale_factor)
    
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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) #new
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) #new
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor #new
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=4)

    # print("discriminator output shape", discriminator.output_shape)
    #parallelize the models
    # generator = torch.nn.DataParallel(generator, device_ids=[0, 1])
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1])
    
    feature_extractor = FeatureExtractor()

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
        
    for epoch in range(opt.epoch, opt.n_epochs):

        train_bar = tqdm(train_loader)
        train_i = 0
        
        generator.train()
        discriminator.train()
        train_images = []
        train_results = {'mse': 0, 'psnr': 0, 'batch_unsigned':0, 'batch_sizes': 0}

        for data, target in train_bar:
            batch_size = data.size(0)
            train_results['batch_sizes'] += batch_size

            train_i += 1
            imgs_lr = Variable(data.type(Tensor)) #new [1, 1, 128, 128]

            imgs_hr = Variable(target.type(Tensor)) #new [1, 1, 512, 512]

            
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False) #new [1, 1, 32, 32]

            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False) #new [1, 1, 32, 32]
            
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
            train_results['mse'] += batch_mse * batch_size

            train_results['psnr'] = 10 * log10((target.max()**2) / (train_results['mse'] / train_results['batch_sizes']))
            # --------------
            #  Log Progress
            # --------------
            un_norm_sr = range_unnormalized(gen_hr) #from here we will calculate with unnormalized 

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, train_i, len(train_loader), loss_D.item(), loss_G.item())
            )

            train_bar.set_description(
                desc='[train converting LR images to SR images] epoch:%d, PSNR: %.4f dB, loss_G: %.4f, loss_D:%.4f' % (
                    epoch, train_results['psnr'], loss_G, loss_D))

#             train_images.extend(
#                 [display_transform()(imgs_lr.squeeze(0)), display_transform()(imgs_hr.data.cpu().squeeze(0)),
#                  display_transform()(gen_hr.data.cpu().squeeze(0))])

#         train_images = torch.stack(train_images) #len = 30

#         train_images = torch.chunk(train_images, train_images.size(0) // opt.total_image)
#         train_save_bar = tqdm(train_images, desc='[saving training results]')
#         index = 1

#         for image in train_save_bar: # 2
#             image = utils.make_grid(image, nrow=opt.nrow, padding=5)
#             utils.save_image(image, result_train_image_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
#             index += 1      
            

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            # torch.save(generator.state_dict(), result_snapshots_path+"/generator_%d.pth" % epoch)
            # torch.save(discriminator.state_dict(), result_snapshots_path +"/discriminator_%d.pth" % epoch)
        
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                }, '/scratch/sshrestha8/epochs/generator__epoch_%d.pth' % (epoch))

            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
                }, '/scratch/sshrestha8/epochs/discriminator_epoch_%d.pth' % (epoch))
        
        #val
        generator.eval()
        discriminator.eval()
        
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'psnr': 0, 'batch_unsigned':0, 'batch_sizes': 0}
            batch_unsigned = 0
            corr2 = 0
            
            val_images = []
            val_i = 0
            for data, target, original in val_bar:

                val_i += 1
            
                batch_size = data.size(0)
                valing_results['batch_sizes'] += batch_size
                
                imgs_lr = Variable(data.type(Tensor)) #new [1, 1, 128, 128]
                imgs_hr = Variable(target.type(Tensor)) #new [1, 1, 512, 512]

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False) #new [1, 1, 32, 32]
                fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False) #new [1, 1, 32, 32]

                gen_hr = generator(imgs_lr) #new
                
                loss_GAN = criterion_GAN(discriminator(gen_hr), valid) #new
            
                # Content loss
                gen_features = feature_extractor(gen_hr) #new
                real_features = feature_extractor(imgs_hr) #new
                loss_content = criterion_content(gen_features, real_features.detach())    #new

                # Total loss
                loss_G = loss_content + 1e-3 * loss_GAN
                val_total_loss_G += loss_G
            
                loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2
            
                val_total_loss_D += loss_D
            
                
                batch_mse = ((gen_hr - imgs_hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                
                valing_results['psnr'] = 10 * log10((target.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                
                un_norm_sr = range_unnormalized(gen_hr) #from here we will calculate with unnormalized 

                batch_unsigned += unsigned_flux_metric(un_norm_sr, original)
                valing_results['batch_unsigned'] = batch_unsigned / valing_results['batch_sizes']

                val_bar.set_description(
                    desc='[val converting LR images to SR images] epoch:%d, PSNR: %.4f dB batch_unsigned: %.4f, loss_G: %.4f, loss_D:%.4f' % (
                    epoch, valing_results['psnr'], valing_results['batch_unsigned'], loss_G, loss_D))

            
                    
                    
                if epoch % 5 == 0 and val_i %100 == 0 :
                    val_images.extend(
                        [display_transform()(imgs_lr.squeeze(0)), display_transform()(imgs_hr.data.cpu().squeeze(0)),
                         display_transform()(gen_hr.data.cpu().squeeze(0))])
                
                # break
                
            if len(val_images) > 0:
                val_images = torch.stack(val_images) #len = 30
                val_images = torch.chunk(val_images, val_images.size(0) // opt.total_image)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 1

                for image in val_save_bar: # 2
                    image = utils.make_grid(image, nrow=opt.nrow, padding=5)
                    utils.save_image(image, result_val_image_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1      
            
            scheduler.step(val_total_loss_G)

            
opt = parser.parse_args()


profiler = cProfile.Profile()
profiler.enable()
main(opt)
profiler.disable()
stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
stats.print_stats(40)
    

