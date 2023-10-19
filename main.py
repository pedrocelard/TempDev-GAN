import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import csv
import random
import traceback
from fvd.fvd import *

from PIL import ImageFile
from PIL import Image
from skvideo import io
from natsort import natsorted
from torchsummary import summary

from data.MovingApples import MovingApples
from data.MovingEmbryo import MovingEmbryo
from data.MovingYeast import MovingYeast
from models.tganv2_gen import Generator_CLSTM
from models.tganv2_dis import DisMultiResNet
from models.biodev_dis import BioDevDisc
from models.biodev_DASA import DASA_metric


from tqdm.gui import tqdm
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import shift
from pathlib import Path


def genSamples(g, n=4, e=1):
    img_size = 256
    
    with torch.no_grad():
        s = g(torch.rand((n**2, 256), device='cuda')*2-1,
              test=True).cpu().detach().numpy()
    out = np.zeros((1, 17, img_size*n, img_size*n))

    for j in range(n):
        for k in range(n):
            out[:, :, img_size*j:img_size*(j+1), img_size*k:img_size*(k+1)] = s[j*n + k, 0, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (np.concatenate([out, out, out], axis=3)+1) / 2 * 255
    io.vwrite(f'tganv2moving/gensamples_id{e}.gif', out)


def subsample_real(h, frames=4):
    h = h[:, np.random.randint(min(frames, h.shape[1]))::frames]
    return h


def full_subsample_real(h, frames=4):
    out = []
    for i in range(4):
        if i:
            out.append(subsample_real(out[i-1], frames=frames))
        else:
            out.append(h)

    for i in range(4):
        for j in range(3-i):
            out[i] = F.avg_pool3d(out[i], kernel_size=(1, 2, 2))
    return out


def zero_centered_gp(real_data, pr):
    gradients = torch.autograd.grad(outputs=pr, inputs=real_data,
                                    grad_outputs=torch.ones_like(pr),
                                    create_graph=True, retain_graph=True)

    return sum([torch.sum(torch.square(g)) for g in gradients])

def dataGen():
    while True:
        for d in loader:
            yield d
            
def train(epochs, 
          batch_size, 
          lambda_val, 
          colors, 
          img_size, 
          log_file,
          bio_loss,
          dataset,
          num_classes,
          n_frames,
          bio_device,
          mode):
    
    zt_dim_size = int(img_size/16)
    
    min_fvd = sys.float_info.max
    dg = dataGen()
    dis = DisMultiResNet(channels=[32, 32, 64, 128, 256], colors=colors).cuda()
    
    if(bio_loss):
        bio_dis = BioDevDisc(num_classes, bio_device)
        checkpoint = 'ResNet18_'+dataset+'_'+str(img_size)+'_'+str(num_classes)+'.pth'
        checkpoint_path = os.path.join("../../Classification/BioLossClassifier/checkpoint/",checkpoint)
        bio_dis.load_checkpoint(checkpoint_path)
    
    # 64x64
    gen = Generator_CLSTM(
        tempc=256,
        zt_dim=zt_dim_size,
        upchannels=[128],
        subchannels=[64, 32, 32],
        n_frames=n_frames,
        colors=colors
    ).cuda()
    
    
    dasa = DASA_metric()
    dasa.load_distribution("./classified_distributions/"+dataset.lower()+'_'+str(img_size)+'_'+str(num_classes)+".txt")

    disOpt = torch.optim.Adam(dis.parameters(), lr=5e-5, betas=(0, 0.9)) #5e-5
    genOpt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9)) #1e-4

    embryo_classes = ('empty','t2', 't3', 't4', 't5', 't6',
            't7', 't8', 't9+', 'tB', 'tEB',
            'tHB', 'tM', 'tPB2', 'tPNa', 'tPNf', 'tSB')
    
    for epoch in tqdm(range(epochs)):

        disOpt.zero_grad()

        if(colors==3):
            real = next(dg).cuda().permute(0,1,4,2,3)
        else:
            real = next(dg).cuda().unsqueeze(2)

        
        real = real.to(dtype=torch.float32) / 255 * 2 - 1
        real = full_subsample_real(real)
        
        for i in real:
            i.requires_grad = True

        pr = dis(real)

        dis_loss = zero_centered_gp(real, pr) * lambda_val

        with torch.no_grad():
            fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)

        pf = dis(fake)

        dis_loss += torch.mean(F.softplus(-pr)) + torch.mean(F.softplus(pf))
        dis_loss.backward()
        disOpt.step()

        genOpt.zero_grad()
        if(False):#bio_loss):
            b, *fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1, bio=bio_loss)
        else:
            fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1, bio=False)

        pf = dis(fake)

        
        if(bio_loss):
            loss_weight = epoch/epochs
            
            batch_samples = 5
            with torch.no_grad():
                b = gen(torch.rand((batch_samples, 256), device='cuda')*2-1, test=True, bio=False)
            
            classification = [None]*20
            dasa_metric = 0
            
            for batch_sample in range(batch_samples):    
                for frame in range(n_frames):
                    inputs = ((b[batch_sample,:,frame,:,:])+1) / 2 * 255
                    
                    if(colors==1):
                        inputs = inputs.repeat(3, 1, 1)
                        inputs = inputs[None,:,:,:]
                    else:
                        inputs = inputs[None,:,:,:]
                    
                    biof = bio_dis(inputs)
    
                    if(dataset == "EMBRYO"):
                        class_item = embryo_classes[biof.item()]
            
                        if(class_item == 'tPB2'):    class_result = 0
                        elif(class_item == 'tPNa'):  class_result = 1
                        elif(class_item == 'tPNf'):  class_result = 2
                        elif(class_item == 't2'):    class_result = 3
                        elif(class_item == 't3'):    class_result = 4
                        elif(class_item == 't4'):    class_result = 5
                        elif(class_item == 't5'):    class_result = 6
                        elif(class_item == 't6'):    class_result = 7
                        elif(class_item == 't7'):    class_result = 8
                        elif(class_item == 't8'):    class_result = 9
                        elif(class_item == 't9+'):   class_result = 10
                        elif(class_item == 'tM'):    class_result = 11
                        elif(class_item == 'tSB'):   class_result = 12
                        elif(class_item == 'tB'):    class_result = 13
                        elif(class_item == 'tEB'):   class_result = 14
                        elif(class_item == 'tHB'):   class_result = 15
                        else: class_result = 16
                        
                    else:
                        class_result = biof.item()
                    
                    classification[frame] = class_result
                    
                    
               
                dasa_metric = dasa_metric + dasa.compute_DASA(classification)

            dasa_mean = torch.tensor(dasa_metric/batch_size)
            gen_loss = torch.mean(F.softplus(-pf)) + dasa_mean


        else:
            gen_loss = torch.mean(F.softplus(-pf))


        gen_loss.backward()
        genOpt.step()


        if epoch % 10000 == 0:
            fvd = 0
            num_train_samples = 10
            min_fvd = 0

            if(fvd < min_fvd):
                min_fvd = fvd
                torch.save({
                            'GEN': gen.state_dict(),
                        }, './checkpoints/'+dataset+mode+"_"+str(size)+'.pth')
                l = open(log_file+'.txt', 'a')
                l.write('SAVING... Epoch '+ str(epoch) + ' Dis '+ str(dis_loss.item())+ ' Gen '+ str(gen_loss.item())+' FVD '+ str(fvd)+"\n")
                l.close()

            print('Epoch', epoch, 'Dis', dis_loss.item(), 'Gen', gen_loss.item(), 'FVD', fvd)
            l = open(log_file+'.txt', 'a')
            l.write('Epoch '+ str(epoch) + ' Dis '+ str(dis_loss.item())+ ' Gen '+ str(gen_loss.item())+' FVD '+ str(fvd)+"\n")
            l.close()

    torch.save({
        'GEN': gen.state_dict(),
    }, './checkpoints/'+dataset+mode+"_"+str(size)+'_FINALEPOCH.pth')
    

    def generate_train_samples(net, colors, img_size, num_samples):
    
        out_samples = np.zeros((num_samples, 20, img_size, img_size, 3))
        
        with torch.no_grad():
            for i in range(num_samples):
                s = net(torch.rand((1, 256), device='cuda')*2-1, test=True).cpu().detach().numpy()
                out = np.zeros((colors, 20, img_size, img_size))

                # print(out.shape)
                out[:, :, :, :] = s[:, :, :, :, :]
                
                out = (out.transpose((1, 2, 3, 0))+1) / 2 * 255
                if (colors==1):
                    # out = (np.concatenate([out, out, out], axis=3)+1) / 2 * 255
                    out = np.concatenate([out, out, out], axis=3)
                out_samples[i, :, :, :, :] = out[:, :, :, :]
                
            return out_samples



def generate_final_samples(colors, img_size, log_file, num_samples):
    zt_dim_size = int(img_size/16)
    
    # 64x64
    trained_gen =  Generator_CLSTM(
        tempc=256,
        zt_dim=zt_dim_size,
        upchannels=[128],
        subchannels=[64, 32, 32],
        n_frames=20,
        colors=colors
    ).cuda()


    trained_gen.load_state_dict(torch.load("./checkpoints/"+log_file+".pth")["GEN"]) 
    
    trained_gen.eval()
    

    generate_samples = True
    generate_images = True

    if(generate_samples):
    
        if(not os.path.exists("final_samples/"+log_file)):
            os.mkdir("final_samples/"+log_file)
        
        with torch.no_grad():
            for i in range(num_samples):
                s = trained_gen(torch.rand((1, 256), device='cuda')*2-1,
                      test=True).cpu().detach().numpy()
                out = np.zeros((colors, 20, img_size, img_size))

                out[:, :, :, :] = s[:, :, :, :, :]
                out = (out.transpose((1, 2, 3, 0))+1) / 2 * 255
                
                if (colors==1):
                    out = np.concatenate([out, out, out], axis=3)
                
                if(generate_images):
                    if(not os.path.exists(os.path.join("final_samples/",log_file, "sample_"+str(i)))):
                        os.mkdir(os.path.join("final_samples/",log_file, "sample_"+str(i)))
                        
                    for frame in range(20):
                        im = Image.fromarray(out[frame].astype(np.uint8))
                        im.save(os.path.join("final_samples/",log_file, "sample_"+str(i),str(frame)+".png"))
                
                io.vwrite(f'final_samples/{log_file}/sample_{i}.gif', out)
    else:
        with torch.no_grad():
            summary(trained_gen, (1, 256))
            print(sum(p.numel() for p in trained_gen.parameters() if p.requires_grad))



action = "train"
bio_loss = True

if(bio_loss): 
    mode_id = "BIOLOSS"
else:
    mode_id = "VANILLA"


sizes = [64,128]
classes = [5,20]

for s in sizes:
    for c in classes: 
        size = s
        num_classes = c
        
        dataset = "" 
        colors = 3
        epochs =  50000
        batch_size = 20
        lambda_val = 0.5
        n_frames = 20
        bio_device = 'cuda:0'
        mode = mode_id+str(num_classes) #VANILLA   BIOLOSS

        log_file = os.path.join("./logs/",dataset+mode+"_"+str(size))

        checkpoint_file = dataset+mode+"_"+str(size)


        if dataset == "EMBRYO":
            data = MovingEmbryo("../../../Datasets/Embryo/", 
                                train=True, 
                                process=True,
                                size=size,
                                data_name="embryo_20")
        if dataset == "APPLE":
            data = MovingApples("../../../Datasets/AppleSet/", 
                                train=True, 
                                download=False, 
                                process=True, 
                                size=size,
                                data_name="apple_20")
        if dataset == "YEAST":
            data = MovingYeast("../../../Datasets/Yeast/", 
                               train=True, 
                               download=False, 
                               process=True, 
                               size=size,
                               data_name="yeast_20")


        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)



        if action == "train":    
            print("Training model")
            train(epochs=epochs, 
                  batch_size=batch_size, 
                  lambda_val=lambda_val, 
                  colors=colors, 
                  img_size=size, 
                  log_file=log_file,
                  bio_loss = bio_loss,
                  dataset = dataset,
                  num_classes = num_classes,
                  n_frames = n_frames,
                  bio_device = bio_device,
                  mode = mode)


        if action == "infer":
            num_samples = 100

            print("Generating new samples")
            generate_final_samples(colors=colors, 
                                   img_size=size, 
                                   log_file=checkpoint_file, 
                                   num_samples=num_samples)
