from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class ROCO_Dataset(Dataset):
    
    def __init__ (self, img_list, label_list, curr_dir, transform, phase, num_classes, config):
    
        self.transform = transform
        self.img_list = img_list
        if phase == 'test':
            if np.size(num_classes) > 1:
                self.listImageLabels = label_list[:,[0,2]]
                self.listImageLabels_sec = label_list[:,1]
            else:
                self.listImageLabels = label_list[:,0]
                self.listImageLabels_sec = label_list[:,1]
        else:
            self.listImageLabels = label_list
        self.curr_dir = curr_dir
        self.phase = phase
        self.pad = config['img_pad']
        
        # check if multi label setting
        if np.size(num_classes) > 1:
            # iterate over labels
            for j in range(0,len(num_classes)):
                imgLabel_cnt = np.zeros(num_classes[j])
        
                # iterate over imgs
                for i in range(len(img_list)):
                    imgLabel = self.listImageLabels[i,:]
                    tmp = np.zeros(num_classes[j])
                    if (imgLabel[j] > num_classes[j]) or (imgLabel[j] < 0):
                        continue
                    else:
                        tmp[imgLabel[j]] = 1
                    imgLabel_cnt = imgLabel_cnt + tmp
                print(imgLabel_cnt)    
        else:
            imgLabel_cnt = np.zeros(num_classes)
        
            # iterate over imgs
            for i in range(len(img_list)):
                imgLabel = self.listImageLabels[i]
                tmp = np.zeros(num_classes)
                tmp[imgLabel] = 1
                imgLabel_cnt = imgLabel_cnt + tmp
            print(imgLabel_cnt)
    
    def __getitem__(self, index):
        
        # video path
        curr_file = self.curr_dir + self.img_list[index] + '.jpg'
        
        # load and pad image
        with Image.open(curr_file).convert('RGB') as img:
            # plt.subplot(1,4,1)
            # plt.imshow(img)
            # print('Original size: '+ str(img.size))
            
            # center crop
            height, width = img.size
            r_min = max(0,np.floor((height-width)/2))
            r_max = r_min + min(height,width)
            c_min = max(0,np.floor((width-height)/2))
            c_max = c_min + min(height,width)
            img = img.crop((r_min,c_min,r_max,c_max))
            # plt.subplot(1,4,2)
            # plt.imshow(img)
            # print('Size after crop: '+ str(img.size))
            
            # padding
            if self.phase == 'train':
                pad = self.pad
                height, width  = img.size
                temp = np.zeros((height + 2 * pad, width + 2 * pad, 3))
                temp[pad:-pad, pad:-pad,:] = img 
                i, j = np.random.randint(0, 2 * pad, 2)
                img = temp[i:(i + height), j:(j + width),:]
                img = Image.fromarray(img.astype('uint8'),'RGB')
                # plt.subplot(1,4,3)
                # plt.imshow(img)
                # print('Size after padding: '+ str(img.size))
        
        # transform
        if self.transform != None: img = self.transform(img)   
        # plt.subplot(1,4,4)
        # plt.imshow(img.permute(1,2,0))
        # print('End Size: '+ str(img.shape))
        
        # image label
        if self.phase == 'test':
            img_label = [self.listImageLabels[index],self.listImageLabels_sec[index]]
        else:
            img_label = self.listImageLabels[index]
        
        return img, img_label, curr_file
    
    def __len__(self):
        
        return len(self.listImageLabels)  

class MNIST_Dataset(Dataset):
    
    def __init__ (self, imgs, img_list, label_list, curr_dir, transform, phase, num_classes, config):
    
        self.transform = transform
        self.img_list = img_list
        if phase == 'test':
            self.listImageLabels = label_list[:,0]
            self.listImageLabels_sec = label_list[:,1]
            self.config = config['experiment_tpye']
            if config['experiment_tpye'] == 'control':
                self.listImageLabels_orig = label_list[:,2]
        else:
            self.listImageLabels = label_list
        self.curr_dir = curr_dir
        self.phase = phase
        self.imgs = imgs
        
        imgLabel_cnt = np.zeros(num_classes)
        
        # iterate over imgs
        for i in range(len(img_list)):
            imgLabel = self.listImageLabels[i]
            tmp = np.zeros(num_classes)
            tmp[imgLabel] = 1
            imgLabel_cnt = imgLabel_cnt + tmp
        print(imgLabel_cnt)
    
    def __getitem__(self, index):
        
        # video path
        curr_file = self.curr_dir + self.img_list[index]
        
        # get the image
        curr_img = self.imgs[:,:,:,index]
        curr_img *= (255.0/curr_img.max())
        img = Image.fromarray(curr_img.astype(np.uint8))
        
        # transform
        if self.transform != None: img = self.transform(img)   
        # plt.imshow(img.permute(1,2,0))
        # print('End Size: '+ str(img.shape))
        
        # image label
        if self.phase == 'test':
            if self.config == 'control':
                img_label = [self.listImageLabels[index],self.listImageLabels_sec[index],self.listImageLabels_orig[index]]
            else:
                img_label = [self.listImageLabels[index],self.listImageLabels_sec[index]]
        else:
            img_label = self.listImageLabels[index]   
        
        return img, img_label, curr_file
    
    def __len__(self):
        
        return len(self.listImageLabels)  
    
class MedMNIST_Dataset(Dataset):
    
    def __init__ (self, img_list, label_list, transform, phase, num_classes, config):
    
        self.transform = transform
        self.img_list = img_list
        if phase == 'test':
            self.listImageLabels = label_list[:,0]
            self.listImageLabels_sec = label_list[:,1]
        else:
            self.listImageLabels = label_list
        self.phase = phase
        
        imgLabel_cnt = np.zeros(num_classes)
        
        # iterate over imgs
        for i in range(len(img_list)):
            imgLabel = self.listImageLabels[i]
            tmp = np.zeros(num_classes)
            tmp[imgLabel] = 1
            imgLabel_cnt = imgLabel_cnt + tmp
        print(imgLabel_cnt)
    
    def __getitem__(self, index):
        
        # get the image
        curr_img = self.img_list[index,:,:]
        if curr_img.max() != 0:
            curr_img = curr_img*(255.0/curr_img.max())
        curr_img = np.stack((curr_img,)*3, axis=-1)
        img = Image.fromarray(curr_img.astype(np.uint8))
        # print('End Size: '+ str(img.shape))
        
        # transform
        if self.transform != None: img = self.transform(img)   
        # plt.imshow(img.permute(1,2,0))
        # print('End Size: '+ str(img.shape))
        
        # image label
        if self.phase == 'test':
            img_label = [self.listImageLabels[index],self.listImageLabels_sec[index]]
        else:
            img_label = self.listImageLabels[index]
        
        return img, img_label, index
    
    def __len__(self):
        
        return len(self.listImageLabels)      