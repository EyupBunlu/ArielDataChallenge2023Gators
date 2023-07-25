from torch import nn
from torch.utils.data import Dataset
import torch


def construct_FNN(layers,activation=nn.ReLU,output_activation=None,Dropout = None):
    layer = [j for i in layers for j in [nn.LazyLinear(i),activation()] ][:-1]
    if Dropout:
        layer.insert(len(layer)-2,nn.Dropout(Dropout))
    if output_activation is not None:
        layer.append(output_activation())
    return nn.Sequential(*layer)


class Parameter_ML(nn.Module):
    def __init__(self,Dropout=0):
        activation = nn.LeakyReLU
        super(Parameter_ML, self).__init__()
    
        self.attention_layer = construct_FNN([64,64,52],activation=activation)
        
        self.aux_layer =  construct_FNN([64,64,64],activation=activation)
        
        
        self.main_layer = construct_FNN([192,384,384,20],output_activation=nn.Sigmoid,Dropout=Dropout,activation=activation)

        self.noise_layer = construct_FNN([64,64,52],activation=activation)

    def forward(self, input1, input2,input3): #input1:spectra, input2: auxxilary, input3:noise


        a = input1*self.noise_layer(torch.cat((input3,input2,input1),axis=1 ) )
        b=torch.cat( (a*self.attention_layer(a),self.aux_layer(input2) ) ,axis=1 )
        return self.main_layer(b).reshape(a.shape[0],4,5) *torch.Tensor([1,11,11,-6])[None,:,None]-torch.Tensor([0,12,12,-6])[None,:,None]

class RpT_ML(nn.Module):
    def __init__(self,alpha,Dropout=0):
        super(RpT_ML, self).__init__()
        activation = nn.LeakyReLU
       
        self.main_layer=construct_FNN([192,192,192,64,len(alpha[0])],activation=activation,output_activation=nn.Sigmoid,Dropout=Dropout)
        self.alpha=alpha
    def forward(self, input1, input2,input3): #input1:spectra, input2: auxxilary, input3:noise


        return  self.main_layer(torch.cat((input3,input2,input1),axis=1 ) ).reshape(input1.shape[0],len(self.alpha[0]))*self.alpha[1]+self.alpha[0]

class Combined_Quart_Dataset(Dataset):
    def __init__(self,spectra,aux,quart,noise,alphas=None,transform=None):
        self.data = spectra
        self.labels = quart
        self.noise = noise
        self.aux = aux
        self.alphas = alphas
        
        self.transform = transform

    def __len__(self):

        return len(self.labels)
    def __getitem__(self, idx):
        sample={'input1': self.data[idx], 'output': self.labels[idx],'input3':self.noise[idx,:],'input2':self.aux[idx]}
        if type(self.alphas) !=type(None):sample['alphas']=self.alphas[idx]
        if self.transform is not None:          
            sample = self.transform(sample)
        return sample








