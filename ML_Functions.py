import os
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from tqdm.notebook import tqdm
from time import time
import numpy as np  
import scipy
import h5py

# Contains training function, preprocessor, distribution functions and some small functions 
def standardize(data,mean,std):
    return (data-mean)/std

def train(model,optimizer, loss_fn, train_dl, val_dl, epochs,lrs, device):
    
    history = {} 

    history['val_loss'] = []
    history['loss'] = []
    history['worst_of'] = []
    
    start_time_sec = time()

    bars = [tqdm(range( i),leave=True) for i in epochs]
    # bars = [range( i) for i in epochs]

    for lr,bar in zip(lrs,bars):
        for g in optimizer.param_groups:
            g['lr'] = lr
        for epoch in bar:

            model.train()
            train_loss         = 0.0
            num_train_examples = 0

            # for batch in tqdm(train_dl,leave=False):
            for batch in train_dl:

                optimizer.zero_grad()

                y = batch.pop('output')
                if 'alphas' in batch.keys():
                    alphas = batch.pop('alphas')
                    yhat = model(**batch)
                    loss = (loss_fn(y,yhat)*alphas).mean()
                
                else:
                    yhat = model(**batch)
                    loss = (loss_fn(y,yhat)).mean()
                
                
                loss.backward()

                optimizer.step()

                train_loss         += loss.data.item() * yhat.shape[0]
                num_train_examples += yhat.shape[0]

            train_loss  = train_loss / len(train_dl.dataset)



            # --- EVALUATE ON VALIDATION SET -------------------------------------
            model.eval()
            val_loss       = 0.0
            worst_of = 0.0
            num_val_examples = 0

            for batch in val_dl:

                y = batch.pop('output')
                if 'alphas' in batch.keys():
                    alphas = batch.pop('alphas')
                    yhat = model(**batch)
                    loss = (loss_fn(y,yhat)*alphas).mean().detach()
                
                else:
                    yhat = model(**batch)
                    loss = (loss_fn(y,yhat)).mean().detach()

                
                val_loss         += (loss).mean().item()* yhat.shape[0]


            val_loss = val_loss / len(val_dl.dataset)
            worst_of = worst_of / len(val_dl.dataset)


            history['worst_of'].append(worst_of)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if np.min(history['val_loss'])== history['val_loss'][-1]:
                torch.save(model.state_dict().copy(),'temp')

            text = [ 'Tr: %.3E, Val: %.3E, B: %.3E, lr: %.1E'%(train_loss,val_loss,np.min(history['val_loss']),optimizer.param_groups[0]['lr'])]

            bar.set_postfix_str(*text,refresh=True)

    end_time_sec       = time()
    total_time_sec     = end_time_sec - start_time_sec
    
    print('Time total: %5.2f sec' % (total_time_sec) + ', Best Val. Loss:{}'.format(np.min(history['val_loss'])))
    return {'state':torch.load('temp'),'history':history}




def sample_fast(params,q_RpT,    N=5000):

    y_dist = np.zeros((q_RpT.shape[0],7,N))
    
    for i in tqdm(range(params.shape[0])):
        
        for j in range(5):
            n = int(N*params[i,0,j])
            y_dist[i,j+2,:n] = np.random.normal(params[i,2,j], params[i,3,j],size=(n))
            y_dist[i,j+2,n:] = np.random.uniform(size=(N-n))*(12+params[i,1 , j])-12

            np.random.shuffle(y_dist[i,j+2])  
            
    # Correlated Dist for R_p and T
    rho = -.7
    
    std = torch.stack([q_RpT[:,0,1]**2,q_RpT[:,0,1]*q_RpT[:,1,1]*rho,q_RpT[:,0,1]*q_RpT[:,1,1]*rho,q_RpT[:,1,1]**2]).T.reshape(-1,2,2)
    y_dist[:,:2] = torch.distributions.multivariate_normal.MultivariateNormal(q_RpT[...,0].type(torch.float64),std)\
    .sample(sample_shape=[N]).swapaxes(0,1).swapaxes(1,2)

    return np.ones([q_RpT.shape[0],N])/N,y_dist


trace_path='TrainingData/Ground Truth Package/Tracedata.hdf5'

# Parametrizations
def dist_diff(params,x,y):
    # params = [a,b,c,d]
    temp = (1-params[0])*1/2*(1+scipy.special.erf( (x-params[2])/(params[3]*np.sqrt(2) ) )  )
    temp[x>params[1]] += params[0]
    temp[x<=params[1]] += params[0]*(-12-x[x<=params[1]])/(-12-params[1])
    return ((y-temp)**2).sum()

def param_finder(trace_path,safe,quart_tr):
    f = h5py.File(trace_path,'r')
    new_params = np.zeros((quart_tr.shape[0],4,5))
    for k,kk in enumerate(tqdm(torch.where(safe)[0])):

            kk = kk.item()
            temp = np.array(f['Planet_train'+str(kk+1)]['tracedata'][:])
            W   = np.array(f['Planet_train'+str(kk+1)]['weights'][:])

            X = [[],[]]
            Y = [[],[]]

            for i in range(2,7):

                x = np.linspace(temp[:,i].min(),temp[:,i].max(),1000)
                if (-temp[:,i].min()+temp[:,i].max())<1:
                    x = np.linspace(temp[:,i].min(),temp[:,i].max(),100)
                y = np.zeros(x.shape)
                for re in range(x.shape[0]):
                    y[re] = ( W[temp[:,i]<=x[re]]).sum()

                X.append(x)
                Y.append(y)
                x_0=np.array([0.01,quart_tr[k,0,i].item(),quart_tr[k,1,i].item(),(quart_tr[k,2,i].item()-quart_tr[k,0,i].item())/2])
                bounds = ([1e-5,-12,-12,1e-5],[1,-1,-1,6])
                # new_params[k,:,i-2] = scipy.optimize.curve_fit(dist_diff,X[-1],Y[-1],x_0,maxfev=10000,#)[0]
                #                                  ,method='lm')[0]
                new_params[k,:,i-2] =scipy.optimize.minimize(dist_diff, x_0, args=(X[i],Y[i]),
                                        bounds=[ [q,j] for q,j in zip(bounds[0],bounds[1])]).x
    f.close()
    return new_params


# For Rp and T loss Functions
def solve_quadratic(params):
    return torch.where( params[:,[0]].abs()>0,torch.stack([
        (-params[:,1]+(params[:,1]**2-4*params[:,2]*params[:,0]).sqrt())/(2*params[:,0]),
        (-params[:,1]-(params[:,1]**2-4*params[:,2]*params[:,0]).sqrt())/(2*params[:,0])],axis=1),
                       (-params[:,2]/params[:,1])[:,None].broadcast_to(params.shape[0],2))

def ks_score_approx(y,yhat):
    m1,std1=torch.split(y,[1,1],dim=1)
    m2,std2=torch.split(yhat,[1,1],dim=1)
    a=(1/std2**2-1/std1**2)
    b=2*(m1/std1**2-m2/std2**2)
    c=2*torch.log(std2/std1)+(m2**2/std2**2-m1**2/std1**2)

    x = solve_quadratic(torch.cat([a,b,c],dim=1))
    return (1-(torch.erf( (x-m1)/(std1*1.414) )-torch.erf( (x-m2)/std2*1/1.414 )).abs().max(axis=1).values*1/2)
    