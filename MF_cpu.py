# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:58:29 2019

@author: jx4315
"""
import numpy as np
from numpy import random
import torch
import os
import torch.nn.functional as F
from timeit import default_timer as timer

fold=os.path.join('CPU_MLAM','2')

global_taining_steps = 100
Optimizee_Train_Steps = 10
update_steps = 10
UnRoll_STEPS = 100
layer=2
hidden_sizex=50
hidden_sizew=50
Epo=10
Evaluate_period = 1 #可修改 changeable
optimizer_lr = 0.001 #可修改 changeable
items=10
users=10
Y=[]
G=[]
rank=5
observe=80


for k in range(global_taining_steps):
    M0=np.random.rand(items,users)
    [s,v,d]=np.linalg.svd(M0)
    V_1=np.diag(v)
    M_re=s.dot(V_1)
    M_re=M_re.dot(d)
    end=len(v)
    v[rank:end]=0
    v_d=np.diag(v)
    M_0=s.dot(v_d)
    M_1=M_0.dot(d)
    M_index=np.ones((items,users))
    M_indexv=M_index.reshape(items*users)
    M_indexv[0:observe]=0
    random.shuffle(M_indexv)
    M_index_pick=M_indexv.reshape(items,users)
    M1=M_1*M_index
    Y.append(M1)
    G.append(M_1)




M1 = np.array(Y[0])
N, M, K = len(M1), len(M1[0]),rank
U = np.random.rand(N,K)
V = np.random.rand(M,K)
DIMx1 = M
DIMx2 = K
DIMw1=N
DIMw2=K


def Loss(R ,P, Q, K, R_index):

    R_t=R.astype(float)
    R_t=torch.from_numpy(R_t)
    R_t=R_t.double()
    tempQ = Q.t()
    R_pre=P.mm(tempQ)
    R_pre_index=R_pre*R_index
    R_pre_index=R_pre_index.double()
    R_loss=R_t-R_pre_index
    loss_R1=R_loss*(R_loss)
    loss_R=sum(sum(loss_R1))
    return loss_R


def Adam():
    return torch.optim.Adam()

gradient = 0

def learner_x_train(optimizee, unroll_train_step, w, y, R_index,  retain_graph_flag=True):

    x = torch.empty(DIMx1,DIMx2)
    torch.nn.init.uniform_(x,a=0,b=1)
    global_loss_graph = 0
    x.requires_grad = True 

    if x.requires_grad == True:
        losses=[]
        for i in range (unroll_train_step):
            x.retain_grad()
            loss=Loss(y ,w, x, K, R_index)
            global_loss_graph =global_loss_graph+ loss
            loss.backward(retain_graph = retain_graph_flag,create_graph = True)
            x_next=optimizee(x.grad.detach())
            del x.grad
            del x_next.grad
            x= x+0.01*x_next
            del x_next

        return losses, global_loss_graph,x

    
def learner_w_train(optimizee, unroll_train_step, x,y,R_index,  retain_graph_flag=True, reset_theta=False):

    w = torch.empty(DIMw1,DIMw2)
    torch.nn.init.uniform_(w,a=0,b=1)   
    global_loss_graph = 0
    w.requires_grad = True

    if w.requires_grad == True:
        losses=[]
        for i in range (unroll_train_step):
            w.retain_grad()
            loss=Loss(y ,w, x, K, R_index)
            global_loss_graph += loss 
            loss.backward(retain_graph = retain_graph_flag,create_graph = True)
            w_next=optimizee(w.grad.detach())
            del w.grad
            del w_next.grad
            w= w+0.01*w_next
            del w_next

        return losses, global_loss_graph,w

input_sizex=DIMx2
output_sizex=DIMx2
batchsizex=DIMx1


input_sizew=DIMw2
output_sizew=DIMw2
batchsizew=DIMw1

class Optimizee_x(torch.nn.Module):
    def __init__(self ):
        super(Optimizee_x,self).__init__()

        self.fnn1 = torch.nn.Linear(input_sizex, hidden_sizex)
        self.fnn2 = torch.nn.Linear(hidden_sizex, hidden_sizex)
        self.out = torch.nn.Linear(hidden_sizex,output_sizex)

    def forward(self,gradient):
            gradient=gradient.unsqueeze(0)

            update = F.relu(self.fnn1(gradient))
            update = F.relu(self.fnn2(update))
            update = self.out(update)
            update = update.squeeze(0)
            return update




class Optimizee_w(torch.nn.Module):
    def __init__(self ):
        super(Optimizee_w,self).__init__()

        self.fnn1 = torch.nn.Linear(input_sizew, hidden_sizew)
        self.fnn2 = torch.nn.Linear(hidden_sizew, hidden_sizew)
        self.out = torch.nn.Linear(hidden_sizew, output_sizew)
        
    def forward(self,gradient):
            gradient=gradient.unsqueeze(0)

            update = F.relu(self.fnn1(gradient))
            update = F.relu(self.fnn2(update))
            update = self.out(update)
            update = update.squeeze(0)
            return update

Epoch=Epo
epoch_loss_list = []
optimizee_x=Optimizee_x()
adam_global_optimizer_x = torch.optim.Adam(optimizee_x.parameters(),lr = optimizer_lr)#用adam优化lstm优化器的参数
optimizee_w=Optimizee_w()
adam_global_optimizer_w = torch.optim.Adam(optimizee_w.parameters(),lr = optimizer_lr)#用adam优化lstm优化器的参数
print(optimizee_w)
print(optimizee_x)
for epoch in range(Epoch):
    T=np.arange(global_taining_steps)
    random.shuffle(T)        
    global_loss_list=[]    
    for i in range(global_taining_steps):#遍历100个矩阵
        index=T[i]
        AP_loss_list=[]
        y=Y[index]
        g=G[index]
        g_tensor=torch.from_numpy(g)
        R_index=torch.zeros(N,M)
        for q in range(M):
            for p in range(N):
                if y[q,p]>0:
                    R_index[q,p]=1
        global_loss_x=0
        global_loss_w=0
        global_loss=0
        print('\n=======> Epoch {} global training steps: {}'.format(epoch+1.0,i))
        start = timer()
        w= torch.empty(DIMw1,DIMw2)
        torch.nn.init.uniform_(w,a=0,b=1)
        for num in range(UnRoll_STEPS):
            x_loss, x_sum_loss,x = learner_x_train(optimizee_x,Optimizee_Train_Steps,w.clone().detach(),y,R_index)
            w_loss,w_sum_loss,w = learner_w_train(optimizee_w,Optimizee_Train_Steps,x,y,R_index)
            lossw=Loss(y ,w, x.detach(), K, R_index)
            lossx=Loss(y ,w.detach(), x, K, R_index)
            loss=Loss(y ,w, x, K, R_index)
            global_loss_x += lossw
            global_loss_w += lossx
            global_loss=loss

            if (num+1)% update_steps == 0:
                adam_global_optimizer_x.zero_grad()
                adam_global_optimizer_w.zero_grad()
                E_global_loss_x=global_loss_x/update_steps
                E_global_loss_w=global_loss_w/update_steps
                E_global_loss_x.backward(retain_graph=True,create_graph = True)
                E_global_loss_w.backward(retain_graph=True,create_graph = True)
                adam_global_optimizer_x.step()#
                adam_global_optimizer_w.step()

                time = timer() - start           
                RC=w.mm(x.t()).double()
                RC_d=RC-g_tensor.double()
                RC_loss=np.linalg.norm(RC_d.detach().numpy(),'fro')/np.linalg.norm(g,'fro')
                print('->  step :' ,num+1,'Ob_loss=','%.2f'%global_loss.detach().numpy(),'reconstruction error =','%.2f'%RC_loss,'time=','%.0f'%time)
                global_loss_x=0
                global_loss_w=0
                global_loss=0
            

    torch.save(optimizee_x, os.path.join(fold, 'X_ob_20_rk_5_epoch_{}.pth'.format(epoch)))
    torch.save(optimizee_w, os.path.join(fold, 'W_ob_20_rk_5_epoch_{}.pth'.format(epoch)))

