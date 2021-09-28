#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from torch import device
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset

from read_data import create_data


#%%
class my_dataset(Dataset):
    def __init__(self,data,attribute_label):
        super(my_dataset,self).__init__()
        self.data=data
        self.attribute_label=attribute_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        batch_data=self.data[index]
        batch_label=self.attribute_label[index]
        return batch_data,batch_label
        
#%%
device=torch.device('cuda')

np.random.seed(904)

def pre_model(model, traindata, train_attributelabel, testdata, testlabel, attribute_matrix):
    model_dict = {'rf': RandomForestClassifier(n_estimators=100),'NB': GaussianNB(),'SVC_linear': SVC(kernel='linear'),'LinearSVC':LinearSVC()}

    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix - pre_res), axis=1)).argmin()
        label_lis.append(np.unique(testlabel)[loc])
    label_lis = np.mat(np.row_stack(label_lis))
    return test_pre_attribute,label_lis, testlabel

#%%
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

#%%
class Embedding_Net(nn.Module):

    def __init__(self,dim,lambda_):
        super(Embedding_Net,self).__init__()
        self.l11=nn.Linear(6,dim[0])
        self.l12=nn.Linear(dim[0],dim[1])
        self.l13=nn.Linear(2*dim[1],6)

        self.l21=nn.Linear(4,dim[0])
        self.l22=nn.Linear(dim[0],dim[1])
        self.l23=nn.Linear(2*dim[1],4)

        self.bn1=nn.BatchNorm1d(dim[0])
        self.bn2=nn.BatchNorm1d(dim[1])
        
        self.lambda_=lambda_


    def compability_loss(self,z1,z2):
        N,D=z1.shape

        c=self.bn2(z1).T @ self.bn2(z2)/N

        on_diag=torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag=off_diagonal(c).pow_(2).sum()
        loss=on_diag+self.lambda_[3]*off_diag

        return loss

    def compute_loss(self,z1,z2,x,a,x_,a_):
        loss_R1=self.lambda_[0]*F.mse_loss(a,a_)
        loss_R2=self.lambda_[1]*F.mse_loss(x,x_)
        loss_CM=self.compability_loss(z1,z2)
        loss_CM=self.lambda_[2]*loss_CM
        loss=loss_R1+loss_R2+loss_CM
        return loss_R1,loss_R2,loss_CM,loss

    def transform(self,x,a):
        z1=self.l11(x)
        z1=torch.relu(self.bn1(z1))
        z1=self.l12(z1)

        z2=self.l21(a)
        z2=torch.relu(self.bn1(z2))
        z2=self.l22(z2)
        return z1,z2

    def reconstruction(self,z1,z2):
        f1=torch.cat([z1,z2],dim=1)
        f2=torch.cat([z2,z1],dim=1)
        x_=self.l13(f1)
        a_=torch.sigmoid(self.l23(f2))
        return x_,a_


    def forward(self,x,a):
        z1,z2=self.transform(x,a)
        x_,a_=self.reconstruction(z1,z2)

        loss_R1,loss_R2,loss_CM,loss=self.compute_loss(z1,z2,x,a,x_,a_)
        package={'z1':z1,'z2':z2,'x':x,'x_':x_,'r1':loss_R1,
                'r2':loss_R2,'cm':loss_CM,'loss':loss}
        
        return package

#%%
datapath='data/classData.csv'
modes=['NB'] #'rf'
test_classes={'test_class':[2,3]}
for key,value in test_classes.items():
    print('========================================{}:[{}:{}]========================================='.format(modes,key,value))
    df = pd.read_csv(datapath)
    df['fault_type'] = df['G'].astype('str') + df['C'].astype('str') + df['B'].astype('str') + df['A'].astype('str')
    traindata,trainlabel,train_attributelabel, train_attributematrix,testdata,testlabel,test_attributelabel,test_attributematrix,attribute_matrix=create_data(df,value)

    _,y_pre,y_true=pre_model(modes[0], traindata, train_attributelabel, testdata, testlabel, test_attributematrix)
    original_acc=accuracy_score(y_pre,y_true)

    traindata=torch.from_numpy(traindata).float().to(device)
    label=torch.from_numpy(trainlabel.squeeze()).long().to(device)

    testdata=torch.from_numpy(testdata).float().to(device)
    batch_size=400
    trainset=my_dataset(traindata,torch.from_numpy(train_attributelabel).float().to(device))
    train_loader=DataLoader(trainset,batch_size=batch_size,shuffle=True)

    lambda_=[1,1e-5,1,0.25]
    dim=[6,12]
    model=Embedding_Net(dim,lambda_=lambda_)
    model.to(device)

    optimizer=optim.RMSprop(model.parameters(),lr=1e-2)

    L1,L2,L3,L=[],[],[],[]
    model.train()

    accs=[]
    best_acc=0
    for epoch in range(200):
        model.train()
        for batch,(batch_data,batch_label) in enumerate(train_loader):

            optimizer.zero_grad() 
            package=model(batch_data,batch_label)
            loss_R1,loss_R2,loss_CM,loss=package['r1'],package['r2'],package['cm'],package['loss']
            
            loss.backward()
            optimizer.step()

            L1.append(loss_R1.item())
            L2.append(loss_R2.item())
            L3.append(loss_CM.item())
            L.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_package=model(traindata,torch.from_numpy(train_attributelabel).float().to(device))
            f_train=train_package['z1']
            f_train=torch.cat([f_train,traindata],dim=1).detach().cpu().numpy()

            test_package=model(testdata,torch.from_numpy(test_attributelabel).float().to(device))
            f_test=test_package['z1']
            f_test=torch.cat([f_test,testdata],dim=1).detach().cpu().numpy()

            test_preattribute,label_lis, testlabel=pre_model(modes[0], f_train, train_attributelabel, f_test, testlabel, test_attributematrix)
            acc=accuracy_score(label_lis, testlabel)
            accs.append(acc)
            if acc>best_acc:
                best_acc=acc

        print('epoch:{:d}, best_acc:{:.4f}'.format(epoch,best_acc))

print('finished! FDAT:{:.4f}, SCE:{:.4f}'.format(original_acc,best_acc))

# %%