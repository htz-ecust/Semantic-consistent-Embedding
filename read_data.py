#%%
import numpy as np

#%%
def get_data(data_list,train_index,attribute_matrix):
    traindata=[]
    trainlabel=[]
    train_attributelabel=[]
    for item in train_index:
        data=data_list[item]
        traindata.append(data[:,4:10])
        num=data.shape[0]
        trainlabel +=[item]*num
        train_attributelabel +=[attribute_matrix[item,:]]*num
    traindata=np.row_stack(traindata)
    trainlabel=np.row_stack(trainlabel)
    train_attributelabel=np.row_stack(train_attributelabel)
    return traindata,trainlabel,train_attributelabel

def create_data(df,test_index=[3,4]):
    
    fault0=df[df['fault_type']=='0000']
    fault1=df[df['fault_type']=='1001']
    fault2=df[df['fault_type']=='0110']
    fault3=df[df['fault_type']=='1011']
    fault4=df[df['fault_type']=='0111']
    fault5=df[df['fault_type']=='1111']


    train_index=list(set(np.arange(6))-set(test_index))
    attribute_matrix=np.array([[0,0,0,0],
                            [1,0,0,1],
                            [0,1,1,0],
                            [1,0,1,1],
                            [0,1,1,1],
                            [1,1,1,1]])

    data_list=[fault0.values.astype(np.float64),fault1.values.astype(np.float64),fault2.values.astype(np.float64),
                fault3.values.astype(np.float64),fault4.values.astype(np.float64),fault5.values.astype(np.float64)]

    train_index.sort()
    test_index.sort()
    train_attributematrix=attribute_matrix[train_index]
    test_attributematrix=attribute_matrix[test_index]

    print("train_classes: {}, test_classes:{}".format(train_index,test_index))
    traindata,trainlabel,train_attributelabel=get_data(data_list,train_index,attribute_matrix)
    testdata,testlabel,test_attributelabel=get_data(data_list,test_index,attribute_matrix)
    print("traindata: {}, testdata:{}".format(traindata.shape,testdata.shape))

    return  traindata,trainlabel,train_attributelabel, train_attributematrix,testdata,testlabel,test_attributelabel,test_attributematrix,attribute_matrix
 
#%%
