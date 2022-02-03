import torch
import torch.nn as nn
import torch.utils.data
import torch.autograd
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import numpy.random as npr
import time
import Aug_MLP_RPY_model as MLP
from math import sqrt

import os

if (torch.cuda.device_count()):
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))

#Assign cuda GPU located at location '0' to a variable
  cuda0 = torch.device('cuda:0')
else:
  cuda0 = torch.device('cpu')


def freezeLayers(Net,dim):
    if dim == "P":
        for parameter in Net.Pos_FFNet.parameters():
            parameter.requires_grad = False

    if dim == "RPY":
        for parameter in Net.Roll_FFNet.parameters():
            parameter.requires_grad = False

        for parameter in Net.Pitch_FFNet.parameters():
            parameter.requires_grad = False

        for parameter in Net.Yaw_FFNet.parameters():
            parameter.requires_grad = False

    return Net


def freezeRPYLayers(Net,dim):

    if dim == 0:
        for parameter in Net.Yaw_FFNet.parameters():
            parameter.requires_grad = False

        for parameter in Net.Pitch_FFNet.parameters():
            parameter.requires_grad = False

    if dim == 1:
        for parameter in Net.Roll_FFNet.parameters():
            parameter.requires_grad = False

        for parameter in Net.Yaw_FFNet.parameters():
            parameter.requires_grad = False

    if dim == 2:
        for parameter in Net.Roll_FFNet.parameters():
            parameter.requires_grad = False

        for parameter in Net.Pitch_FFNet.parameters():
            parameter.requires_grad = False

    return Net

def resetLayers(Net):
    for parameter in Net.Pos_FFNet.parameters():
        parameter.requires_grad = True

    for parameter in Net.Roll_FFNet.parameters():
        parameter.requires_grad = True

    for parameter in Net.Pitch_FFNet.parameters():
        parameter.requires_grad = True

    for parameter in Net.Yaw_FFNet.parameters():
        parameter.requires_grad = True

    return Net

if __name__ == "__main__":

    #----Load Training Data
    folderData = "Data/"
    
    q_file = folderData+"q_KUKA_N_30.txt" #joint positions file
    dq_file = folderData+"dq_KUKA_N_30.txt" #joint velocities file

    Pos_file = folderData+"P_eul_sincos_KUKA_N_30.txt" #Tip position and tirgonometric representation for RPY
    dPos_file = folderData+"dP_deul_sincos_KUKA_N_30.txt" #Rate of change of tip position and tirgonometric representation for RPY

    # q_file = folderData+"q_KUKA_random.txt" #joint positions file
    # dq_file = folderData+"dq_KUKA_random.txt" #joint velocities file
    #
    # Pos_file = folderData+"P_eul_sincos_KUKA_random.txt" #Tip position and tirgonometric representation for RPY
    # dPos_file = folderData+"dP_deul_sincos_KUKA_random.txt" #Rate of change of tip position and tirgonometric representation for RPY

    #---- Training Setup
    Deriv = True #Include differential relationship in training

    folder_save = folderData+"Aug_ModelKUKA_RPY_orient_indep_100_50_50_100_50_angle_Release_/"

    nj = 7 #number of joints
    task_dim = 3*3 #task dimension: P in R^3, sin(RPY) in R^3, cos(RPY) in R^3

    q = MLP.load_data(nj,q_file) #in R^mxnj
    P = MLP.load_data(task_dim,Pos_file) #in R^mxtask_dim
    dq = MLP.load_data(nj,dq_file) #in R^mxnj
    dP = MLP.load_data(task_dim,dPos_file) #in R^mxtask_dim

    print("############")
    print(q.shape)
    print(P.shape)

    v = np.linspace(1, q.shape[0], q.shape[0])
    # theta
    fig, ax = plt.subplots(4, 2)
    for i in range(nj):
        j = i % 2
        k = i // 2
        ax[k, j].plot(v, q[:, i], lw=2)
        ax[k, j].set_title('Theta' + str(i))

    fig, ax = plt.subplots(5, 2)
    for i in range(task_dim):
        j = i % 2
        k = i // 2
        ax[k, j].plot(v, P[:, i], '-b',lw=2)
        ax[k, j].set_title('P ' + str(i))

    fig, ax = plt.subplots(4, 2)
    for i in range(nj):
        j = i % 2
        k = i // 2
        ax[k, j].plot(v, dq[:, i], lw=2)
        ax[k, j].set_title('dTheta' + str(i))

    fig, ax = plt.subplots(5, 2)
    for i in range(task_dim):
        j = i % 2
        k = i // 2
        ax[k, j].plot(v, dP[:, i], '-b',lw=2)
        ax[k, j].set_title('dP ' + str(i))

    plt.show()

    #------------------------------------------
    #----Set network inputs outputs for training
    Input = np.hstack((q,dq)) #joint positions and velocities
    Output = np.hstack((P, dP))# tip position and trigonometric representation for RPY and their rate of change

    #----Split training and test set
    perc = 0.8 #percentage of training data
    lengths = [int(perc*q.shape[0]), q.shape[0]-int(perc*q.shape[0])]
    X_train, X_test = torch.utils.data.random_split(Input, lengths)

    train_idx = X_train.indices
    test_idx = X_test.indices

    X_train = Input[train_idx,:]
    Y_train = Output[train_idx,:]
    X_test = Input[test_idx,:]
    Y_test = Output[test_idx,:]
    v_train = np.linspace(1, X_train.shape[0], X_train.shape[0])
    v_test = np.linspace(1, X_test.shape[0], X_test.shape[0])

    X_train = torch.tensor(X_train,device=cuda0)
    Y_train = torch.tensor(Y_train,device=cuda0)
    X_test= torch.tensor(X_test,device=cuda0)
    Y_test = torch.tensor(Y_test,device=cuda0)

    X_train = X_train.float()
    Y_train = Y_train.float()
    X_test = X_test.float()
    Y_test = Y_test.float()

    X = torch.tensor(Input,device=cuda0)
    X = X.float()

    #----------------------------------
    #--- Network Training Initializations
    input_dim = Input.shape[1] #input dimension
    out_dim = Output.shape[1] #output dimension
    lr = [1e-03, 1e-03] #learning rates for traiing P and RPY

    sizes = [
        [100,50,50], #neurons in each hidden layer for tip position
             [50,10], #neurons in each hidden layer for roll
             [50,10], #neurons in each hidden layer for pitch
              [50,10] #neurons in each hidden layer for yaw
             ]

    dropout = 0
    Epochs = [1, 5000] #epochs for traiing P and RPY
    batch_size = [X_train.shape[0]//5, #batch size for training P
                  X_train.shape[0] // 5] #batch size for training RPY

    if Deriv:
        w_p = 1e-01 #weight on tip position loss
        w_dp = 1 #weight on tip velocity loss
        w_rpy = 1e-01 #weight on rpy loss (in trigonometric representation)
        w_drpy = 1 #weight on rpy rate of change loss (in trigonometric representation)
    else:
        w_p = 1 #weight on tip position loss
        w_dp = 0 #weight on tip velocity loss
        w_rpy = 1 #weight on rpy loss (in trigonometric representation)
        w_drpy = 0 #weight on rpy rate of change loss (in trigonometric representation)

    w_norm = 1e-03 #weight on sin^2+cos^=1

    #---- Network Initialization
    net = MLP.Aug_MLP_P_RPY_sincos_unc_indep(input_dim, sizes[0], 3, sizes[1:], act='sigmoid', dropout=dropout,
                                             mask=[1,1,1,1,1,1,1,1,1])
    net.to(cuda0)

    #----Train Network part for tip position
    mask_RPY = [0,0,0,0,0,0] #do not consider RPY in loss
    net = freezeLayers(net,"RPY") #do not consider RPY in loss

    net, loss, TrainLoss, TrainLoss_P, TrainLoss_dP,TrainLoss_rpy,TrainLoss_drpy = MLP.trainNetBatch_P_RPY_indep(net, X_train, Y_train,X_test,Y_test,
                                                                        lr[0], batch_size[0],mask_RPY,
                                                                        P_dim = 3,
                                                                        Err=1e-08, Dloss=1e-10, Iter=Epochs[0],
                                                                        NormalizeData=False,
                                                                        w_p = w_p, w_dp= w_dp,w_rpy=w_rpy,
                                                                        w_drpy=w_drpy,w_norm = w_norm)

    #----Train each roll, pitch, yaw network
    for n_or in range(3):
        net = resetLayers(net) #reset each weight to being trainable
        net = freezeLayers(net, "P") #freeeze the network for training P
        net = freezeRPYLayers(net, n_or) #free network for unused angle
        mask_RPY = [0, 0, 0, 0, 0, 0] #initalize mask to zeros
        mask_RPY[n_or] = 1 #set mask for sin of angle to 1
        mask_RPY[n_or+3] = 1 #set mask for cos of angle to 1
        net, loss_2, TrainLoss_2, TrainLoss_P_2, TrainLoss_dP_2,TrainLoss_rpy_2,TrainLoss_drpy_2 = MLP.trainNetBatch_P_RPY_indep(net, X_train, Y_train,X_test,Y_test,
                                                                        lr[1], batch_size[1],mask_RPY,
                                                                        P_dim = 3,
                                                                        Err=1e-08, Dloss=-1e-10, Iter=Epochs[1],
                                                                        NormalizeData=False,
                                                                          w_p=w_p,
                                                                          w_dp=w_dp,
                                                                          w_rpy=w_rpy,
                                                                          w_drpy=w_drpy,
                                                                          w_norm=w_norm)

        TrainLoss = TrainLoss + TrainLoss_2
        TrainLoss_P = TrainLoss_P + TrainLoss_P_2
        TrainLoss_dP = TrainLoss_dP + TrainLoss_dP_2
        TrainLoss_rpy = TrainLoss_rpy + TrainLoss_rpy_2
        TrainLoss_drpy = TrainLoss_drpy + TrainLoss_drpy_2

    #-------------------------------------------------------------------
    #---- test network predictions
    net.to('cpu')
    X_train = X_train.cpu()
    Y_train = Y_train.cpu()
    X_test = X_test.cpu()
    Y_test = Y_test.cpu()

    Out_pred_train = net(X_train)[0]
    Out_pred_train = Out_pred_train.detach().numpy()

    plt.figure()
    plt.plot(range(len(TrainLoss)), TrainLoss, '-b', lw=2, label='loss_Tot')
    plt.plot(range(len(TrainLoss)), TrainLoss_P, '-g', lw=2, label='loss_P')
    plt.plot(range(len(TrainLoss)), TrainLoss_dP, '-', lw=2, label='loss_dP')
    plt.plot(range(len(TrainLoss)), TrainLoss_rpy, '-', lw=2, label='loss_quat')
    plt.plot(range(len(TrainLoss)), TrainLoss_drpy, '-', lw=2, label='loss_dquat')
    plt.title('Train loss')
    plt.legend()

    Y_train = Y_train.numpy()

    P_train = Y_train[:, 0:task_dim]
    dP_train = Y_train[:, task_dim:]
    P_pred_train = Out_pred_train[:, 0:task_dim]
    dP_pred_train = Out_pred_train[:, task_dim:]

    #
    err_train = MLP.mse(Y_train,Out_pred_train)

    ##err test
    Out_pred_test = net(X_test)[0]
    Out_pred_test = Out_pred_test.detach().numpy()
    Y_test = Y_test.detach().numpy()

    P_test = Y_test[:, 0:task_dim]
    dP_test = Y_test [:, task_dim:]
    P_pred_test  = Out_pred_test[:, 0:task_dim]
    dP_pred_test  = Out_pred_test[:, task_dim:]

    err_test = MLP.mse(Y_test,Out_pred_test)

    err_train = sqrt(err_train)
    err_test = sqrt(err_test)

    print("loss ",loss)
    print("err train",err_train)
    print("err test", err_test)

    X = X.cpu()
    Out_tot = net(X)[0]
    Out_tot = Out_tot.detach().numpy()
    P_tot = Out_tot[:, 0:task_dim]
    dP_tot = Out_tot[:, task_dim:]


    fig, ax = plt.subplots(5, 2)
    for i in range(task_dim):
        j = i % 2
        k = i // 2
        ax[k, j].plot(v, P_tot[:,i], '-g',lw=2,label='pred')
        ax[k, j].plot(v, P[:, i], '-k', lw=2, label='true')
        ax[k, j].set_title('P compare ' + str(i))
        ax[k, j].legend()

    fig, ax = plt.subplots(5, 2)
    for i in range(task_dim):
        j = i % 2
        k = i // 2
        ax[k, j].plot(v, dP_tot[:,i], '-g',lw=2,label='pred')
        ax[k, j].plot(v, dP[:, i], '-k', lw=2, label='true')
        ax[k, j].set_title('Pd compare ' + str(i))
        ax[k, j].legend()
    plt.show()

    #----Save data
    info = {'input_dim': input_dim, 'hidden_sizes': sizes, 'output_dim': out_dim, 'err': [err_train,err_test]},

    os.mkdir(folder_save)
    file_info_net = folder_save + "Net_info.txt"
    f = open(file_info_net, 'w')
    f.write(str(info))
    f.close()

    torch.save(net, folder_save + "net.pth")  # save whole model



