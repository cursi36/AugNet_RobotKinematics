import torch
import torch.nn as nn
import torch.autograd
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import numpy.random as npr
import time
import os

def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

class SiLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return silu(input) # simply apply already implemented SiLU

def Normalize(x,mean,std):

    m = x.shape[0]
    mean = mean.repeat(m,1)
    std = std.repeat(m,1)

    x = torch.div((x-mean),std)

    return x

def mse(y,t):

    y = y.reshape(-1)
    t = t.reshape(-1)
    err = (y-t)**2
    err = np.mean(err)

    return err

def load_data(dim,file):

    Data = np.loadtxt(file)

    if Data.ndim > 1:
        nr, nc = Data.shape
    else:
        nr = Data.shape


    if nr == dim:
        Data = Data.transpose()

    return Data

#Decoupling Layer: Splits the inputs
class DecoupleLayer(nn.Module):
    def __init__(self,in_size):
        super(DecoupleLayer,self).__init__()

        self.in_size = in_size

    def forward(self,x):

        v1 = int(self.in_size/2)
        out1 = x[:,0:v1]
        out2 = x[:,v1:]
        return out1,out2


#returns Positiona nd expected velocity.
#input is P,J_vec,qd in R^mxinput_dim
class OutputLayer(nn.Module):
    def __init__(self,P_size,nj):
        super(OutputLayer,self).__init__()

        self.P_size = P_size
        self.nj = nj

    def forward(self,x):

        m = x.shape[0]
        J_size = int(self.P_size*self.nj)
        P_expect = x[:,0:self.P_size]
        J_expect = x[:,self.P_size:self.P_size+J_size]
        jointVels = x[:,self.P_size+J_size:]

        out = torch.zeros(m,self.P_size*2)
        out[:,0:self.P_size] = P_expect
        vel = torch.zeros(m,self.P_size)

        J_expect_trans = torch.transpose(J_expect,0,1)
        J = torch.reshape(J_expect_trans, (self.P_size, self.nj, m))

        jointVels = torch.reshape(jointVels,(m,self.nj,1))
        J_perm = J.permute(2,0,1)
        vel = torch.bmm(J_perm, jointVels)

        vel = vel[:,:,0].permute(0,1)

        out[:,self.P_size:] = vel
        return out,J

# Inputs:
# - in_size = input size
# - h_sizes = list of neurons in the hiddens layers e.g [30,20] --> two hidden layers with 30 and 20 neurons
# - out_size = output size. The output size is 2 for each angle since it is [sin,cos]
# - act = activation function
class FeedForwardNet(nn.Module):
    def __init__(self,in_size,h_sizes,out_size, act='sigmoid'):
        super(FeedForwardNet,self).__init__()

        self.in_size = in_size
        self.h_sizes = h_sizes
        self.out_size = out_size

        self.hidden = nn.ModuleList()
        self.input = torch.nn.Linear(in_size, h_sizes[0])
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        self.predict = torch.nn.Linear(h_sizes[len(h_sizes) - 1], out_size)

        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        if act == 'softplus':
            self.act = torch.nn.Softplus()
        if act == 'swish':
            self.act = SiLU()
        if act == 'relu':
            self.act = torch.nn.ReLU()

    def forward(self,x):
        x = self.input(x)
        x = self.act(x)

        for k in range(len(self.h_sizes)-1):
            x = self.hidden[k](x)
            x = self.act(x)

        x = self.predict(x)

        return x

    # Computes J as derivatives of output wrt inputs
    def get_jacobian(self, x):

        n = x.size()[0]
        J = torch.zeros(self.out_size,self.in_size,n,device=x.device)
        x.requires_grad = True
        y = self.forward(x)

        for j in range(self.out_size):
            g = torch.autograd.grad(y[:,j],x,retain_graph=True,grad_outputs=torch.ones(n,device=x.device))
            g = g[0].permute(1, 0)
            g = torch.reshape(g,(1,self.in_size,n))
            J[j,:,:] = g

        return J

# Network to obtain the RPY
# It returns both [sin_roll sin_pitch sin_yaw cos_roll cos_pitch cos_yaw] and
# [roll pitch yaw]
# Inputs:
# - in_size = input size
# - h_sizes = list of neurons in the hiddens layers e.g [30,20] --> two hidden layers with 30 and 20 neurons
# - out_size = output size. The output size is 2 for each angle since it is [sin,cos]
# - act = activation function
# - dropout = dropout value
class FeedForwardNet_RPY(nn.Module):
    def __init__(self,in_size,h_sizes,out_size=2,act='sigmoid',dropout=0):
        super(FeedForwardNet_RPY,self).__init__()

        self.in_size = in_size
        self.h_sizes = h_sizes
        self.out_size = out_size

        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.ModuleList()
        self.input = torch.nn.Linear(in_size, h_sizes[0])
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        self.predict = torch.nn.Linear(h_sizes[len(h_sizes) - 1], self.out_size)

        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        if act == 'softplus':
            self.act = torch.nn.Softplus()
        if act == 'swish':
            self.act = SiLU()
        if act == 'relu':
            self.act = torch.nn.ReLU()

    def forward(self,x):
        x = self.input(x)
        x = self.act(x)
        for k in range(len(self.h_sizes)-1):
            x = self.hidden[k](x) #linear transf
            x = self.act(x)
            x = self.dropout(x)

        x = self.predict(x)

        sines = torch.sin(x[:,0].reshape(-1,1))
        cosines = torch.cos(x[:, 1].reshape(-1,1))
        x = torch.hstack((sines,cosines))

        x_angle = torch.atan2(x[:,0],x[:,1])
        x_angle = x_angle.reshape(-1,1)

        return x,x_angle

    # Computes J as derivatives of trigonometric representation wrt inputs
    # J = [dsin_roll/dx ... dsin_yaw/dx dcos_roll/dx ... dcos_yaw/dx]
    # and J_angle
    # J_angle = J = [droll/dx dpitch/dx dyaw/dx]
    def get_jacobian(self, x):

        n = x.size()[0]
        J = torch.zeros(self.out_size,self.in_size,n,device =x.device)
        J_angle = torch.zeros(1, self.in_size, n, device=x.device)
        x.requires_grad = True
        y,y_angle = self.forward(x)

        for j in range(self.out_size):
            g = torch.autograd.grad(y[:,j],x,retain_graph=True,grad_outputs=torch.ones(n,device=x.device))
            g = g[0].permute(1, 0)
            g = torch.reshape(g,(1,self.in_size,n))
            J[j,:,:] = g #in noxnixB

        g_angle = torch.autograd.grad(y_angle[:,0],x,retain_graph=True,grad_outputs=torch.ones(n,device=x.device))
        g_angle = g_angle[0].permute(1, 0)
        g_angle = torch.reshape(g_angle,(1,self.in_size,n))
        J_angle[0,:,:] = g_angle #in noxnixB

        return J,J_angle

# Full Augmented Network to obtain P and RPY
# Inputs:
# - in_size = input size
# - h_sizes = list of neurons in the hiddens layers e.g [30,20] --> two hidden layers with 30 and 20 neurons
# - out_size_P = output size for position.
# - h_sizes_rpy = list of list for hidden layers of each rpy angle
# - act = activation function
# - dropout = dropout value
# - mask = mask vector for [Px Py Pz sin_roll sin_pitch sin_yaw cos_roll cos_pitch cos_yaw]
class Aug_MLP_P_RPY_sincos_unc_indep(nn.Module):
    def __init__(self, in_size,h_sizes, out_size_P,
                 h_sizes_rpy,
                 act='sigmoid',dropout=0,mask = [1, 1, 1, 1, 1, 1,1,1,1]):

        super(Aug_MLP_P_RPY_sincos_unc_indep, self).__init__()

        self.h_sizes = h_sizes
        self.in_size_FF = int(in_size/2) #dimension of joint values
        self.out_size_FF = out_size_P
        self.orient_dim = 2 # [sin cos]

        self.Decouple = DecoupleLayer(in_size) #decouples joint pos and velocities
        self.Pos_FFNet = FeedForwardNet(self.in_size_FF,h_sizes,self.out_size_FF,act=act) #returns P

        self.Roll_FFNet = FeedForwardNet_RPY(self.in_size_FF, h_sizes_rpy[0], 2,
                                              act=act,dropout=dropout)  # returns sin(R), cos(R)

        self.Pitch_FFNet = FeedForwardNet_RPY(self.in_size_FF, h_sizes_rpy[1], 2,
                                              act=act,dropout=dropout)  # returns sin(P), cos(P)

        self.Yaw_FFNet = FeedForwardNet_RPY(self.in_size_FF, h_sizes_rpy[2], 2,
                                              act=act,dropout=dropout)  # returns sin(Y), cos(Y)

        self.mask = mask

    def forward(self,x):
        m = x.shape[0]

        M = torch.as_tensor(self.mask,device=x.device)
        M = M.repeat(m,1)
        M_P = M[:,0:3]
        M_RPY = M[:, 3:]

        JointPos,JointVels = self.Decouple(x)
        jointVels = torch.reshape(JointVels, (m, self.in_size_FF, 1))

        #---- Get position values
        P = self.Pos_FFNet(JointPos) #Pos
        J = self.Pos_FFNet.get_jacobian(JointPos)
        J_P = J.permute(2,0,1) #Jacobian of position
        vel = torch.bmm(J_P, jointVels)
        vel = vel[:,:,0].permute(0,1) #Cartesian velocity

        P = torch.mul(P,M_P)
        vel = torch.mul(vel, M_P)

        P_dim = P.shape[1]

        #---- Initialize Jacobians
        # Total Jacobian for P and RPY, with RPY in trigonometric representation
        J_tot = torch.zeros((m,P_dim+3*self.orient_dim,self.in_size_FF),device=P.device)
        # Jacobian of RPY in tirgonometric representation
        J_RPY = torch.zeros((m, 3 * self.orient_dim, self.in_size_FF), device=P.device)

        # Total Jacobian for P and RPY
        J_tot_angles= torch.zeros((m,P_dim+3,self.in_size_FF),device=P.device)

        #---- Get RPY values
        # Roll Net
        Roll,Roll_angle = self.Roll_FFNet(JointPos)  # sin cos roll
        J,J_Roll_angle = self.Roll_FFNet.get_jacobian(JointPos)
        J_Roll = J.permute(2, 0, 1) #in Bxnixno
        J_Roll_angle = J_Roll_angle .permute(2, 0, 1)

        vel_Roll = torch.bmm(J_Roll, jointVels) #rate of change of roll in trigonometric representation
        vel_Roll = vel_Roll[:, :, 0].permute(0, 1)

        # Pitch Net
        Pitch,Pitch_angle = self.Pitch_FFNet(JointPos)  # sin cos pitch
        J,J_pitch_angle = self.Pitch_FFNet.get_jacobian(JointPos)
        J_Pitch = J.permute(2, 0, 1) #in Bxnixno
        J_Pitch_angle = J_pitch_angle.permute(2, 0, 1)  # in Bxnixno

        vel_Pitch = torch.bmm(J_Pitch, jointVels) #rate of change of pitch in trigonometric representation
        vel_Pitch = vel_Pitch[:, :, 0].permute(0, 1)

        # Yaw Net
        Yaw, Yaw_angle= self.Yaw_FFNet(JointPos)  # sin cos yaw
        J,J_Yaw_angle = self.Yaw_FFNet.get_jacobian(JointPos)
        J_Yaw = J.permute(2, 0, 1) #in Bxnixno
        J_Yaw_angle = J_Yaw_angle.permute(2, 0, 1)

        vel_Yaw = torch.bmm(J_Yaw, jointVels) #rate of change of yaw in trigonometric representation
        vel_Yaw = vel_Yaw[:, :, 0].permute(0, 1)

        #----------------------------
        RPY = torch.hstack((Roll[:,0].reshape(-1,1),Pitch[:,0].reshape(-1,1),Yaw[:,0].reshape(-1,1))) #sines of RPY
        RPY = torch.hstack((RPY,Roll[:, 1].reshape(-1,1), Pitch[:, 1].reshape(-1,1), Yaw[:, 1].reshape(-1,1))) #cosines of RPY

        vel_RPY = torch.hstack((vel_Roll[:, 0].reshape(-1,1), vel_Pitch[:, 0].reshape(-1,1), vel_Yaw[:, 0].reshape(-1,1))) #rates of changes of sin(RPY)
        vel_RPY = torch.hstack((vel_RPY, vel_Roll[:, 1].reshape(-1,1), vel_Pitch[:, 1].reshape(-1,1), vel_Yaw[:, 1].reshape(-1,1))) #rates of changes of cos(RPY)

        J_RPY[:,0,:] = J_Roll[:,0,:] #Jacobian of sin_roll
        J_RPY[:, 1, :] = J_Pitch[:,0,:] #Jacobian of sin_pitch
        J_RPY[:, 2, :] = J_Yaw[:,0,:] #Jacobian of sin_yaw
        #cosines
        J_RPY[:,3,:] = J_Roll[:,1,:] #Jacobian of cos_roll
        J_RPY[:, 4, :] = J_Pitch[:,1,:] #Jacobian of cos_pitch
        J_RPY[:, 5, :] = J_Yaw[:,1,:] #Jacobian of cos_yaw

        J_RPY_angles = torch.hstack((J_Roll_angle,J_Pitch_angle,J_Yaw_angle)) #Jacobian of RPY
        RPY = torch.mul(RPY,M_RPY)
        vel_RPY = torch.mul(vel_RPY, M_RPY) #rates of change of RPY

        out = torch.hstack((P,RPY)) #total output of P and RPY in tirgonometric representation
        twist = torch.hstack((vel, vel_RPY)) #total output of rates of change of P and RPY in tirgonometric representation

        out = torch.cat((out, twist), dim=1) #total output

        out_angles = torch.hstack((P,Roll_angle,Pitch_angle,Yaw_angle)) #total output of P and RPY

        # total Jacobian of P and RPY in tirgonometric representation
        J_tot[:, 0:P_dim, :] = J_P
        J_tot[:, P_dim:, :] = J_RPY

        # total Jacobian of P and RPY
        J_tot_angles[:, 0:P_dim, :] = J_P
        J_tot_angles[:, P_dim:, :] = J_RPY_angles

        #out = [P sinR sinP sinY cosR cosP cosY dP dsinR dsinP dsinY dcosR dcosP dcosY]
        #out_angles = [P RPY]
        return out,J_tot,out_angles,J_tot_angles

def normLoss(RPY_trig,loss_func = torch.nn.MSELoss()):

    n = torch.norm(RPY_trig,dim=1)**2.
    one_vec = 3.*torch.ones(n.shape,device=n.device)
    loss = loss_func(one_vec,n)

    return loss

def normLoss_indep(RPY_trig,loss_func = torch.nn.MSELoss()):

    n = torch.norm(RPY_trig,dim=1)**2.
    one_vec = 1.*torch.ones(n.shape,device=n.device)
    loss = loss_func(one_vec,n)

    return loss


def trainNetBatch_P_RPY_indep(Net,x,y,x_test,y_test,lr,batch_size,mask_RPY,
                         P_dim = 3,
                         Iter = 1e5, Err = 1e-06, Dloss = 1e-08, NormalizeData = True,
                         w_p = 1., w_dp = 1.,w_rpy = 1., w_drpy = 1.,w_norm = 1.):

    # optimizer = torch.optim.SGD(Net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(Net.parameters(), lr=lr)
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                           verbose=False)

    t = 0
    loss_tot = 1

    y_mean = torch.mean(y,dim=0)
    y_std = torch.std(y,dim = 0)

    if NormalizeData:
        y = Normalize(y,y_mean,y_std)

    n_batches = x.shape[0]//batch_size

    if n_batches < 1:
        n_batches = 1

    rem = x.shape[0]%batch_size

    Loss = []
    Loss_P = []
    Loss_dP = []
    Loss_rpy  = []
    Loss_drpy  = []

    task_dim = y.shape[1]//2

    M = torch.as_tensor(mask_RPY, device=x.device)
    M = M.repeat(batch_size, 1)

    M_test = torch.as_tensor(mask_RPY, device=x.device)
    M_test = M_test.repeat(x_test.shape[0], 1)

    while (loss_tot >= Err and t <= Iter):
        loss_agg = 0.
        loss_old = loss_tot

        loss_P_agg = 0
        loss_dP_agg = 0
        loss_rpy_agg = 0
        loss_drpy_agg = 0

        loss_norm_agg = 0

        for i in range(n_batches):

            x_batch = x[i*batch_size:(i+1)*batch_size,:]
            y_batch = y[i*batch_size:(i + 1)*batch_size, :]

            y_pred = Net(x_batch)[0]
            if NormalizeData:
                y_pred = Normalize(y_pred, y_mean, y_std)

            Pose_true = y_batch[:,0:task_dim]
            dPose_true = y_batch[:, task_dim:]
            Pose_pred = y_pred[:,0:task_dim]
            dPose_pred = y_pred[:, task_dim:]

            P_true = Pose_true[:,0:P_dim]
            P_pred = Pose_pred[:,0:P_dim]
            dP_true = dPose_true[:,0:P_dim]
            dP_pred = dPose_pred[:,0:P_dim]

            rpy_true = Pose_true[:,P_dim:]
            rpy_pred = Pose_pred[:,P_dim:] #sin(RPY), cos(RPY)
            drpy_true = dPose_true[:,P_dim:]
            drpy_pred = dPose_pred[:,P_dim:]

            loss_P = loss_func(P_true, P_pred)
            loss_dP = loss_func(dP_true, dP_pred)

            rpy_true = torch.mul(rpy_true, M)
            rpy_pred = torch.mul(rpy_pred, M)
            drpy_true = torch.mul(drpy_true, M)
            drpy_pred = torch.mul(drpy_pred, M)

            loss_norm = normLoss_indep(rpy_pred, loss_func=loss_func)

            loss_rpy = loss_func(rpy_true, rpy_pred)
            loss_drpy = loss_func(drpy_true, drpy_pred)

            loss = w_p*loss_P+w_dp*loss_dP+w_rpy*loss_rpy+w_drpy*loss_drpy+w_norm*loss_norm

            loss_agg = loss_agg+loss.item()
            loss_P_agg = loss_P_agg+loss_P.item()
            loss_dP_agg = loss_dP_agg + loss_dP.item()
            loss_rpy_agg = loss_rpy_agg+loss_rpy.item()
            loss_drpy_agg = loss_drpy_agg + loss_drpy.item()

            loss_norm_agg = loss_norm_agg+loss_norm.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("batch step ",i)


        loss_tot = loss_agg
        d_loss = abs(loss_tot - loss_old)

        scheduler.step(loss_tot)

        Loss.append(loss_tot)
        Loss_P.append(loss_P_agg)
        Loss_dP.append(loss_dP_agg)
        Loss_rpy.append(loss_rpy_agg)
        Loss_drpy.append(loss_drpy_agg)

        t = t + 1
        print("******************")
        print("d loss \t", d_loss)
        print("eoch \t", t)
        print("loss \t", loss_tot,"  loss P: ", loss_P_agg, "  loss dP: ", loss_dP_agg)
        print("loss norm \t", loss_norm_agg, "  loss rpy: ", loss_rpy_agg, "  loss drpy: ", loss_drpy_agg)
        print("lr", optimizer.param_groups[0]["lr"])

        #-----------TEST SET------------
        x_batch = x_test
        y_batch = y_test

        y_pred = Net(x_batch)[0]
        if NormalizeData:
            y_pred = Normalize(y_pred, y_mean, y_std)

        Pose_true = y_batch[:, 0:task_dim]
        dPose_true = y_batch[:, task_dim:]
        Pose_pred = y_pred[:, 0:task_dim]
        dPose_pred = y_pred[:, task_dim:]

        P_true = Pose_true[:, 0:P_dim]
        P_pred = Pose_pred[:, 0:P_dim]
        dP_true = dPose_true[:, 0:P_dim]
        dP_pred = dPose_pred[:, 0:P_dim]

        rpy_true = Pose_true[:, P_dim:]
        rpy_pred = Pose_pred[:, P_dim:]  # sin(RPY), cos(RPY)
        drpy_true = dPose_true[:, P_dim:]
        drpy_pred = dPose_pred[:, P_dim:]

        loss_P = loss_func(P_true, P_pred).item()
        loss_dP = loss_func(dP_true, dP_pred).item()

        rpy_true = torch.mul(rpy_true, M_test)
        rpy_pred = torch.mul(rpy_pred, M_test)
        drpy_true = torch.mul(drpy_true, M_test)
        drpy_pred = torch.mul(drpy_pred, M_test)

        loss_rpy = loss_func(rpy_true, rpy_pred).item()
        loss_drpy = loss_func(drpy_true, drpy_pred).item()

        loss_norm = normLoss(rpy_pred, loss_func=loss_func).item()

        loss = w_p * loss_P + w_dp * loss_dP + w_rpy * loss_rpy + w_drpy * loss_drpy + w_norm * loss_norm

        print("loss test\t", loss,"  loss P test: ", loss_P, "  loss dP test: ", loss_dP)
        print("loss norm test\t", loss_norm, "  loss rpy test: ", loss_rpy, "  loss drpy test: ", loss_drpy)

        if d_loss < Dloss:
            break


    return Net,loss,Loss,Loss_P,Loss_dP,Loss_rpy,Loss_drpy







