function getData_KUKA_v2
close all;
clear all;
DH_table = ...
    [0.5	 pi	 0	 pi/2
    0	 pi	 0	 pi/2
    0.5	 0	 0	 pi/2
    0	 pi	 0	 pi/2
    0.5 	 0	 0	 pi/2
    0	 pi	 0	 pi/2
    0.125	 0	 0	 0];
type = 'rrrrrrr';

T_init = eye(4,4);

Robot = Robot_class(DH_table,type, T_init);
Robot.Visualize(zeros(7,1),false);

q_lims = [-180 180;-120 120; -120 120;-120 120;-120 120;-120 120;-120 120]*pi/180;


T = 40;
T_set = 2;
dt = 100*1e-03;
iter = 1;

n_freq = 10;
N_rand = 30;

rng default;
seed_numbers = randi([1 1000],1,N_rand);

R_old = eye(3);

for N = 1:N_rand
    
    seed_number = seed_numbers(N);
    
    Q_rand = RandomMotion(q_lims,T,T_set,dt,seed_number,n_freq);
    
    Iter_tot = N_rand*length(Q_rand);
    
    figure(10)
    for i = 1:size(Q_rand,1)
        subplot(4,2,i)
        hold on
        plot((1:length(Q_rand)),Q_rand(i,:),'LineWidth',2)
        box on
    end
    
    j = 1;
    
    RPY_old = [0 0 0];
    
    while (j <= length(Q_rand))
        
        qi = Q_rand(:,j);
        
        Pose = Robot.getPose(qi);
        %         Jtot = Robot.getJacobian(qi);
        R = Pose(1:3,1:3);
        P(:,iter) = Pose(1:3,4);
        quat(:,iter) = rotm2quat(R); % w,x,y,z
        axang = rotm2axang(R); % n,angle
        angax(:,iter) = [axang(4);axang(1:3)'];
        
        angax_vect(:,iter) = [axang(4)*axang(1:3)';0];
        
        eulZYX(:,iter) = [fliplr(rotm2eul(R,'ZYX'))';0];
%         RPY_old = fliplr(rotm2eul(R,'ZYX'));
        
        %         alpha_psi(:,iter) = getAngles(axang(1:3));
        %
        %         eulZYX(:,iter) = rotm2eul(R)';
        %
        dR = R*R_old';
        S_w = dR-eye(3);
        w_app = [S_w(3,2)-S_w(2,3), S_w(1,3)-S_w(3,1), S_w(2,1)-S_w(1,2)]/2;
        
        axang_old = rotm2axang(R_old);
        axang_w = rotm2axang(dR);
        w_0(:,iter) = [axang_w(4)*axang_w(1:3)';0];
        R_old = R;
        
        
        RPY_dot = getRPYfromOmega(RPY_old,w_0(1:3,iter));
        RPY = RPY_old+RPY_dot;
        
        eulZYX_num(:,iter) = [RPY';0];
        RPY_old = RPY;
        
        
        dangax(:,iter) = [axang_w(4);axang_w(1:3)'];
        dquat(:,iter) = rotm2quat(dR);
        
        Q(:,iter) = qi;
        %                 dq(:,iter) = qdi*dt;
        %         dP(:,iter) = Jtot(1:3,:)*qdi*dt;
        %
        %         ang_vel(:,iter) = Jtot(4:6,:)*qdi*dt;
        
        %         t = t+dt;
        
        iter = iter+1;
        j = j+1;
        
        disp("iter "+num2str(iter)+"/"+num2str(Iter_tot))
        
    end
    
end

%         Robot.Animate(Q);

%         save("Q",'Q')
%     save("P",'P')
%
noiseP = [0.01;0.01;0.01].*randn(3,length(P));
noiseQ = 0.05*ones(7,1).*randn(7,length(Q));

% %     noisedP = 0*0.01*randn(3,length(dP));
P_noise = P+0*noiseP;
Q_noise = Q+0*noiseQ;
%
dQ = [zeros(length(type),1),diff(Q,[],2)];
dP = [zeros(3,1),diff(P,[],2)];
% dquat = [zeros(4,1),diff(quat,[],2)];
% dangax = [zeros(4,1),diff(angax,[],2)];
eulZYX_sines = sin(eulZYX);
eulZYX_cosines = cos(eulZYX);

deulZYX_sines = [zeros(4,1),diff(eulZYX_sines,[],2)];
deulZYX_cosines = [zeros(4,1),diff(eulZYX_cosines,[],2)];

eul_sincos = [eulZYX_sines(1:3,:);eulZYX_cosines(1:3,:)];
deul_sincos = [deulZYX_sines(1:3,:);deulZYX_cosines(1:3,:)];

for i = 1:4
    
    dangax(i,:) = smooth(dangax(i,:),5);
end
    deulZYX = [zeros(4,1),diff(eulZYX,[],2)];

P_quat = [P;quat];
P_eul = [P;eulZYX];

P_eul_sincos = [P;eul_sincos];
dP_deul_sincos = [dP;deul_sincos];

dP_dquat = [dP;dquat];

P_angax = [P;angax];
dP_deul = [dP;deulZYX];

dP_omega = [dP;w_0];

figure()
sgtitle("P")
for i = 1:3
    subplot(2,2,i)
    plot(1:length(P),P(i,:),'k')
    hold on
    %     plot(1:length(P),P_noise(i,:),'.','Color',[0.7 0.7 0.7])
end
subplot(2,2,4)
plot3(P(1,:),P(2,:),P(3,:),'.')
axis equal

figure()
sgtitle("dP")
for i = 1:3
    subplot(2,2,i)
    plot(1:length(dP),dP(i,:),'k')
end


figure()
sgtitle("w in 0")
for i = 1:3
    subplot(2,2,i)
    hold on
    plot(1:length(w_0),w_0(i,:),'k','LineWidth',2)
end
subplot(2,2,4)
plot3(w_0(1,:),w_0(2,:),w_0(3,:),'.')
axis equal

% figure()
% sgtitle("angax vect")
% for i = 1:4
%     subplot(2,2,i)
%     hold on
%     plot(1:length(angax),angax_vect(i,:),'k','LineWidth',2)
% end

%
%
figure()
sgtitle("eulZYX")
for i = 1:3
    subplot(2,2,i)
    hold on
    plot(1:length(eulZYX),eulZYX(i,:),'k','LineWidth',2)
    plot(1:length(eulZYX),eulZYX_num(i,:),'r','LineWidth',2)
end

figure()
sgtitle("deulZYX")
for i = 1:3
    subplot(2,2,i)
    hold on
    plot(1:length(deulZYX),deulZYX(i,:),'k','LineWidth',2)
end
%
% figure()
% sgtitle("alpah psi")
% for i = 1:2
%     subplot(2,1,i)
%     plot(1:length(alpha_psi),alpha_psi(i,:),'k')
%     hold on
%     plot(1:length(dalpha_psi),dalpha_psi(i,:),'r')
%
% end
%
% figure()
% sgtitle("dalpah psi")
% for i = 1:2
%     subplot(2,1,i)
%     plot(1:length(dalpha_psi),dalpha_psi(i,:),'k')
% end

figure()
sgtitle("angax")
for i = 1:4
    subplot(2,2,i)
    hold on
    plot(1:length(angax),angax(i,:),'k','LineWidth',2)
    plot(1:length(angax),angax_vect(i,:),'r','LineWidth',2)
end

figure()
sgtitle("dangax")
for i = 1:4
    subplot(2,2,i)
    plot(1:length(angax),dangax(i,:),'k')
end

figure()
sgtitle("Quat")
for i = 1:4
    subplot(2,2,i)
    plot(1:length(quat),quat(i,:),'k')
end

figure()
sgtitle("dQuat")
for i = 1:4
    subplot(2,2,i)
    plot(1:length(dquat),dquat(i,:),'k')
end

figure()
sgtitle("q")
for i = 1:size(Q,1)
    subplot(4,2,i)
    hold on
    plot(1:length(Q),Q(i,:))
end

figure()
sgtitle("dq")
for i = 1:size(Q,1)
    subplot(4,2,i)
    hold on
    plot(1:length(dQ),dQ(i,:))
    
end

figure()
for i = 1:size(Q,1)
    hold on
    histogram(Q(i,:))
end

figure()
sgtitle("P")
for i = 1:size(P,1)
    hold on
    histogram(P(i,:))
end

figure()
sgtitle("axang")
for i = 1:4
    hold on
    histogram(angax(i,:))
end

figure()
sgtitle("RPY")
for i = 1:3
    hold on
    histogram(eulZYX(i,:))
end

figure()
sgtitle("w o")
for i = 1:3
    hold on
    histogram(w_0(i,:))
end


% save("P_KUKA","P")
folder = "DataKUKA_sim";
mkdir(folder)
dlmwrite(folder+"/q_KUKA_N_"+num2str(N_rand)+".txt",Q','Delimiter'," ");
dlmwrite(folder+"/dq_KUKA_N_"+num2str(N_rand)+".txt",dQ','Delimiter'," ");
dlmwrite(folder+"/P_quat_KUKA_N_"+num2str(N_rand)+".txt",P_quat','Delimiter'," ");
dlmwrite(folder+"/dP_dquat_KUKA_N_"+num2str(N_rand)+".txt",dP_dquat','Delimiter'," ");
dlmwrite(folder+"/P_angax_KUKA_N_"+num2str(N_rand)+".txt",P_angax','Delimiter'," ");
dlmwrite(folder+"/P_eul_KUKA_N_"+num2str(N_rand)+".txt",P_eul','Delimiter'," ");

dlmwrite(folder+"/dP_deul_KUKA_N_"+num2str(N_rand)+".txt",dP_deul','Delimiter'," ");

% dlmwrite(folder+"/dP_dangax_KUKA_N_"+num2str(N_rand)+".txt",dP_dangax','Delimiter'," ");
dlmwrite(folder+"/dP_omega_KUKA_N_"+num2str(N_rand)+".txt",dP_omega','Delimiter'," ");
% dlmwrite("ang_axis_KUKA.txt",angax','Delimiter'," ");
% dlmwrite("ang_vel_KUKA.txt",ang_vel','Delimiter'," ");

dlmwrite(folder+"/P_eul_sincos_KUKA_N_"+num2str(N_rand)+".txt",P_eul_sincos','Delimiter'," ");
dlmwrite(folder+"/dP_deul_sincos_KUKA_N_"+num2str(N_rand)+".txt",dP_deul_sincos','Delimiter'," ");
end

function RPY_dot = getRPYfromOmega(RPY,omega)
alpha  = RPY(1);
beta  = RPY(2);
gamma  = RPY(3);

T = [cos(gamma)*cos(beta), -sin(gamma) 0;
    sin(gamma)*cos(beta) cos(gamma) 0;
    -sin(beta) 0 1];

I = eye(3,3);
lambda = 1e-0;

T_pinv = pinv(T*T'+lambda*I)*T';

RPY_dot = T_pinv*omega;
RPY_dot = RPY_dot';

end
%alpha, psi
function angles = getAngles(r)

sin_psi = sqrt(r(1).^2+r(2).^2); %take only positive one
cos_psi = r(3);

psi = atan2(sin_psi,cos_psi);
alpha = atan2(r(2),r(1));

angles = [alpha,psi];
end
function coeff = getCoeff()

A = [
    0 0 0 1;
    1 1 1 1;
    0 0 1 0;
    3 2 1 0];
b = [0;1;0;0];

coeff = inv(A)*b;

end

function Q = RandomMotion(q_lims,T,T_set,dt,seed_number,n_freq)

% ampl = [1.5695 2.7535 0.31921 3.2466];
% phase = [-0.49899 0.6879 -1.5914 -1.9745];
rng(seed_number);
% rng(100)


ampl = randn(7,n_freq);
phase = randn(7,n_freq);
t = 0;

t = 0:dt:T;

for j = 1:7
    F(j,:) = zeros(1,length(t));
    for i = 1:n_freq
        F(j,:) = F(j,:)+ ampl(j,i)*cos(2*pi*i/T*t-phase(j,i));
    end
end


% figure()
% plot(1:length(V),V)

max_F = max(F,[],2);
min_F = min(F,[],2);

F = (F-min_F)./(max_F-min_F);

Q = (q_lims(:,2)-q_lims(:,1)).*F+q_lims(:,1);

t_set = 0:dt:T_set;
Q_set_init = Q(:,1)/T_set.*t_set;
Q_set_fin = Q(:,end)-(Q(:,end)/T_set.*t_set);

Q = [Q_set_init,Q,Q_set_fin];


end

function [s,sd] = time_law(t,T,coeff)

val = t/T;
time = [val^3;val^2;val;1];
s = coeff'*time;

time = [3*val^2;2*val;1;0]/T;
sd = coeff'*time;
end