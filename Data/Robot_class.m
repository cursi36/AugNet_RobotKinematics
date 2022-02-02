classdef Robot_class
    
    properties
        
        m_DH_table = [];
        m_n_jnts = [];
        m_n_jnts_ctrl = [];
        m_type = [];
        
        m_T_init = [];
        
        m_NN = [];
        m_n_NN = [];
        
        m_NN_IK = [];
    end
    
    methods
        
        function obj =Robot_class(DH_table,type, T_init)
            [n,~] = size(DH_table);
            obj.m_type  = type;
            obj.m_n_jnts = n;
            obj.m_n_jnts_ctrl = n;
            obj.m_T_init = T_init;
            obj.m_DH_table = DH_table;
        end
        
        function obj = Visualize(obj,q,teach_en)
           VisualizeRobot(obj.m_type,q,obj.m_DH_table,teach_en);

        end
        
        function obj = Animate(obj,q)
           AnimateRobot(obj.m_type,q,obj.m_DH_table);

        end
        
                
        function obj = Plot(obj,q)
          PlotRobot(obj.m_type,q,obj.m_DH_table);

        end
        
        
        %%%%%%%%%%%%%%%%%
        T_DH = DH_mat(DH_params);
        
        %%%%%%%%%%%%%%FKine
        function Pose = getPose(obj,q_ctrl)
            
            Pose = obj.m_T_init;
            
            q = (q_ctrl);
            
            for i = 1:length(q)
                DH_params = obj.m_DH_table(i,:);
                if obj.m_type(i) == 'r'
                    DH_params(2) = DH_params(2)+q(i);
                elseif obj.m_type(i) == 'p'
                    DH_params(1) = DH_params(1)+q(i);
                end
                
                T_i = DH_mat(DH_params);
                Pose = Pose*T_i;
                
            end
        end
        
        function J = getJacobian(obj,q_ctrl)
            
            q = (q_ctrl);
            
            T_ee = obj.getPose(q_ctrl);
            r_ee = T_ee(1:3,4);
            Pose = obj.m_T_init;
            
            J = zeros(6,length(q_ctrl));
            J_tot = zeros(6,length(q));
            
            for i = 1:length(q)
                
                z_0 = Pose(1:3,3);
                
                DH_params = obj.m_DH_table(i,:);
                if obj.m_type(i) == 'r'
                    DH_params(2) = DH_params(2)+q(i);
                elseif obj.m_type(i) == 'p'
                    DH_params(1) = DH_params(1)+q(i);
                end
                
                r_i = Pose(1:3,4);
                r = r_ee-r_i;
                
                if obj.m_type(i) == 'r'
                    J_tot(1:3,i) = cross(z_0,r);
                    J_tot(4:6,i) = z_0;
                elseif obj.m_type(i) == 'p'
                    J_tot(1:3,i) = z_0;
                end
                
                
                T_i = DH_mat(DH_params);
                Pose = Pose*T_i;
                
            end
            
            J = J_tot;
        end
        
        
    end
    
end

function T = DH_mat(DH_params)

% T = zeros(4,4);
d = DH_params(1);    theta = DH_params(2);
a = DH_params(3); alpha = DH_params(4);

T(1,1) = cos(theta); T(1,2) = -sin(theta)*cos(alpha);    T(1,3) = sin(theta)*sin(alpha); T(1,4) = a*cos(theta);
T(2,1) = sin(theta);    T(2,2) = cos(theta)*cos(alpha); T(2,3) = -cos(theta)*sin(alpha);    T(2,4) = a*sin(theta);
T(3,1) = 0; T(3,2) = sin(alpha);    T(3,3) = cos(alpha);    T(3,4) = d;
T(4,1:3) = 0;   T(4,4) = 1;
end

function VisualizeRobot(joint_types,q,DH_tab,teach_en)

n_jnts = length(joint_types);
sigma = zeros(length(joint_types),1);
f_prism = find(joint_types == 'p');
f_rev = find(joint_types == 'r');
sigma(f_prism) = 1;

%d, theta, a, alpha
eps = 1e-15;
%convert to Robotic Toolbox convention
theta = eps*ones(n_jnts,1);
d = eps*ones(n_jnts,1);
offset = zeros(n_jnts,1);

offset(f_rev) = DH_tab (f_rev,2);
offset(f_prism) = DH_tab (f_prism,1);
theta(f_prism) = DH_tab (f_prism,2);
d = DH_tab (:,1);
a = DH_tab (:,3);
alpha = DH_tab (:,4);

% %theta,d,a,alpha,sigma (0 = rev) offset
dh = [theta,d,a,alpha,sigma,offset];

robot_RT = SerialLink(dh,'name',"Robot");
if isempty(f_prism) == 0
    robot_RT.qlim(f_prism,:) = [zeros(length(f_prism),1), 1*ones(length(f_prism),1)];
end

figure()
robot_RT.plot(q')

if teach_en
    teach(robot_RT)
    
end
end

function AnimateRobot(joint_types,q,DH_tab)

n_jnts = length(joint_types);
sigma = zeros(length(joint_types),1);
f_prism = find(joint_types == 'p');
f_rev = find(joint_types == 'r');
sigma(f_prism) = 1;

%d, theta, a, alpha
eps = 1e-15;
%convert to Robotic Toolbox convention
theta = eps*ones(n_jnts,1);
d = eps*ones(n_jnts,1);
offset = zeros(n_jnts,1);

offset(f_rev) = DH_tab (f_rev,2);
offset(f_prism) = DH_tab (f_prism,1);
theta(f_prism) = DH_tab (f_prism,2);
d = DH_tab (:,1);
a = DH_tab (:,3);
alpha = DH_tab (:,4);

% %theta,d,a,alpha,sigma (0 = rev) offset
dh = [theta,d,a,alpha,sigma,offset];

robot_RT = SerialLink(dh,'name',"Robot");
if isempty(f_prism) == 0
    robot_RT.qlim(f_prism,:) = [zeros(length(f_prism),1), 1*ones(length(f_prism),1)];
end

figure()
robot_RT.plot(q')
end

function PlotRobot(joint_types,q,DH_tab)

n_jnts = length(joint_types);
sigma = zeros(length(joint_types),1);
f_prism = find(joint_types == 'p');
f_rev = find(joint_types == 'r');
sigma(f_prism) = 1;

%d, theta, a, alpha
eps = 1e-15;
%convert to Robotic Toolbox convention
theta = eps*ones(n_jnts,1);
d = eps*ones(n_jnts,1);
offset = zeros(n_jnts,1);

offset(f_rev) = DH_tab (f_rev,2);
offset(f_prism) = DH_tab (f_prism,1);
theta(f_prism) = DH_tab (f_prism,2);
d = DH_tab (:,1);
a = DH_tab (:,3);
alpha = DH_tab (:,4);

% %theta,d,a,alpha,sigma (0 = rev) offset
dh = [theta,d,a,alpha,sigma,offset];

robot_RT = SerialLink(dh,'name',"Robot");
if isempty(f_prism) == 0
    robot_RT.qlim(f_prism,:) = [zeros(length(f_prism),1), 1*ones(length(f_prism),1)];
end

robot_RT.plot(q')
end
