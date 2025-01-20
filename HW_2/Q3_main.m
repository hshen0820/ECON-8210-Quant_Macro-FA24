%----------------------------------------------------------------
%----------------------------------------------------------------
% 
% Stochastic neoclassical growth model with CRRA utility
% [Question 3] 3rd-Order Perturbation
%
%----------------------------------------------------------------
%----------------------------------------------------------------

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
close all

%----------------------------------------------------------------
% 1. Parameterization
%----------------------------------------------------------------

% Technology
alpha = 0.33;                       % Capital Share
beta  = 0.97;                     % Time discount factor
rho = 1/beta - 1;                % Time discount rate
delta = 0.1;                    % Depreciation
sigma = 1; psi = 1;             % CRRA parameters


% Productivity shocks
rho_z = 0.95;                     % Persistence parameter of the productivity shock
sigma_e  = 0.007;                    % S.D. of the productivity shock Z


%----------------------------------------------------------------
% 2. Deterministic Steady State
%----------------------------------------------------------------

y_to_k = (delta+rho)/alpha;
k_to_l = y_to_k ^ (1 / (alpha-1));
css = (k_to_l^alpha - delta*k_to_l) * ((1-alpha)*(k_to_l)^alpha)^(1/psi);
css = css^(1 / (1 + sigma/psi));
lss = ((1-alpha)*(k_to_l)^alpha * css^(-sigma)) ^ (1/psi);
kss = k_to_l * lss;
yss = y_to_k * kss;
steadyState = [css lss kss yss]';

%----------------------------------------------------------------
% 3. Productivity Shocks (Discretized using Tauchen's Method)
%----------------------------------------------------------------

shock_num = 7;   % number of nodes for technology process Z
m = 3;            % max +- 3 std. devs.
sigma_z =  sigma_e / sqrt(1-rho_z^2); % std. dev. of Z
zmax=   m*sigma_z;   zmin=   -m*sigma_z;                             
dz = (zmax-zmin) / (shock_num-1);  % step size
Z = zmin + ((1:shock_num)-1)*dz;   % productivity grid

PI = normcdf((Z + dz/2 - rho_z*Z')/sigma_e) - ... % transition matrix
       normcdf((Z - dz/2 - rho_z*Z')/sigma_e);
PI(:,1) = normcdf( (Z(1) + dz/2 - rho_z*Z')/sigma_e );
PI(:,shock_num) = 1 - normcdf((Z(shock_num) - dz/2 - rho_z*Z')/sigma_e);

if(shock_num == 1)
    Z = 0; PI = 1;
end


%----------------------------------------------------------------
% 3. Declare state space
%----------------------------------------------------------------

% Capital Grid
cover_grid = 0.25;
k_min = kss*(1-cover_grid);
k_max = kss*(1+cover_grid);
dnk = 1001;    % number of grid points for k
grid_k = linspace(k_min,k_max,dnk)';  % capital grid

%----------------------------------------------------------------
% 4. Perturbation
%----------------------------------------------------------------

dynare myModel.mod;

% check steady states
fprintf('\n Check steady states... \n');
disp([steadyState oo_.steady_state(1:4)]);

decisionRule = oo_.dr;           % decision rules
approx_order = options_.order;    % order of Taylor approximation

% State space w.r.t. steady state
dK = grid_k - kss; 
dZ = Z;

% Compute policy functions
pf_name = {'c','l','k'};
pf_idx = zeros(length(pf_name),1);
for i = 1:length(pf_name)
    idx = find(contains( oo_.var_list, pf_name{i} ));
    pf_idx(i) =  find(oo_.dr.order_var == idx);
end

cpf = perturbation_PF(dK, dZ, decisionRule, pf_idx(1), approx_order);
lpf = perturbation_PF(dK, dZ, decisionRule, pf_idx(2), approx_order);
kpf = perturbation_PF(dK, dZ, decisionRule, pf_idx(3), approx_order);


% Simulations
rng(42);
T = 1e5;  T0 = 1e4;  T1 = T+T0;
et = sigma_e * randn(T1,1);
fprintf('\n Simulation.\n');
tic;
[kt_sim, zt_sim]= simulate_states(grid_k, Z, kpf, kss, et, rho_z, T0, T1);
toc;

% Euler equation residuals
resid = zeros(dnk, shock_num);    
for iz = 1:shock_num 

    % k_{t+1}
    kp = kpf(:,iz);     
    dKp = kp - kss;
    % c_{t+1}
    Cp = perturbation_PF(dKp, dZ, decisionRule, pf_idx(1), approx_order);
    % l_{t+1}
    Lp = ((1-alpha) * kp.^alpha * exp(Z)) .* Cp.^(-sigma);    
    Lp = Lp.^(1/(psi+alpha));
    % R_{t+1}
    Rp = 1 + alpha*(kp./Lp).^(alpha-1) .*exp(Z) - delta;

    % EE residuals
    resid(:,iz) = 1 - cpf(:,iz).^(sigma) .* beta.*(Rp.* Cp.^(-sigma))*PI(iz,:)';
end


%% 
%===============================================================================
%                               FIGURES
%===============================================================================
set(groot,'defaultAxesXGrid','on');
set(groot,'defaultAxesYGrid','on');
set(groot,'defaultAxesBox','on');

% Policy Functions
figure(1);
subplot(1,3,1);
plot(grid_k,cpf);
xlabel('k');
ylabel('$c$', 'Interpreter', 'latex');
title('Consumption Decision Rule');
subplot(1,3,2);
plot(grid_k,lpf);
xlabel('k');
ylabel('$l$', 'Interpreter', 'latex');
title('Labor Decision Rule');
subplot(1,3,3);
plot(grid_k,kpf);
xlabel('k');
ylabel('$k''$', 'Interpreter', 'latex');
title('Capital Decision Rule');

% Euler eqn errors
figure(2);
plot(grid_k,resid);
xlabel('k');
ylabel('Euler Residuals');
title('Euler Equation Error (Perturbation)');


% Distribution of Simulated Capital
figure(3);
histogram(kt_sim, 'Normalization', 'probability');
xlabel('Capital');
ylabel('Density');
title('Density of Capital');


%%
%===============================================================================
%                               FUNCTIONS
%===============================================================================


%-------------------------------------------------------------------------------
%  Compute Policy Functions using 3rd-Order Perturbation
%-------------------------------------------------------------------------------


function pf = perturbation_PF(dK, dZ, decisionRule, i, approx_order)

    G0 = decisionRule.ys(decisionRule.order_var);

    % stochastic steady-state correction
    if (approx_order > 1)
        GS2 = decisionRule.ghs2(decisionRule.order_var);
    else
        GS2 = 0*G0;
    end

    % constants
    constants =  G0(i) + 1/2 * GS2(i);

    % 1st-order terms
    G1 = decisionRule.ghx;    % coefficients
    g_k = G1(i,1)*dK;         % approximation terms
    g_z = G1(i,2)*dZ;
    order1 = g_k + g_z;
   
    % 2nd-order terms
    order2 = 0;
    if (approx_order > 1)
        G2 = decisionRule.ghxx;        % coefficients
        g_k2 = G2(i,1) * dK.^2;        % approximation terms
        g_kz = 1/2 * sum(G2(i,[2 3])) * dK * dZ;
        g_z2 = G2(i,4) * dZ.^2;
        order2 = 1/2 * (g_k2 + g_kz + g_z2);
    end

    % 3rd-order terms
    order3 = 0;
    if (approx_order > 2)
        G3 = decisionRule.ghxxx;        % coefficients
        g_k3 = G3(i,1) * dK.^3;         % approximation terms
        g_zk2 = 1/3 * sum(G3(i,[2 3 5])) * dK.^2 *dZ;
        g_kz2 = 1/3 * sum(G3(i,[4 6 7])) * dK * dZ.^2;
        g_z3 = G3(i,8) * dZ.^3;
        order3 = 1/6 * (g_k3 + g_zk2 + g_kz2 + g_z3);
    end    
    
    % aggregate to compute policy function
    pf = constants + order1 + order2 + order3;

end



%-------------------------------------------------------------------------------
%  Simulate States of the Economy
%-------------------------------------------------------------------------------

function [kt,zt] = simulate_states(kgrid, Z, kpf, k0, ...
    et, rho_z, T0, T1)

    [kk,zz] = ndgrid(kgrid,Z);
    Kpf = griddedInterpolant(kk, zz, kpf);
    zt = zeros(T1+1,1);
    kt = zeros(T1+1,1);

    kt(1) = k0;
    for t = 1:T1   
        zt(t+1) = rho_z*zt(t) + et(t);
        kt(t+1) = Kpf(kt(t), zt(t));   % interpolate policy fn to get k_{t+1}
    end
    
    zt = zt(T0+1:T1);
    kt = kt(T0+1:T1);
end

