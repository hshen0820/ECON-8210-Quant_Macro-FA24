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

PF = oo_.dr;                      % policy functions
approx_order = options_.order;    % order of Taylor approximation

% State space w.r.t. steady state
dK = grid_k - kss; 
dZ = Z;