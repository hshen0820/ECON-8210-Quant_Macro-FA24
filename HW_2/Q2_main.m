%----------------------------------------------------------------
%----------------------------------------------------------------
% 
% Stochastic neoclassical growth model with CRRA utility
% [Question 2] Finite Elements Method using Galerkin Weighting
%
%----------------------------------------------------------------
%----------------------------------------------------------------
%%
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
eta = 1;


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
steady_state = [css lss kss yss]';

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
% 4. Declare state space
%----------------------------------------------------------------

n = 71;
cover_grid = 0.25;    
guess = 1;                    % refine guess with collocation
k_min = kss*(1-cover_grid);
k_max = kss*(1+cover_grid);
nk = n;    
grid_k = k_min + (k_max-k_min)/(n-1)* (0:nk-1)';   % capital grid

dnk = 1001;
dgrid_k = linspace(k_min,k_max,dnk)';    % dense capital grid



%----------------------------------------------------------------
% 4. Finite Elements Method using Galerkin Weights
%----------------------------------------------------------------

% Quadrature for capital
nq = 10;        
xg = zeros(nq, n-1);   % nodes
wg = zeros(nq, n-1);   % weights

for i = 1:n-1
    [xg(:,i),wg(:,i)]= gaussLegendre_quadrature(grid_k(i),grid_k(i+1));
end

%% q2_fixed.m, line 89

options = optimset('Display','Iter','TolFun',1e-15,'TolX',1e-15);

fprintf('\n Guessing initial coefficients.\n');
tic;
ths = coeff_guess(grid_k, nk, Z,shock_num, PI, ...
    beta, eta, alpha, delta, guess, options);
th0 = ths(:);
toc;





fprintf('\n Solving Finite Elements.\n');






%% 
%===============================================================================
%                               FIGURES
%===============================================================================
set(groot,'defaultAxesXGrid','on');
set(groot,'defaultAxesYGrid','on');
set(groot,'defaultAxesBox','on');






%%
%===============================================================================
%                               FUNCTIONS
%===============================================================================

%-------------------------------------------------------------------------------
%  Compute Gauss-Legendre quadrature nodes and weights
%-------------------------------------------------------------------------------
function [x,w]= gaussLegendre_quadrature(a,b)
    
% Gauss-Legendre nodes & weights for n=10 nodes over the interval [a,b]. 

    % nodes
    x = [ 0.1488743389  0.4333953941  0.6794095682  0.8650633666  0.9739065285 ];
    x = [ -x(5:-1:1) x ]';
    % weights
    w = [ 0.2955242247  0.2692667193  0.2190863625  0.1494513491  0.0666713443 ];
    w = [ w(5:-1:1) w ]';

    % transform to interval [a,b]
    x = (b-a)/2 * x + (a+b)/2;
    w = (b-a)/2 * w;
end


%-------------------------------------------------------------------------------
%  Guess Coefficients
%-------------------------------------------------------------------------------
function [theta]= coeff_guess(grid_k, nk, Z, shock_num, PI, ...
    beta, eta, alpha, delta, guess, opt)

    % initial guess: deterministic model w/o labor
    c0 = (1 - alpha*beta) * grid_k.^alpha * exp(Z);
    theta = c0(:);

    % refine guess: collocation with B1-splines
    if (guess == 1)
        th = fsolve(@(theta) err_B1_collocation(theta,grid_k,nk,Z,shock_num,PI), ...
            theta, opt);
        theta = reshape(th, nk, shock_num);
    end
    
    % collocation residuals
    function [res, Chat, L, Kp] = err_B1_collocation(theta,grid_k,nk,Z,shock_num,PI)
        theta = reshape(theta, nk, shock_num);
    
        % c_{t}
        Chat = theta;    
        % l_{t}
        L = ( (1-alpha) * grid_k.^alpha * exp(Z) ) ./ Chat;
        L = L.^(1/(eta+alpha));
        % k_{t+1}
        Kp = grid_k.^alpha.*L.^(1-alpha).*exp(Z) + (1-delta)*grid_k - Chat;
    
        %  compute residuals
        Res = ones(nk,shock_num);
        for iz = 1:shock_num
            % current variables over (k,z) grid
            c = Chat(:,iz);        % c_{t}
            kp = Kp(:,iz);         % k_{t+1}
            
            % future variables over (k'(k;z),z') grid
            % c_{t+1}
            Cp = interp1( grid_k,Chat, kp, 'linear','extrap');
            % l_{t+1}
            Lp = ( (1-alpha) *kp.^alpha *exp(Z) )./Cp;
            Lp = Lp.^(1/(eta+alpha));
            % rtn on capital {t+1}
            Rp = 1 + alpha*(kp./Lp).^(alpha-1) .*exp(Z) - delta;
        
            % Euler eqn error
            Res(:,iz) = 1 - beta* c.*(Rp./Cp)*PI(iz,:)';
        end
        res= Res(:); % vectorize
    end
end