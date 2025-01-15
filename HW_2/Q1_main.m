%----------------------------------------------------------------
%----------------------------------------------------------------
% 
% Stochastic neoclassical growth model with CRRA utility
% [Question 1] Spectral Method using Chebychev Polynomials
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
steady_state = [css lss kss yss]';

%----------------------------------------------------------------
% 3. Productivity Shocks (Discretized using Tauchen's Method)
%----------------------------------------------------------------

shock_num = 9;   % number of nodes for technology process Z
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
% 4. Spectral Method using Chebychev Polynomials
%----------------------------------------------------------------

% Find Zeros of the Chebychev Polynomial
p = 6;      % order-p Chebyshev polynomial
ZC = -cos((2*(1:p)'-1) * pi / (2*p));    

% Project collocation points in the K space
grid_k = ((ZC+1)*(k_max-k_min))/2 + k_min;
nk = p;  

% Initial Guess for Chebyshev coefficients
c_init = (1-alpha*beta) * grid_k.^alpha * exp(Z);
theta0 = zeros(p, shock_num);

for iz = 1:shock_num
    theta0(:,iz)= fsolve( @(theta) err_cguess(theta, p, grid_k, nk, ...
                          c_init(:,iz) ), zeros(p,1), ...
                          optimset('Display','off'));
end

theta0 = theta0(:);   % policy function fit

% Solve for Chebyshev coefficients
%%%
%%%



options = optimset('Display','Iter','TolFun',10^(-15),'TolX',10^(-15));
coefs =  fsolve( @(theta) residuals_euler_eqn( theta,p, grid_k,nk, Z,shock_num,PI, ...
                                           beta,sigma,psi,alpha,delta ), theta0, options );
% get coefficients: (p,z)
coefs =  reshape(coefs, p, shock_num);




%===============================================================================
%                               FUNCTIONS
%===============================================================================



%-------------------------------------------------------------------------------
%  Evaluate Chebyshev polynomials
%-------------------------------------------------------------------------------
function T = chebyshev_poly(xg, nx, m)

    % construct polynomial
    T = ones(nx,m);                             % order 0
    T(:,2) = xg;                                % order 1
    for p = 3:m                                % higher orders
        T(:,p)= 2* xg .* T(:,p-1) - T(:,p-2);          
    end    
end

%-------------------------------------------------------------------------------
%  Evaluate policy function error using Chebyshev collocation
%-------------------------------------------------------------------------------
function res = err_cguess(Tcoef, n, grid_k, nk, cpf )
    res =  cpf - chebyshev_poly(grid_k, nk, n) * Tcoef;          % residuals
end

%-------------------------------------------------------------------------------
%  Evaluate Euler Equation Residuals
%-------------------------------------------------------------------------------
function res = residuals_euler_eqn(theta, p, grid_k, nk, Z, shock_num, PI, ...
                                   beta, sigma, psi, alpha, delta)
    theta = reshape(theta, p, shock_num);

    % compute policy functions
    % consumption {t}
    chat = chebyshev_poly(grid_k, nk, p)*theta; 

    % labor {t}
    lpf = ((1-alpha) * grid_k.^alpha * exp(Z)) .* chat.^(-sigma);
    lpf = lpf.^(1/(psi+alpha));

    % capital{t+1}
    kpf = grid_k.^alpha .*lpf.^(1-alpha) .* exp(Z)  +  (1-delta)*grid_k - chat;

    % consumption {t+1}
    Cp =  zeros(nk, shock_num);
    for iz = 1:shock_num                                    % loop over z
        T_kp = chebyshev_poly( kpf(:,iz), nk, p);    % polynomials over k'(k;z)
        Cp(:,iz)= T_kp * theta(:,iz);                 % policy at (k'(k;z);z)
    end


    % Euler equation residuals
    res= NaN(nk, shock_num);    
    for iz= 1:shock_num                                    % loop over z

        % capital {t+1}
        kp= kpf(:,iz);

        % labor {t+1}:  l( k'(k;z), z' )
        Lp= ( (1-alpha) * kp.^alpha * exp(Z) ) .*Cp.^(-sigma);
        Lp= Lp.^(1/(psi+alpha));

        % return on capital {t+1}
        Rp= 1 + alpha*(kp./Lp).^(alpha-1) .*exp(Z) - delta;

        % EE residuals
        res(:,iz)=  chat(:,iz).^(-sigma)  -  beta*( Rp.* Cp.^(-sigma) )*PI(iz,:)';
    end

    res = res(:);

end
