//------------------------------------------------------------------------------
// Stochastic neoclassical growth model with CRRA utility
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//  Parameterization
//------------------------------------------------------------------------------

parameters alpha, beta, sigma, psi, delta, rho_z, sigma_e
           rho y_to_k k_to_l;

alpha = 0.33;
beta = 0.97;

sigma = 1;   psi = 1;
delta = 0.1;

rho_z = 0.95;
sigma_e  = 0.007;  

rho = 1/beta - 1;
y_to_k = (delta+rho)/alpha;
k_to_l = y_to_k ^ (1 / (alpha-1));



//------------------------------------------------------------------------------
//  Declare variables.
//------------------------------------------------------------------------------
var     c, l, k, y, z;
varexo  e;


//------------------------------------------------------------------------------
//  optimality conditions
//------------------------------------------------------------------------------

model;

// euler eqn
c^(-sigma) = beta * (alpha*y(+1)/k + (1-delta)) *c(+1)^(-sigma);

// intratemporal condition
l^psi = (1-alpha)* y/l *c^(-sigma);

// resource constraint
c + k =  y + (1-delta)*k(-1);

// output
y =  exp(z) *k(-1)^alpha *l^(1-alpha);

// productivity
z =  rho_z*z(-1) + e;

end;

//------------------------------------------------------------------------------
//  stochastic shocks
//------------------------------------------------------------------------------

shocks;
var e; stderr sigma_e;
end;

// simulations
stoch_simul(order = 3, irf = 0);


//------------------------------------------------------------------------------
//  deterministic steady state
//------------------------------------------------------------------------------
initval;

// steady states
c = (k_to_l^alpha - delta*k_to_l) * ((1-alpha)*(k_to_l)^alpha)^(1/psi);
c = c^(1 / (1 + sigma/psi));
l = ((1-alpha) * k_to_l^alpha * c^(-sigma))^(1/psi);
k = k_to_l * l;
y = y_to_k * k;
e = 0;

end;

