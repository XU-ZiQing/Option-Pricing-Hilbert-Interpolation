function f = bessel(nu,z)

% This function estimates the bessel function with the specified tolerance
% the generalized gamma series, noted below.  This solves Bessel's
% differential equation:
%
%       x^2*g''(x) + zg'(x) - (z^2+n^2)y= 0
% It has a branch cut discontinuity in the complex z plane running from -Inf to 0.

% Default tolerance. Feel free to change this as needed.
tol = 1e-16;

% Estimates the value by summing powers of the generalized 
% series until the specified tolerance is acheived.

x=(z.^2)./4;
n = 1;
term = x./gamma_com(nu+n+1);
f = 1./gamma_com(nu+1) + term;
nmin = 5;
nmax = 100;

while ( (( n < nmin )|| max((max(abs(term./f)) > tol))) && ( n < nmax ) )
  n = n + 1;
  term = term.*x./(n*nu+n*n);
  f = f + term;
end

f=exp(nu.*log(z/2)).*f;    

end