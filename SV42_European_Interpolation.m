% The code for pricing European call options using Hilbert interpolation 
% in the paper "Analytical Solvability and Exact Simulation in Models 
% with Affine Stochastic Volatility and L\'evy Jumps". Model: 4/2 SV.


clear
warning off
format long


% set parameters
S0 = 100;
X0 = log(S0);
K = 100; % strike
T = 1; 
V0 = 0.04;
a = 0.5;
b = V0-a*V0;
kappa = 1.8;
theta =  0.04;
sigma = 0.2;
r = 0.02;
rho = -0.7;

truePrice = 9.1069; % benchmark

N = 1024e4; % number of simulation paths
num = round(1e8/N); % number of trials
price = zeros(num,1);
stde_price = zeros(num,1);

tic

for temp = 1:num   
    
    
    %% Step 1: generate the V_t conditional on V_0
    t = T;
    D=4*kappa*theta/(sigma^2);
    C0 = (sigma*sigma)*(1-exp(-kappa*t))/(4*kappa);
    if D>1
        LAMBDA=V0*(exp(-kappa*t))/C0;    % non-centrality parameter
        Z=randn(1,N);
        X2=chi2rnd(D-1,1,N);
        Vt=C0*((Z+sqrt(LAMBDA)).^2+X2);
    else
        mu = 2*kappa*exp(-kappa*t).*V0./(sigma^2*(1-exp(-kappa*t)));
        poi = random('Poisson',mu,1,N);
        Vt = C0*chi2rnd(V+2*poi,1,N);
    end        
    
 
    %% Step 2: prepare the conditional characteristic functions
    logVt = log(Vt);
    logV0 = log(V0);
    logVt = min(max(logVt,logV0-6*std(logVt)),logV0+2.6*std(logVt));
    
    L_V = 200; % number of grid points in the Vt dimension
    VU = max(logVt)*1.001; % upper bound of logVt
    VL = min(logVt); % lower bound of logVt
      
    L_V1 = L_V-1;
    alpha = (VU-VL)/2;
    c1 = asinh((VL-logV0)/alpha);
    c2 = asinh((VU-logV0)/alpha);
    VGrid = logV0+alpha*sinh(c2*(0:L_V1)/L_V1+c1*(1-(0:L_V1)/L_V1)); % nonuniform grids
    
    M = 1000; 
    KX = 2*M+1; % number of grid points in the Xt dimension
    XU = X0+log(50); % upper bound of Xt
    XL = X0-log(50); % lower bound of Xt
    XV = XL:(XU-XL)/KX:XU; % uniform grids    
    
    h = 2*pi/(XU-XL); % for other choices of h, need to use FrFFT later     
    MX = 2*M+1; % in order to use fft, we have to match the dimension
    ChF = zeros(MX,L_V); % characteristic function
    X = X0;
    x = exp(logV0);
    y = exp(VGrid);    
    z_ChF = 1i*h*((1:MX)-M-1.5);    
    P_ChF=kappa*kappa-2*sigma*sigma.*(z_ChF.*(a*kappa*rho/sigma-a*a/2)+z_ChF.*z_ChF.*(1-rho*rho)*a*a./2);
    Q_ChF=(kappa*theta-sigma*sigma/2)^2-...
        2*sigma*sigma.*(z_ChF.*(b*sigma*rho/2-b*kappa*theta*rho/sigma-b*b/2)+z_ChF.*z_ChF.*(1-rho*rho)*b*b./2);
     
    for j_vec = 1:M+1
        
        z = z_ChF(j_vec);
        P = P_ChF(j_vec);
        Q = Q_ChF(j_vec);
        temp1=exp(z.*X+z.*(r-kappa*(a*theta-b)*rho/sigma-a*b)*t+z.*z*a*b*(1-rho*rho)*t);
        temp2=exp(z*a*rho.*(y-x)./sigma+z*b*rho.*log(y./x)./sigma);
        temp3=sqrt(P)*sinh(kappa*t/2)./(kappa*sinh(sqrt(P)*t/2));
        temp4=exp((x+y).*(kappa*coth(kappa*t/2)-sqrt(P).*coth(sqrt(P)*t/2))./sigma^2);
        v1=2.*sqrt(Q)./sigma^2;
        z1=2.*sqrt(P).*sqrt(x.*y)./(sigma*sigma.*sinh(sqrt(P)*t/2));
        v2=2*kappa*theta/sigma^2-1;
        z2=2*kappa*sqrt(x.*y)./(sigma*sigma*sinh(kappa*t/2));
        temp5=bessel(v1,z1)./bessel(v2,z2);
        
        ChF(j_vec,:) = temp1.*temp2.*temp3.*temp4.*temp5;        
        
    end
    ChF(M+2:end,:) = conj(ChF(M+1:-1:2,:));
    
  
    %% Step 3: compute the CDF on the grids
    ChF = ChF.*exp(-1i*((1:MX)-M-1.5)'*h*XL)./(pi*((1:MX)-M-1.5)');    
    Fx = 0.5+1i*0.5*fft(ChF).*exp(1i*(0:2*M)'*pi);
    Fx = max(min(real(Fx),1),0);
    
    
    %% Step 4: use interpolation to generate samples of Xt
    cdf = rand(1,N); % uniform r.v.   
    idx = (asinh((logVt-logV0)/alpha)-c1)/(c2-c1)*L_V1+1; % locate Vt in the grid points
    idxL = floor(idx);
    idxU = idxL +1;
    
    X_L = zeros(1,N);
    X_U = zeros(1,N);
    
    [sidxL,ind]=sort(idxL);
    
    indsidxL1 = 1;
    indsidxL2 = 1;
        
    for m = 1:L_V-2        
        if sidxL(indsidxL2)==m
            indsidxL1 = indsidxL2;            
            indsidxL2 = find(sidxL(indsidxL1:min(indsidxL1+N*0.03,N))>m,1)+indsidxL1-1;        
            idxM = ind(indsidxL1:indsidxL2-1);
            
            X_L(idxM) = interp1(Fx(:,m)'+ (-M:M)*1E-12,XV(1:2*M+1),cdf(idxM),'linear');
            X_U(idxM) = interp1(Fx(:,m+1)'+ (-M:M)*1E-12,XV(1:2*M+1),cdf(idxM),'linear');          
        end
    end
    
    m = L_V-1;
    if sidxL(indsidxL2)==m        
        indsidxL1 = indsidxL2;
        indsidxL2 = N;
        idxM = ind(indsidxL1:indsidxL2);
        
        F_L = griddedInterpolant(Fx(:,m)'+ (-M:M)*1E-12,XV(1:2*M+1),'linear');
        X_L(idxM) = F_L(cdf(idxM));
        F_U = griddedInterpolant(Fx(:,m+1)'+ (-M:M)*1E-12,XV(1:2*M+1),'linear');
        X_U(idxM) = F_U(cdf(idxM));        
    end
    
    Xt = X_L.*(idxU-idx)+X_U.*(idx-idxL);
    
    
    %% Step 5: evaluate the option price
    call = exp(-r*T)*max(exp(Xt)-K,0);
    price(temp) = mean(call);
    stde_price(temp) = std(call)/sqrt(N);  
    
end

mean_price = mean(price)
bias = mean_price-truePrice
stde = mean(stde_price)
rmse = sqrt(bias^2+stde^2)
cpu = toc/num



