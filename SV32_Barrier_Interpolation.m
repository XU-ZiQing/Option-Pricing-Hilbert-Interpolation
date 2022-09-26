% The code for pricing Bermudan put options using Hilbert interpolation
% in the paper "Analytical Solvability and Exact Simulation in Models 
% with Affine Stochastic Volatility and L\'evy Jumps". Model: 3/2 SV.

clear
warning off
format long


% set parameters
S0 = 100;
X0 = log(S0);
K = 100; % strike
SU = 125; % upper bound for barrier option
SL = 75; % lower bound for barrier option
V0 = 1/0.060025;
kappa = 22.84*0.4669^2;
theta = (8.56^2+22.84)/4.9790;
sigma = -8.56;
r = 0.0048;
rho = -0.99;
T = 1;
AF = 12; % annualization factor

truePrice = 1.9580; % benchmark

N =1024e4; % number of simulation paths
num = round(1e8/N); % number of trials
put_Price = zeros(1,num);
stde_Price = zeros(1,num);

tic
for j = 1:num   
      
    t = 1/AF;
    steps = T/t;
    Xt = ones(steps+1,N)*X0;    
    
    %% Step 1: simulation of the variance process
    V=4*kappa*theta/(sigma*sigma);
    Vt = ones(N,steps+1)*V0;
    D=4*kappa*theta/(sigma^2);
    C0 = (sigma*sigma)*(1-exp(-kappa*t))/(4*kappa);
    
    if D>1
        for temp = 1:steps
            LAMBDA=Vt(:,temp)*(exp(-kappa*t))/C0;    % non-centrality parameter
            Z=randn(N,1);
            X2=chi2rnd(D-1,N,1);
            Vt(:,temp+1)=C0*((Z+sqrt(LAMBDA)).^2+X2);
        end
    else
        for temp = 1:steps
            mu = 2*kappa*exp(-kappa*t).*Vt(:,temp)./(sigma^2*(1-exp(-kappa*t)));
            poi = random('Poisson',mu,N,1);
            Vt(:,temp+1) = C0*chi2rnd(V+2*poi,N,1);
        end
    end
    
    
    %% Step 2: prepare the conditional characteristic functions
    % truncation in the variance domain
    logVt = log(Vt);
    logV0 = log(V0);
    logVt = max(min(logVt,logV0 + 2.7*std(logVt(:,end))),logV0-6*std(logVt(:,end)));
    logVt = logVt';
    
    % Non-uniform grids in the variacne domain
    L_V = 100; % discretization in sigmat
    VU = max(max(logVt-min(min(logVt))))*1.001+min(min(logVt)); % upper bound of sigmat
    VL = min(min(logVt)); % lower bound of sigmat
    L_V1 = L_V-1;
    alpha = (VU-VL)/20; % grid multiplier
    logVm = log(theta);
    c1 = asinh((VL-logVm)/alpha);
    c2 = asinh((VU-logVm)/alpha);
    VGrid = logVm+alpha*sinh(c2*(0:L_V1)/L_V1+c1*(1-(0:L_V1)/L_V1));
    
    % Non-uniform grids in the Xt domain
    M = 500;
    KX = 2*M+1;  % number of grid points in the Xt dimension    
    XU = log(3);
    XL = log(0.3);      
    alphaX = (XU-XL)/10; 
    XM = (XU+XL)/2;
    c1X = asinh((XL-XM)/alphaX);
    c2X = asinh((XU-XM)/alphaX);
    XV = XM+alphaX*sinh(c2X*(0:KX)/KX+c1X*(1-(0:KX)/KX));
    XV = XV(1:end-1);
    
    h = 0.8;         
    MX = 2500;
    ChFM = zeros(MX, L_V, L_V); % characteristic function
    x = exp(VGrid)';
    z = 1i*h*((1:MX)-0.5);
    Q=(kappa*theta-sigma*sigma/2)^2-2*sigma*sigma.* (z.* ...
        (sigma*rho/2-kappa*theta*rho/sigma-1/2)+z.*z.*(1-rho*rho)/2);
            
    for y_vec = 1:length(VGrid)
        
        y = exp(VGrid(y_vec));
        temp1=exp(z.*(r+kappa*rho/sigma)*t);
        temp2=exp(z*rho.*log(y./x)./sigma);
        temp3=kappa*sinh(kappa*t/2)./(kappa*sinh(kappa*t/2));
        temp4=exp((x+y).*(kappa*coth(kappa*t/2)-kappa.*coth(kappa*t/2))./sigma^2);
        v1=2.*sqrt(Q)./sigma^2;
        z1=2*kappa*sqrt(x.*y)./(sigma*sigma*sinh(kappa*t/2));
        v2=2*kappa*theta/sigma^2-1;
        z2=z1;
        temp5=bessel(v1,z1)./besseli(v2,z2);
        
        ChFM(:,:,y_vec) =transpose(temp1.*temp2.*temp3.*temp4.*temp5);
        
    end    
    
    
    %% Step 3: compute the CDF on the grids
    FxM = zeros(KX,L_V,L_V);    
        
    mv = (1:MX)'-0.5;
    for M_vec = 1:L_V
        ChF = ChFM(:,:,M_vec);
        R = real(ChF);
        I = imag(ChF);
        FxM(:,:,M_vec) = 1/2-1/pi*(cos(h*XV'*mv')*(I./mv)-sin(h*XV'*mv')*(R./mv));
    end    
    
    FxM = max(min(real(FxM),1),0);
    
    
    %% Prepare the conditional characteristic functions in the first stage
    ChF = zeros(MX,L_V);
    x = exp(logV0);
    y = exp(VGrid);
    z_ChF = 1i*h*((1:MX)-0.5);
    Q_ChF=(kappa*theta-sigma*sigma/2)^2-2*sigma*sigma.* (z_ChF.* ...
        (sigma*rho/2-kappa*theta*rho/sigma-1/2)+z_ChF.*z_ChF.*(1-rho*rho)/2);
    
    for j_vec = 1:MX
        
        z = z_ChF(j_vec);
        Q = Q_ChF(j_vec);
        temp1=exp(z.*(r+kappa*rho/sigma)*t);
        temp2=exp(z*rho.*log(y./x)./sigma);
        temp3=kappa*sinh(kappa*t/2)./(kappa*sinh(kappa*t/2));
        temp4=exp((x+y).*(kappa*coth(kappa*t/2)-kappa.*coth(kappa*t/2))./sigma^2);
        v1=2.*sqrt(Q)./sigma^2;
        z1=2*kappa*sqrt(x.*y)./(sigma*sigma*sinh(kappa*t/2));
        v2=2*kappa*theta/sigma^2-1;
        z2=z1;
        temp5=bessel(v1,z1)./besseli(v2,z2);
        
        ChF(j_vec,:) =temp1.*temp2.*temp3.*temp4.*temp5;
        
    end       
    
    mv = (1:MX)'-0.5;
    R = real(ChF);
    I = imag(ChF);
    Fx = 1/2-1/pi*(cos(h*XV'*mv')*(I./mv)-sin(h*XV'*mv')*(R./mv));
    Fx = max(min(real(Fx),1),0);    
    
    
    %% Step 4: use interpolation to generate samples of Xt
    % Interpolation in the first step
    cdf = rand(1,N);
    idx = (asinh((logVt(2,:)-logVm)/alpha)-c1)/(c2-c1)*L_V1+1;
    idxL = floor(idx);
    idxU = idxL +1;
    
    X_L = zeros(1,N);
    X_U = zeros(1,N);
    
    [sidxL,ind]=sort(idxL);
    
    indsidxL1 = 1;
    indsidxL2 = 1;
    for m = 1:sidxL(end)-1
        if sidxL(indsidxL2)==m
            indsidxL1 = indsidxL2;
            indsidxL2 = find(sidxL(indsidxL1:min(indsidxL1+N*0.1,N))>m,1)+indsidxL1-1;
            idxM = ind(indsidxL1:indsidxL2-1);
            cdf(idxM) = min(max(cdf(idxM),max(Fx(1,m+1),Fx(1,m))),min(Fx(end,m),Fx(end,m+1)));
            
            X_L(idxM) = interp1(Fx(:,m)'+ (-M:M)*1E-11,XV,cdf(idxM),'linear');
            X_U(idxM) = interp1(Fx(:,m+1)'+ (-M:M)*1E-11,XV,cdf(idxM),'linear');         
        end        
    end
    
    m = sidxL(end);
    if sidxL(indsidxL2)==m
        indsidxL1 = indsidxL2;
        indsidxL2 = N;
        idxM = ind(indsidxL1:indsidxL2);
        cdf(idxM) = min(max(cdf(idxM),max(Fx(1,m+1),Fx(1,m))),min(Fx(end,m),Fx(end,m+1)));
        
        
        X_L(idxM) = interp1(Fx(:,m)'+ (-M:M)*1E-11,XV,cdf(idxM),'linear');
        X_U(idxM) = interp1(Fx(:,m+1)'+ (-M:M)*1E-11,XV,cdf(idxM),'linear');        
    end
    
    Xt(2,:) = Xt(1,:)+X_L.*(idxU-idx)+X_U.*(idx-idxL);
    
    % Interpolation in the following steps
    logVu = logVt(2:steps,:)';
    logVt = logVt(3:end,:)';
    logVu = logVu(:)';
    logVt = logVt(:)';
    cdf = rand(1,(steps-1)*N);
    idxu = (asinh((logVu-logVm)/alpha)-c1)/(c2-c1)*L_V1+1;
    idxuL = floor(idxu);
    idxuU = idxuL+1;
    [sidxuL, indu]=sort(idxuL);
    indsidxuL1 = 1;
    indsidxuL2 = 1;
    dXL = zeros(1,(steps-1)*N);
    dXU = dXL;
    
    for mu = 1:sidxuL(end)-1
        if sidxuL(indsidxuL2)==mu
            indsidxuL1 = indsidxuL2;
            indsidxuL2 = find(sidxuL(indsidxuL1:min(indsidxuL1+(steps-1)*N*0.05,(steps-1)*N))>mu,1)+indsidxuL1-1;
            idxuM = indu(indsidxuL1:indsidxuL2-1);
            cdfL = cdf(idxuM);
            cdfU = cdf(idxuM);
            N_u = indsidxuL2-indsidxuL1;
            
            idx = (asinh((logVt(idxuM)-logVm)/alpha)-c1)/(c2-c1)*L_V1+1;
            idxL = floor(idx);
            idxU = idxL+1;
            XL_L = zeros(1,N_u);
            XL_U = zeros(1,N_u);
            XU_L = zeros(1,N_u);
            XU_U = zeros(1,N_u);
            [sidxL,ind]=sort(idxL);
            FxL = FxM(:,mu,:);
            FxU = FxM(:,mu+1,:);
            
            indsidxL1 = 1;
            indsidxL2 = 1;
            if length(sidxL)>0
                for m = 1:sidxL(end)-1
                    if sidxL(indsidxL2)==m
                        indsidxL1 = indsidxL2;
                        indsidxL2 = find(sidxL(indsidxL1:min(indsidxL1+N_u*0.3,N_u))>m,1)+indsidxL1-1;
                        idxM = ind(indsidxL1:indsidxL2-1);
                        cdfL(idxM) = min(max(cdfL(idxM),max(FxL(1,m+1),FxL(1,m))),min(FxL(end,m),FxL(end,m+1)));
                        cdfU(idxM) = min(max(cdfU(idxM),max(FxU(1,m+1),FxU(1,m))),min(FxU(end,m),FxU(end,m+1)));
                        
                        XL_L(idxM) = interp1(FxL(:,m)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                        XL_U(idxM) = interp1(FxL(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                        XU_L(idxM) = interp1(FxU(:,m)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                        XU_U(idxM) = interp1(FxU(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                    end
                end
                
                m = sidxL(end);
                if sidxL(indsidxL2)==m
                    indsidxL1 = indsidxL2;
                    indsidxL2 = N_u;
                    idxM = ind(indsidxL1:indsidxL2);
                    cdfL(idxM) = min(max(cdfL(idxM),max(FxL(1,m+1),FxL(1,m))),min(FxL(end,m),FxL(end,m+1)));
                    cdfU(idxM) = min(max(cdfU(idxM),max(FxU(1,m+1),FxU(1,m))),min(FxU(end,m),FxU(end,m+1)));
                                       
                    XL_L(idxM) = interp1(FxL(:,m)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                    XL_U(idxM) = interp1(FxL(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                    XU_L(idxM) = interp1(FxU(:,m)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                    XU_U(idxM) = interp1(FxU(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                end
            end
            
            dXL(idxuM) = XL_L.*(idxU-idx)+XL_U.*(idx-idxL);
            dXU(idxuM) = XU_L.*(idxU-idx)+XU_U.*(idx-idxL);
            
        end
    end
    
    mu = sidxuL(end);
    if sidxuL(indsidxuL2)==mu
        indsidxuL1 = indsidxuL2;
        indsidxuL2 = (steps-1)*N;
        idxuM = indu(indsidxuL1:indsidxuL2);
        cdfL = cdf(idxuM);
        cdfU = cdf(idxuM);
        N_u = indsidxuL2-indsidxuL1+1;
        
        idx = (asinh((logVt(idxuM)-logVm)/alpha)-c1)/(c2-c1)*L_V1+1;
        idxL = floor(idx);
        idxU = idxL+1;
        XL_L = zeros(1,N_u);
        XL_U = zeros(1,N_u);
        XU_L = zeros(1,N_u);
        XU_U = zeros(1,N_u);
        [sidxL,ind]=sort(idxL);
        FxL = FxM(:,mu,:);
        FxU = FxM(:,mu+1,:);
        
        indsidxL1 = 1;
        indsidxL2 = 1;
        if length(sidxL)>0
            for m = 1:sidxL(end)-1
                if sidxL(indsidxL2)==m
                    indsidxL1 = indsidxL2;
                    indsidxL2 = find(sidxL(indsidxL1:min(indsidxL1+N_u*0.3,N_u))>m,1)+indsidxL1-1;
                    idxM = ind(indsidxL1:indsidxL2-1);
                    cdfL(idxM) = min(max(cdfL(idxM),max(FxL(1,m+1),FxL(1,m))),min(FxL(end,m),FxL(end,m+1)));
                    cdfU(idxM) = min(max(cdfU(idxM),max(FxU(1,m+1),FxU(1,m))),min(FxU(end,m),FxU(end,m+1)));
                    
                    XL_L(idxM) = interp1(FxL(:,m)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                    XL_U(idxM) = interp1(FxL(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                    XU_L(idxM) = interp1(FxU(:,m)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                    XU_U(idxM) = interp1(FxU(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                end
            end
            
            m = sidxL(end);
            if sidxL(indsidxL2)==m
                indsidxL1 = indsidxL2;
                indsidxL2 = N_u;
                idxM = ind(indsidxL1:indsidxL2);
                cdfL(idxM) = min(max(cdfL(idxM),max(FxL(1,m+1),FxL(1,m))),min(FxL(end,m),FxL(end,m+1)));
                cdfU(idxM) = min(max(cdfU(idxM),max(FxU(1,m+1),FxU(1,m))),min(FxU(end,m),FxU(end,m+1)));
                
                XL_L(idxM) = interp1(FxL(:,m)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                XL_U(idxM) = interp1(FxL(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfL(idxM),'linear');
                XU_L(idxM) = interp1(FxU(:,m)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
                XU_U(idxM) = interp1(FxU(:,m+1)'+ (-(M):(M))*1E-11,XV,cdfU(idxM),'linear');
            end
        end
        
        dXL(idxuM) = XL_L.*(idxU-idx)+XL_U.*(idx-idxL);
        dXU(idxuM) = XU_L.*(idxU-idx)+XU_U.*(idx-idxL);
        
    end
    dXt = dXL.*(idxuU-idxu)+dXU.*(idxu-idxuL);
    
    % indicator part in the payoff
    idxBarrier = (Xt(2,:)<log(SU)).*(Xt(2,:)>log(SL)); 
    for temp = 2:steps
        Xt(temp+1,:) = Xt(temp,:)+dXt(1+N*(temp-2):N*(temp-1));
        idxBarrier = idxBarrier.*(Xt(temp+1,:)<log(SU)).*(Xt(temp+1,:)>log(SL));
    end
    
    
    %% Step 5: evaluate the option price
    payoff = exp(-r*T)*max(K-exp(Xt(end,:)),0).*idxBarrier;
    put_Price(j) = mean(payoff);
    stde_Price(j) = std(payoff)/sqrt(N);
    

end

path = N
bias = mean(put_Price)-truePrice
stde = mean(stde_Price)
rmse = sqrt(bias^2+stde^2)
cpu = toc/num