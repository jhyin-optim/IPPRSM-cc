function out = IPPRSM_cc(H,d,delta,opts)
% out = FJADMM(H,F,mu,opts)
%
% Alternating Directions Method (ADM) applied to TV/L2.
%
% Suppose the data accuquisition model is given by: F = K*Xbar + Noise,
% where Xbar is an original image, K is a convolution matrix, Noise is
% additive noise, and F is a blurry and noisy observation. To recover
% Xbar from F and K, we solve TV/L2 (ROF) model
%
% ***     min_W \sum_i ||Di*W|| + delta/2*||K*W - d||^2      ***
%
% Inputs:
%         H  ---  convolution kernel representing K
%         d  ---  blurry and noisy observation
%         delta ---  model prameter (must be provided by user)
%         opts --- a structure containing algorithm parameters {default}
%                 * opst.sigma    : a positive constant {10}
%                 * opst.alpha   : a constant in (0,1.618] {1.618}
%                 * opst.maxitr  : maximum iteration number {500}
%                 * opst.relchg  : a small positive parameter which controls
%                                  stopping rule of the code. When the
%                                  relative change of X is less than
%                                  opts.relchg, then the code stops. {1.e-3}
%                 * opts.print   : print inter results or not {1}
%
% Outputs:
%         out --- a structure contains the following fields
%                * out.snr   : SNR values at each iteration
%                * out.f     : function valuse at each itertion
%                * out.relchg: the history of relative change in X
%                * out.sol   : numerical solution obtained by this code
%                * out.itr   : number of iterations used
%

% Copyright (c), May, 2009
%       Junfeng Yang, Dept. Math., Nanjing Univiversity
%       Wotao Yin,    Dept. CAAM, Rice University
%       Yin Zhang,    Dept. CAAM, Rice University


[m,n,d3] = size(d);

if d3 == 3
    error('Error, Grayscale image only!');
end

if nargin < 4; opts = []; end
opts = getopts(opts);

C = getC;
[D,Dt] = defDDt;

% initialization
W = d;
Lam1 = zeros(m,n);
Lam2 = Lam1;
sm = opts.sigma;
%tau = opts.tau;
theta = opts.theta;
alpha = opts.alpha;

print = opts.print;

% finite diff
[D1W,D2W] = D(W);
Y1 = D1W;
Y2 = D2W;
U1 = Y1;
U2 = Y2;

f = fval;

out.snr = [];
out.relchg = [];
out.f = f;
tau = (1+alpha)/2+0.001;  % 
tausm = tau*sm;
ctausm = 1 + tausm;

tic
out.time = 0;
%% Main loop
for ii = 1:opts.maxitr
 %% Predictor
  
    % ========================== 
    %  Convex  combination
    % ========================== 
    barU1 =(1-theta)*Y1 + theta*U1;
    barU2 =(1-theta)*Y2 + theta*U2;
    
    % ==================
    %   Shrinkage Step
    % ==================
    X1 = tausm/ctausm*D1W + 1/ctausm*barU1-tau/ctausm*Lam1;
    X2 = tausm/ctausm*D2W + 1/ctausm*barU2-tau/ctausm*Lam2;
    V = X1.^2 + X2.^2;
    V = sqrt(V);
    V(V==0) = 1;
    V = max(V - tau/ctausm, 0)./V;
    Y1 = X1.*V; 
    Y2 = X2.*V; 
 
%     if print
%        fprintf('Iter: %d, snrW: %4.2f',ii,snrW);
%     end
 
    % ==================
    %    Update Lam
    % ==================
    barLam1 = Lam1 + alpha*sm*(Y1 - D1W);
    barLam2 = Lam2 + alpha*sm*(Y2 - D2W);
    
    % ==================
    %    W-subprolem
    % ==================
    Wp = W;
    W = (delta*C.Ktd + Dt(barLam1,barLam2))/sm + Dt(Y1,Y2); 
    W = fft2(W)./(C.eigsDtD + (delta/sm)*C.eigsKtK);
    W = real(ifft2(W));
   
    % finite diff.
    [barD1W,barD2W] = D(W);
    Sum1 =  Y1 - barD1W;
    Sum2 =  Y2 - barD2W;
    
    % ==================
    %    Update Lam
    % ==================
    Lam1 =  barLam1 + alpha*sm*Sum1;
    Lam2 =  barLam2 + alpha*sm*Sum2;
    
    snrW = snr(W);
    out.snr = [out.snr; snrW];
    relchg = norm(W - Wp,'fro')^2/norm(Wp,'fro')^2;
    out.relchg = [out.relchg; relchg];
    
    if print
        fprintf('Iter: %d, snrW: %4.2f, relchg: %4.2e\n',ii,snrW,relchg);
    end
    
    % ====================
    % Check stopping rule
    % ====================
    if relchg < opts.relchg
        out.sol = W;
        out.itr = ii;
        [D1W,D2W] = D(W);
        f = fval;
        out.f = [out.f; f];
        return
    end
    
    % finite diff.
    [D1W,D2W] = D(W);
    f = fval;
    out.f = [out.f; f];
    out.time = [out.time; toc];
    U1 = barU1;
    U2 = barU2;
end
out.sol = W;
out.itr = ii;
out.exit = 'Exist Normally';
%out.time = toc;
% if ii == opts.maxitr
%     out.exit = 'Maximum iteration reached!';
% end

    function opts = getopts(opts)
        
        if ~isfield(opts,'maxitr')%
            opts.maxitr = 500;
        end
        if ~isfield(opts,'sigma')
            opts.sigma = 10;   %10 
        end
         if ~isfield(opts,'theta')
            opts.theta = 0.1;   % 0.01
         end
        if ~isfield(opts,'gamma') %
            opts.alpha = 0.8;  
        end
        if ~isfield(opts,'relchg') %
           opts.relchg = 1.e-6;
        end
        if ~isfield(opts,'print')
           opts.print = 0;
        end
    end


%% nested functions

    function C = getC
        sized = size(d);
        C.eigsK = psf2otf(H,sized); 
        C.Ktd = real(ifft2(conj(C.eigsK) .* fft2(d))); 
        C.eigsDtD = abs(psf2otf([1,-1],sized)).^2 + abs(psf2otf([1;-1],sized)).^2;
        C.eigsKtK = abs(C.eigsK).^2;
    end

    function f = fval
        f = sum(sum(sqrt(D1W.^2 + D2W.^2))); %sum all elements of sqrt(D1X.^2 + D2X.^2)
        KXd = real(ifft2(C.eigsK .* fft2(W))) - d;
        f = f + delta/2 * norm(KXd,'fro')^2;
    end

    function [D,Dt] = defDDt
        % defines finite difference operator D
        % and its transpose operator
        
        D = @(U) ForwardD(U);
        Dt = @(W,X) Dive(W,X);
        
        function [Duw,Dux] = ForwardD(U)
            % Forward finite difference operator
            Duw = [diff(U,1,2), U(:,1) - U(:,end)];
            Dux = [diff(U,1,1); U(1,:) - U(end,:)];
        end
        
        function DtWX = Dive(W,X)
            % Transpose of the forward finite difference operator
            DtWX = [W(:,end) - W(:, 1), -diff(W,1,2)];
            DtWX = DtWX + [X(end,:) - X(1, :); -diff(X,1,1)];
        end
    end

end