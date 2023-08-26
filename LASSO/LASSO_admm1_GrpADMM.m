
%
% LASSO 
%
% $$ \min_{w,x} \frac{1}{2} \|Cx-d\|_2^2 + \mu \|x\|_1,\quad \mathrm{s.t.}\quad w=x, $$

function [x, out] = LASSO_admm1_GrpADMM(x0, mu, opts)
%%%
global C d

% 迭代准备。
k = 0;
u = x0;
y = x0;
x = x0;
out = struct();
%%%
[~, n] = size(C);
sm = opts.sigma;
psi = opts.psi;

tau = psi/sm;

%符号简记
a = 1/psi;
h = mu*tau;

%%%
% 计算并记录起始点的目标函数值。
nrmC = inf;
f = Func(C, d, mu, x0);
out.fvec = f;
%%%
% Cholesky 分解， $R$ 为上三角矩阵且 $R^\top R=C^\top C + \sigma I_n$。
%  由于罚因子在算法的迭代过程中未变化，事先缓存 Cholesky 分解可以加速迭代过程。
CtC = C'*C;
R = chol(CtC + sm*eye(n));
Ctd = C'*d;
%%%
% 开始时间
tic
out.time = 0;
%% 迭代主循环
% 迭代主循环，当 (1) 达到最大迭代次数或 (2) 自变量 $x$ 的变化量小于阈值时，退出迭代循环。
while k < opts.maxit && nrmC > opts.gtol 
    
    % 凸组合技术
    baru = (1-a)*x + a*u;
    
    %求解$x$子问题
    c = baru - tau*y;
    x = prox(c,h);
    
    %求解$w$子问题
    p = Ctd + sm*x + y;
    w = R \ (R' \ p);
    
    %更新乘子$y$
    y = y + sm * (x - w);
    
    f = Func(C, d, mu, x);
    nrmC = norm(x - w,2);
    
    %%%
    % 输出每步迭代的信息。迭代步 $k$ 加一，记录当前步的函数值。
    if opts.verbose
        fprintf('itr: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
        fprintf('time: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    end
    k = k + 1;
    u = baru;
    out.fvec = [out.fvec; f];
    out.time = [out.time; toc];
end
%%%
out.fval = f;
out.itr = k;
out.time = toc;
out.nrmC = nrmC; 
end
%% 辅助函数
%%%
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
%%%
% LASSO 问题的目标函数 $f(w)=\frac{1}{2}\|Cw-d\|_2^2+\mu \|w\|_1$。
function f = Func(C, d, mu, w)
z = C * w - d;
f = 0.5 * (z' * z) + mu*norm(w, 1);
end