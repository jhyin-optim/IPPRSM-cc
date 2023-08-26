
%
% LASSO 
%
% $$ \min_{w,x} \frac{1}{2} \|Cx-d\|_2^2 + \mu \|x\|_1,\quad \mathrm{s.t.}\quad w=x, $$

function [x, out] = LASSO_admm1_GrpADMM(x0, mu, opts)
%%%
global C d

% ����׼����
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

%���ż��
a = 1/psi;
h = mu*tau;

%%%
% ���㲢��¼��ʼ���Ŀ�꺯��ֵ��
nrmC = inf;
f = Func(C, d, mu, x0);
out.fvec = f;
%%%
% Cholesky �ֽ⣬ $R$ Ϊ�����Ǿ����� $R^\top R=C^\top C + \sigma I_n$��
%  ���ڷ��������㷨�ĵ���������δ�仯�����Ȼ��� Cholesky �ֽ���Լ��ٵ������̡�
CtC = C'*C;
R = chol(CtC + sm*eye(n));
Ctd = C'*d;
%%%
% ��ʼʱ��
tic
out.time = 0;
%% ������ѭ��
% ������ѭ������ (1) �ﵽ������������ (2) �Ա��� $x$ �ı仯��С����ֵʱ���˳�����ѭ����
while k < opts.maxit && nrmC > opts.gtol 
    
    % ͹��ϼ���
    baru = (1-a)*x + a*u;
    
    %���$x$������
    c = baru - tau*y;
    x = prox(c,h);
    
    %���$w$������
    p = Ctd + sm*x + y;
    w = R \ (R' \ p);
    
    %���³���$y$
    y = y + sm * (x - w);
    
    f = Func(C, d, mu, x);
    nrmC = norm(x - w,2);
    
    %%%
    % ���ÿ����������Ϣ�������� $k$ ��һ����¼��ǰ���ĺ���ֵ��
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
%% ��������
%%%
% ���� $h(x)=\mu\|x\|_1$ ��Ӧ���ڽ����� $\mathrm{sign}(x)\max\{|x|-\mu,0\}$��
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
%%%
% LASSO �����Ŀ�꺯�� $f(w)=\frac{1}{2}\|Cw-d\|_2^2+\mu \|w\|_1$��
function f = Func(C, d, mu, w)
z = C * w - d;
f = 0.5 * (z' * z) + mu*norm(w, 1);
end