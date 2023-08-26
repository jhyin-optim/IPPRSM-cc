% LASSO
% 
% $$\min_w \mu\|w\|_1+\frac{1}{2}\|Cx-d\|_2^2.$$
% 
% or, equivalenly,
% $$ 
% \begin{array}{rl}
% \displaystyle\min_{w,x} & \hspace{-0.5em}\frac{1}{2}\|Cw-d\|^2_2+\mu\|x\|_1,\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}w=x,
% \end{array} 
% $$

clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

global C d

data = [200 400;300 500;512 1024;512 2048;1024 2048;2000 3000;2000 4000;3000 5000];

[row,~] = size(data);
fid=fopen('mytext.txt','w');

%SCPRSM
opts1.verbose = 0;
opts1.maxit = 5000;
opts1.sigma = 1;
opts1.alpha = 0.8;
opts1.ftol = 1e-3; 
opts1.gtol = 1e-4;

%GrpADMM
opts2.verbose = 0;
opts2.maxit = 5000;
opts2.psi = 1.618;
opts2.sigma = 1;
opts2.ftol = 1e-3; 
opts2.gtol = 1e-4;

%IPPRSM_cc   \theta\in [0,0.618)
opts3.verbose = 0;
opts3.maxit = 5000;
opts3.sigma = 1;
opts3.alpha = 0.99;
opts3.theta = 0.1;
opts3.ftol = 1e-3; 
opts3.gtol = 1e-4;

%IPPRSM_cc   \theta\in [0.618,1)
opts4.verbose = 0;
opts4.maxit = 5000;
opts4.sigma = 1;
opts4.alpha = 0.99;
opts4.theta = 0.618;
opts4.ftol = 1e-3; 
opts4.gtol = 1e-4;

% the model parameter
mu =0.1 ;

for index=1:row
    m = data(index,1);
    n = data(index,2);   
    progress_r = [];
    for repeats=1:10
        p = 100/n;
        u = sprandn(n, 1, p);
        C = randn(m, n);
        d = C * u;
        x0 = zeros(n,1);
        w0 = zeros(n,1);
            
        disp('Starting LASSO_admm1_SCPRSM')
        [~, out1] = LASSO_admm1_SCPRSM(x0, mu, opts1); 
      
        disp('Starting LASSO_admm1_GrpADMM')
        [~, out2] = LASSO_admm1_GrpADMM(x0, mu, opts2);
     
        disp('Starting LASSO_admm1_IPPRSM_cc')  
        [~, out3] = LASSO_admm1_IPPRSM_cc(x0, mu, opts3);
        
        disp('Starting LASSO_admm1_IPPRSM_cc')  
        [~, out4] = LASSO_admm1_IPPRSM_cc(x0, mu, opts4);
        
     
        progress_r=[progress_r; out1.itr out1.time out1.fval out1.nrmC ... 
             out2.itr out2.time out2.fval out2.nrmC ...
             out3.itr out3.time out3.fval out3.nrmC ...
             out4.itr out4.time out4.fval out4.nrmC];
            
    end
    fprintf(fid,'%d & %d & %.0f/%.3f/%.3e/%.3e & %.0f/%.3f/%.3e/%.3e & %.0f/%.3f/%.3e/%.3e & %.0f/%.3f/%.3e/%.3e\\\\\n', ...  
        m,n,mean(progress_r));  
end
fclose(fid);    
