
function bask_news_test(mat, seed, s, maxt, maxit)

fprintf('Processing %s data set\n', mat);
addpath(genpath('/home/drproduck/Documents/MATLAB/'));
load(mat);
nlabel = max(gnd);
rng(seed);

acc3=zeros(maxit,maxt);
acc4=zeros(maxit,maxt);
acc5 =zeros(maxit, maxt);
acc_d = zeros(maxit,1);

ti3 = zeros(maxit,maxt);
ti4 = zeros(maxit,maxt);
ti5 = zeros(maxit,maxt);
tid = zeros(maxit,1);


for it = 1:maxit
    %common representatives

    % setting up opts
    fprintf('Iteration %d:\n', it);

    opts.s = s;
    % MY ALGORITHM
    [l3,l4,l5,dhillon_acc,t3,t4,t5,dhillon_t] = bask_news(fea, nlabel, maxt, opts);

    ti3(it,:) = t3;
    ti4(it,:) = t4;
    ti5(it,:) = t5;
    tid(it) = dhillon_t;

    for t = 1:maxt
        acc3(it,t) = bestacc(l3(:,t),gnd);
        acc4(it,t) = bestacc(l4(:,t),gnd);
        acc5(it,t) = bestacc(l5(:,t),gnd);
        fprintf('lbdm x %d: %f\n', t*2, acc3(it,t));
        fprintf('lbdm y %d: %f\n',t*2, acc4(it,t));
        fprintf('lbdm odd %d: %f\n', t*2-1, acc5(it,t));
    end   

    acc_d(it) = bestacc(dhillon_acc, gnd);
    fprintf('dhillon: %f\n', acc_d(it));
   
end

% Average accuracy
fprintf('\nAverage accuracy\n')
for t = 1:maxt
    fprintf('lbdm x %d: %f\n', t*2, mean(acc3(:,t)));
    fprintf('lbdm y %d: %f\n',t*2, mean(acc4(:,t)));
    fprintf('lbdm odd %d: %f\n',t*2-1, mean(acc5(:,t)));
end
fprintf('dhillon: %f\n', mean(acc_d));

save(strcat(mat, '_TODAY'), 'acc3','acc4','acc5', ...
    'acc_d',...
    'ti3','ti4','ti5','tid');
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end
