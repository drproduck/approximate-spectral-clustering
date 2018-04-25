% addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING/'));
% load('circledata_50');
% n = 2000;
% w = EuDist2(fea,fea,false);
% w = 1 ./ exp(w ./ (2.0 * 30^2));
% d = sum(w, 2);
% d = sparse(1:n,1:n,d.^(-0.5));
% l = d * w * d;
% [u,s,v] = eigs(l, 2);
% z = d * u;
% [x,r] = discretize(z);
% scatter(fea(:,1), fea(:,2),[], x(:,1)); 
% label = bestMap(gnd, x(:,1));
% ac = sum(label == gnd) / n

function [label,x,r] = discretize(z)
[n,k] = size(z);
%normalize z
xtil = z ./ (sum(z.^2, 2).^(0.5));

%initialize x
r = zeros(k);
r(:,1) = xtil(randsample(n,1),:)';
c = zeros(n,1);
for i = 2:k
    c = c + abs(xtil * r(:,i-1));
    [~, argminc] = min(c);
    r(:,i) = xtil(argminc,:)';
end

% coordinate descent
eps = 0;
while true
    %find best x
    x = xtil * r;
    [~,label] = max(x,[],2);
    pos = (label-1) * n + [1:n]';
    x = zeros(n,k);
    x(pos) = 1;
    
    %find best r
    [u,s,v] = svd(x' * xtil);
    if abs(trace(s) - eps) < 1e-12
        break
    else
        eps = trace(s);
        r = v * u';
    end
end

end

