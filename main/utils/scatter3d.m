function scatter3d(fea, gnd);
rotate3d on;

if nargin == 1
    scatter3(fea(:,1), fea(:,2), fea(:,3));
else
scatter3(fea(:,1), fea(:,2), fea(:,3), [], gnd);
end