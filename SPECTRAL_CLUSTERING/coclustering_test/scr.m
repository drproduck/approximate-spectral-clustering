clear;
% load('circledata_50.mat')
load('spiral.mat')
% fea=fea(randsample(2000, 500), :);
a=fea;
% a = [mvnrnd([1,2],[1,1],200);mvnrnd([5,6],[1,1],200)];

[n,m]=size(a);
knn=2;

d=EuDist2(a,a,true);
d=d+1e12*eye(n);
store=zeros(n,knn);
for i=1:knn
    [~,store(:,i)]=min(d,[],2);
    temp = (store(:,i)-1)*n+(1:n)';
    d(temp) = 1e12; 
end
scatter(a(:,1),a(:,2),20,'black','filled');
for i=1:n
    hold on
    for j=1:knn
        x=a(i,:);
        y=a(store(i,j),:);
        line([x(1),y(1)],[x(2),y(2)],'LineWidth',0.5);
    end
end