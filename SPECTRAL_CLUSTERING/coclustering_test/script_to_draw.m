clear;
load('twomoons_small.mat')
% load('spiral.mat')
% fea=fea(randsample(2000, 500), :);
a=fea;
% a = [mvnrnd([1,2],[1,1],200);mvnrnd([5,6],[1,1],200)];

[n,m]=size(a);
knn=3;

d=EuDist2(a,a,true);
d=d+1e12*eye(n);
store=zeros(n,knn);
val=zeros(n,knn);
for i=1:knn
    [val(:,i),store(:,i)]=min(d,[],2);
    temp = (store(:,i)-1)*n+(1:n)';
    d(temp) = 1e12; 
end
Gidx = repmat((1:n)',1,knn);
Gjdx = store;
val=exp(-val./(2*50^2));
A = sparse(Gidx(:),Gjdx(:),val(:),n,n);
D=sparse(1:n,1:n,sum(A,2).^(-1));
A=D*A;
figure(2);

for p = 1:10
[r,c,v]=find(A^p);

scatter(a(:,1),a(:,2),30,'blue','filled');

%normalize v to range (0,1]
% v=min(v(:))./v;
hold on
x=zeros(2,length(r));
y=zeros(2,length(r));
color=zeros(length(r),3);
for i=1:length(r)
    p1=a(r(i),:);
    p2=a(c(i),:);
%     x(1,i)=p1(1);
%     x(2,i)=p2(1);
%     y(1,i)=p1(2);
%     y(2,i)=p2(2);
%     color(i,3)=v(i);
    line([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',0.1,'Color',[0,0,0,v(i)]);
end
% plot(x,y,'LineWidth',0.1);
hold off
drawnow
pause(0.5)
end