clear;
load('twomoons_small.mat')
nreps=20;
% [lb, reps, ~, VAR] = litekmeans(fea, nreps,'replicates', 10, 'maxiter', 100,...
%     'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
% a=[fea;reps];
%bad landmarks
reps=[145.2479  -76.2282;
   11.5733  104.9832;
  379.6807  -56.0573;
 -196.4738  146.8521;
  255.3684   -1.8322;
  106.6362  226.0479;
  -54.5454  240.1995;
  437.4582   -0.8949;
  313.2814  -95.6969;
  237.9954  -97.4580;
   48.8565  257.3302;
  474.8142   39.8441;
  481.4986   99.7398;
 -139.9801  186.8024;
  149.4275  183.9315;
 -223.3752   60.9102;
 -128.5032  231.6797;
  215.1458  130.6120;
   -3.0494  241.0128;
   33.4184   14.2532];

VAR =...
   1.0e+04 *...
    [2.3703    0.0210    1.7034    0.3882    0.3094    1.5663    0.7816    0.4314...
    1.6300    1.1150    0.8168    0.1534    0.2323    0.4332    0.6592    0.6620...
    0.3989    1.3873    0.5896    1.1664];

%good landmarks
% reps =[-52.3795  239.0579;
%   443.0073    5.4519;
%   241.2760  -94.5590;
%   388.0204  -44.4110;
%   363.1441  -80.3077;
%  -164.0999  170.3924;
%     6.2712  242.4888;
%    30.2976   27.2146;
%   111.9025  219.0763;
%   140.0125  192.5556;
%   482.1724   77.5498;
%   165.2297  -89.9630;
%  -220.3743   93.3090;
%    73.3309  259.7359;
%  -125.4090  223.8095;
%   239.5438  105.9134;
%   317.7405  -71.0971;
%   304.4195 -106.0118;
%   111.0208  -48.7734;
%   180.4314  155.4623];
% 
% VAR =...
%    1.0e+04 *...
%     [0.9075    0.6573    1.2177    0.8514    0.2579    0.6435    0.8529    2.6804...
%     0.9193    0.1826    0.7431    0.6565    1.6914    0.7133    0.6895    1.0352...
%     0.1556    0.5862    0.3603    0.4386];
% 

[n,m]=size(fea);
knn=3;
a=[fea;reps];

d=EuDist2(fea,reps,false);
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
A = sparse(Gidx(:),Gjdx(:),val(:),n,nreps);
W=[zeros(n,n),A;A',zeros(nreps,nreps)];
D=sparse(1:(n+nreps),1:(n+nreps),sum(W,2).^(-1));
A=D*W;


for p = 9
    hold on
    refreshdata
    
%     figure(1)
%     spy(A^p)
%     axis off
    
    
    %normalize v to range (0,1]
    % v=min(v(:))./v;
    hold on
    B=A^p;
    if rem(p,2)
        [r,c,v]=find(B);
        figure(1)

        scatter(fea(:,1),fea(:,2),20,'blue','filled');
        hold on
        scatter(reps(:,1),reps(:,2),60,'red','filled','Marker','o');
        for i=1:length(r)
            p1=a(r(i),:);
            p2=a(c(i),:);
            line([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',0.1,'Color',[0,0,0,v(i)]);
        end
        axis off
    else
        figure(1)
        C=B(1:n,1:n);
        [r,c,v]=find(C);
        scatter(fea(:,1),fea(:,2),20,'blue','filled');
        for i=1:length(r)
            p1=fea(r(i),:);
            p2=fea(c(i),:);
            line([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',0.1,'Color',[0,0,0,v(i)]);
        end
        axis off
        drawnow
        figure(2)
        D=B(n+1:n+nreps,n+1:n+nreps);
        [r,c,v]=find(D);
        scatter(reps(:,1),reps(:,2),60,'red','filled');
        for i=1:length(r)
            p1=reps(r(i),:);
            p2=reps(c(i),:);
            line([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',0.1,'Color',[0,0,0,v(i)]);
        end
        axis off
        drawnow
    end
    
    hold off
%     pause(0.5)
end
