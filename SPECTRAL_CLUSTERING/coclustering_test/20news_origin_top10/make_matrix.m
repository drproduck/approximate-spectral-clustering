load('20newsorigin.mat')
f=fopen('topic')
topic=textscan(f,'%s');
topic=topic{1};

idx=zeros(length(gnd),1);
g =[1,5,7,8,11,12,13,14,15,17];
for i = g
    idx=idx+(gnd==i);
end
idx=uint8(idx);
idx=find(idx==1);
size(idx)

fea=fea(idx,:);
gnd=gnd(idx,:);
j=1;
for i=g
    gnd(gnd==i)=j;
    j=j+1;
end

if sum(fea ~= 0, 1) == 0
    disp('null column exists')
end
if sum(fea ~= 0, 2) == 0
    disp('null row exists')
end

save('20newsorigintop10','fea','gnd','vocab','topic');