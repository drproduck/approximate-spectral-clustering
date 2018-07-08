function colorCode = bestColor(gnd)
k=max(gnd);
colors=distinguishable_colors(k);
colorCode=zeros(length(gnd),3);
for i=1:k
    for j=1:3
        colorCode(gnd==i,j)=colors(i,j);
    end
end
end
