function [count, map, fastaccess] = counter(array, maxlabel, maxcount)
fastaccess = zeros(maxlabel, 1);
map = zeros(maxcount, 1);
count = zeros(maxcount,1);

% map stores the number
% count stores the count
% fastaccess: = 0 means number not seen, = a means number is seen and is
% stored in map(a)

idx = 1;
for i = 1:size(array,1)
    if fastaccess(array(i)) == 0
        fastaccess(array(i)) = idx;
        map(idx) = array(i);
        idx = idx + 1;
    
    end
    where = fastaccess(array(i));
    count(where) = count(where) + 1;
end

count = count(1:idx-1);
map = map(1:idx-1);
[~, order] = sort(map, 'ascend');
map = map(order);
count = count(order);
        