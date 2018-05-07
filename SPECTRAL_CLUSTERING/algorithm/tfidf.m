function fea = tfidf(fea)
%  fea is a document-term frequency matrix, this function return the tfidf ([1+log(tf)]*log[N/df])
%  weighted document-term matrix.

[n,m] = size(fea);
[idx,jdx,vv] = find(fea);
df = full(sum(sparse(idx,jdx,1),1));

df(df==0) = 1;
idf = log(n./df);

fea = sparse(idx,jdx,log(vv)+1,n,m);

% fea = fea .* idf;

fea = fea';
idf = idf';

MAX_MATRIX_SIZE = 5000; % You can change this number based on your memory.
nBlock = ceil(MAX_MATRIX_SIZE*MAX_MATRIX_SIZE/m);
for i = 1:ceil(n/nBlock)
    if i == ceil(n/nBlock)
        smpIdx = (i-1)*nBlock+1:n;
    else
        smpIdx = (i-1)*nBlock+1:i*nBlock;
    end
    fea(:,smpIdx) = fea(:,smpIdx) .* idf(:,ones(1,length(smpIdx)));
end

fea = fea';