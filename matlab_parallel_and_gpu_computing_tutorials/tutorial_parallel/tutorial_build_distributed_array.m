spmd
    N = 1;
    mat = repmat([1;2;3], 1, N) + (labindex - 1) * 3;
    
    codistributor = codistributor1d(...
        codistributor1d.unsetDimension, ...
        codistributor1d.unsetPartition, ...
        [3, numlabs * N]);
    distributedArray = codistributed.build(mat, codistributor);
    
    disp(getLocalPart(distributedArray));
end