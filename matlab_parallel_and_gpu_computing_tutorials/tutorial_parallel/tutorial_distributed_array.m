Z = zeros(4, 4, 'distributed');

%% convert to distributed array
A = reshape(1:32, 4, 8);
A = distributed(A);
spmd
    disp(getLocalPart(A));
end

%% convert to numeric array
A = gather(A);