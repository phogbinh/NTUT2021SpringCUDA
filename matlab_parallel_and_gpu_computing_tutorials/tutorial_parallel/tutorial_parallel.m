function a = tutorial_parallel(M, N)
    % `tutorial_parallel` performs N trials of computing the largest eigen-
    % value for an M-by-M random matrix.
    a = zeros(N, 1);
    parfor i = 1:N
        a(i) = max(eig(rand(M)));
    end
end