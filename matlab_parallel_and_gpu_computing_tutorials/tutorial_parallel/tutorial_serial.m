function a = tutorial_serial(M, N)
    % `tutorial_serial` performs N trials of computing the largest eigenva-
    % lue for an M-by-M random matrix.
    a = zeros(N, 1);
    for i = 1:N
        a(i) = max(eig(rand(M)));
    end
end