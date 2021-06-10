%% settings
M = 300; % output -- #rows
K = 500; % matrix multiply inner dimension
N = 100; % output -- #columns
P = 200; % #pages

%% cpu
tic;
A = rand(M, K);
B = rand(K, N, P);
C = zeros(M, N, P);
for i = 1:P
    C(:, :, i) = A * B(:, :, i);
end
tCpu = toc;
disp( ['cpu time: ' num2str(tCpu)] );

%% gpu
tic;
d_A = rand(M, K, 'gpuArray');
d_B = rand(K, N, P, 'gpuArray');
d_C = zeros(M, N, P, 'gpuArray');
for i = 1:P
    d_C(:, :, i) = d_A * d_B(:, :, i);
end
wait(gpuDevice);
tGpu = toc;
disp( ['gpu time: ' num2str(tGpu)] );

%% improved gpu
tic;
d_A = rand(M, K, 'gpuArray');
d_B = rand(K, N, P, 'gpuArray');
d_C = pagefun(@mtimes, d_A, d_B);
wait(gpuDevice);
tImprovedGpu = toc;
disp( ['improved gpu time: ' num2str(tImprovedGpu)] );