%% initialize
A = magic(5000);
f = ones(1, 20) / 20;

%% cpu
tic;
B = filter(f, 1, A);
tCpu = toc;

disp( ['Total time on CPU: ' num2str(tCpu)] );

%% gpu
tic;
d_A = gpuArray(A);
d_B = filter(f, 1, d_A);
B = gather(d_B);
wait(gpuDevice);
tGpu = toc;

disp( ['Total time on GPU: ' num2str(tGpu)] );

%% gpu -- computational time only
tic;
d_B = filter(f, 1, d_A);
wait(gpuDevice);
tCompGpu = toc;

disp( ['Computational time on GPU: ' num2str(tCompGpu)] );