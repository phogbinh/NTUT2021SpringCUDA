M = 50;
N = 10000;
X_LIM_BEGIN = 23;
X_LIM_END = 27;

tic; a1 = tutorial_serial(M, N); t1 = toc; % running in serial

gcp; % open parallel pool if none is open
tic; a2 = tutorial_parallel(M, N); t2 = toc; % running in parallel

disp( ['Serial processing time: ' num2str(t1)] );
disp( ['Parallel processing time: ' num2str(t2)] );

subplot(1, 2, 1);
histogram(a1, X_LIM_BEGIN:0.2:X_LIM_END), xlim( [X_LIM_BEGIN X_LIM_END] ), title('serial');
subplot(1, 2, 2);
histogram(a2, X_LIM_BEGIN:0.2:X_LIM_END), xlim( [X_LIM_BEGIN X_LIM_END] ), title('parallel');