N = 50;
M = 10000;

% step 1. submit batch job
clust = parcluster('local');
WORKERS_N = 12; % my computer has 12 workers in total (tested it with matl-
                % interactive parallel pool) <- 20210605 but is this corre-
                % ct? resource monitor gives us a view of 16 cpu in total.
job = batch(clust, @tutorial_parallel, 1, {N, M}, 'Pool', WORKERS_N - 1);

get(job, 'State'); % query state of job

% step 2. wait for job to be finished
wait(job, 'finished');

% step 3. retrieve and process results
results = fetchOutputs(job);
a = results{1};
histogram(a);

% step 4. delete job
delete(job);