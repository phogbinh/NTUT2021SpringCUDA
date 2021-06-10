% --------
% Get cluster information
clust = parcluster('local');

% Setting up a job
job = createJob(clust);
 
% Creation of multiple tasks which are each sent to 1 worker, 
for N = 1:4
    % createTask(job, @task, #outputs, {inputs(N)})
    createTask(job, @rand, 1, {N});
end
% (alternatively any sequence of createTask commands)

% Submit job, wait for it to be finished and retrieve results
submit(job);
wait(job, 'finished');
results = fetchOutputs(job);
 
% Destroy the job
delete(job);


% Copyright 2014 The MathWorks, Inc.