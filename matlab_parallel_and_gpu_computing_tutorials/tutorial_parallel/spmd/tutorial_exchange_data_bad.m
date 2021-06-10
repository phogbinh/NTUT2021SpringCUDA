% --------
%% NOTE OF CAUTION!
% While this approach is easy to understand, it is not the most efficient
% and may even fail with larger data: If the labSend command does not
% complete in time, there may be two workers waiting for each other, a
% so-called deadlock.
% See exchangingData2.m for a more efficient approach that is also suitable
% for larger data.

%% Exchanging data
spmd
    % Generate different data on each worker
    dataToSend = labindex * ones(labindex);
    
    % Sending data
    if labindex < numlabs % all workers but last 
        labSend(dataToSend, labindex+1);
    else
        % last worker sends to first
        labSend(dataToSend, 1) 
    end
    
    % Receiving data
    if labindex > 1 % all workers but first
        dataReceived = labReceive(labindex-1);
    else 
        % first worker receives from last
        dataReceived = labReceive(numlabs); 
    end 
    
    disp(dataReceived)
end

%% Getting data back to client
dataOnClient = dataReceived(:);

% Copyright 2014 The MathWorks, Inc.

%% 20210605 thoughts
% I didn't understand the MATLAB caution. why? resolutions?
% For each worker, is the code block run in sequence? or parallel? any oth-
% er possibilities?
% 20210608 for now I assume it runs in sequence.