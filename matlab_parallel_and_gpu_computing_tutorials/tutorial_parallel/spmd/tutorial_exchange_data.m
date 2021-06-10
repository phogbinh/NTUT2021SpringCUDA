spmd
    dataToSend = labindex * ones(labindex); % generate different data on e-
                                            % ach worker

    receiverIndex = mod(labindex, numlabs) + 1;
    senderIndex = mod(labindex - 2, numlabs) + 1;   
    dataReceived = labSendReceive(receiverIndex, senderIndex, dataToSend);
    
    disp(dataReceived);
end