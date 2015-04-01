function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% M = theta*data;
% M = bsxfun(@minus, M, max(M));
% M = exp(M);
% h = bsxfun(@rdivide, M, sum(M));
% [C,pred] = max(h);

% 上面代码也是正确的，但实际上为求出概率最大值位置，是不需要做归一化的（分母），
% 而且由于exp是单调增函数，也不需要实际做exp，只需要看theta*data最大值位置即可
[C,pred] = max(theta*data);

% ---------------------------------------------------------------------

end

