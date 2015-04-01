function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta*data;
M = bsxfun(@minus, M, max(M));
M = exp(M);
h = bsxfun(@rdivide, M, sum(M));
cost = - sum(sum( groundTruth.*log(h) )) / numCases ...
    + lambda/2*sum(sum(theta.^2));

% A = groundTruth - h;
% for j = 1:numClasses
%     grad = -mean(bsxfun(@times, data, A(j,:)),2);
%     thetagrad(j,:) = grad';
% end
% thetagrad = thetagrad + lambda*theta;

% 确实是等价的，不过能推导出来真的很牛b啊！
thetagrad = -1/numCases*(groundTruth - h)*data' + lambda*theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
