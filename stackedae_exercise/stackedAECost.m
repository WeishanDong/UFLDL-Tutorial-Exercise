function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);  % M-->m by Weishan
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% forward %%%%%%%%%%%%%%%%%%%%%%%%%%%%
nl = numel(stack) + 1;
% a = cell(size(stack) + [1 0]);  
a = cell([nl 1]);   % a{1}==data is not actually filled to save memory
for d = 1:numel(stack)
%     z = stack{d}.w*a{d} + repmat(stack{d}.b, 1, m);
%     a = sigmoid(z);
    if d == 1
        a{d+1} = sigmoid( stack{d}.w*data + repmat(stack{d}.b, 1, m) );
    else
        a{d+1} = sigmoid( stack{d}.w*a{d} + repmat(stack{d}.b, 1, m) );
    end
end


% softmax %%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = softmaxTheta*a{nl};
M = bsxfun(@minus, M, max(M));
M = exp(M);
h = bsxfun(@rdivide, M, sum(M));
cost = - sum(sum( groundTruth.*log(h) )) / m ...
    + lambda/2*sum(sum(softmaxTheta.^2));

% penalty = 0;
% for d = 1:numel(stack)
%     penalty = penalty + sum(sum(stack{d}.w.^2));
% end
% cost = cost + lambda/2*penalty;

softmaxThetaGrad = -1/m*(groundTruth - h)*a{nl}' + lambda*softmaxTheta;


% backward %%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta = cell(size(a));
delta{nl} = - softmaxTheta'*(groundTruth - h) .* (a{nl}.*(1-a{nl}));
for d = nl-1:2
    delta{d} = stack{d}.w'*delta{d+1} .* (a{d}.*(1-a{d}));
end
for d = 1:numel(stack)
    if d == 1
        stackgrad{d}.w = (delta{d+1}*data')/m;% + lambda*stack{d}.w;
    else
        stackgrad{d}.w = (delta{d+1}*a{d}')/m;% + lambda*stack{d}.w;
    end
    stackgrad{d}.b = sum(delta{d+1},2)/m;
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
