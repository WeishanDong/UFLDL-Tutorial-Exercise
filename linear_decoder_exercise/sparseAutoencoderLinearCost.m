function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    


% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


m = size(data,2);
x = data;
y = data;

%%%%%%%%%%%%%%%%%%%%%%%%% forward
% disp('forward');
z2 = W1*x + repmat(b1, 1, m);
a2 = sigmoid(z2);
z3 = W2*a2 + repmat(b2, 1, m);
a3 = z3;
JWb = mean(sum((a3 - y).^2)/2);
penalty = sum(sum(W1.^2)) + sum(sum(W2.^2));

rho = mean(a2, 2);
KL = sparsityParam.*log(sparsityParam./rho) ... % not log2!!!!
    + (1-sparsityParam).*log((1-sparsityParam)./(1-rho));

cost = JWb + lambda/2*penalty + beta*sum(KL);

%%%%%%%%%%%%%%%%%%%%%%%%% backward
% disp('backward');
delta3 = -(y - a3);
delta2 = (W2'*delta3 ...
    + repmat(beta*(-sparsityParam./rho + (1-sparsityParam)./(1-rho)), 1, m) ) ...
    .* (a2.*(1-a2));
W2grad = (delta3 * a2')/m + lambda*W2;
b2grad = sum(delta3,2)/m;
W1grad = (delta2 * x')/m + lambda*W1;     % a1 == x
b1grad = sum(delta2,2)/m;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
