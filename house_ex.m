%% Initcialization
clear; close all; clc

fprintf('Loading the data ...\n');

%% Load Data
data = readmatrix('house_data.csv');
[m,n] = size(data) ;
P = 0.70 ;
idx = randperm(m)  ;
Training = data(idx(1:round(P*m)),:) ; 
Testing = data(idx(round(P*m)+1:end),:) ;

X_train = Training(:, 2:end);
y_train = Training(:, 1);

X_test = Testing(:, 2:end);
y_test = Testing(:, 1);





%% Scale features and set them to zero mean

fprintf('Normalizing Features ...\n');

[X_train mu sigma] = featureNormalize(X_train);


% Add intercept term to X
m1 = length(X_train);
X_train = [ones(m1, 1) X_train];


%% Gradient descent

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 1000;

% Init Theta and Run Gradient Descent 
[m2 n2] = size(X_train);
theta = zeros(n2, 1);
[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


%% Test prices

X_test = (X_test - mu) ./ sigma;

[m3 n3] = size(X_test);

price = [ones(m3, 1) X_test]*theta;


accuracy = 1 / (sum(price ./ y_test) / length(y_test));


fprintf('Test set accuracy using gradient descent:  %f \n', accuracy*100);




%% Normal Equations

fprintf('Solving with normal equations...\n');

data_ne = readmatrix('house_data.csv');
[m,n] = size(data_ne) ;
P = 0.70 ;
idx = randperm(m)  ;
Training_ne = data_ne(idx(1:round(P*m)),:) ; 
Testing_ne = data_ne(idx(round(P*m)+1:end),:) ;

X_train_ne = Training_ne(:, 2:end);
y_train_ne = Training_ne(:, 1);

X_test_ne = Testing_ne(:, 2:end);
y_test_ne = Testing_ne(:, 1);

% Add intercept term to X
m = length(X_train_ne);
X_train_ne = [ones(m, 1) X_train_ne];

% Calculate the parameters from the normal equation
theta_ne = normalEqn(X_train_ne, y_train_ne);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta_ne);
fprintf('\n');



[m3 n3] = size(X_test_ne);

price_ne = [ones(m3, 1) X_test_ne]*theta_ne;

accuracy_ne = 1 / (sum(price_ne ./ y_test) / length(y_test));


fprintf('Test set accuracy using normal equation descent:  %f \n', accuracy_ne*100);

