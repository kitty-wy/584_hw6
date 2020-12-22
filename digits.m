clc; clear; close all

data_dir = './';
fig_dir = './figs/';

fn_test = 't10k-images-idx3-ubyte';
fn_test_label = 't10k-labels-idx1-ubyte';
fn_train = 'train-images-idx3-ubyte';
fn_train_label = 'train-labels-idx1-ubyte';

%% read training images + labels
fid_train = fopen([data_dir, fn_train]); % images
[magic_train, nims_train, nrows_train, ncols_train, M_train]...
    = read_images(fid_train);

fid_train_label = fopen([data_dir, fn_train_label]); % labels
[magic_train_l, nlabels_train, M_train_l]...
    = read_labels(fid_train_label);

fclose(fid_train);
fclose(fid_train_label);

%% read test images + labels
fid_test = fopen([data_dir, fn_test]); % images
[magic_test, nims_test, nrows_test, ncols_test, M_test]...
    = read_images(fid_test);

fid_test_label = fopen([data_dir, fn_test_label]); % labels
[magic_test_l, nlabels_test, M_test_l]...
    = read_labels(fid_test_label);

fclose(fid_test);
fclose(fid_test_label);

%% 1. solve AX=B 
A = M_train(:,1:20000);
B = M_train_l(:,1:20000);
B_full = [B; zeros(size(A,1)-size(B,1), size(B,2))];

% 1. pinv()
%X = pinv(A)*B_full; 

% 2. qr
%[Q,R,P] = qr(A); X = (P*pinv(R)*Q') * B_full;

% 3. randomized svd

k = 50; Omega = randn(size(A,2),k);
Y = A*Omega;
[Q,R] = qr(Y, 0);
[U_, S_rec, V_rec] = svd(Q'*A, 'econ');

U_rec = Q*U_;
X = (V_rec*inv(S_rec)*U_rec')*B_full;

M = A*X;


figure

for i_fig = 1:25
    ii = randi(size(A,2));
    subplot(5,5,i_fig)
    %imagesc(reshape(M(:, i_fig), [nrows_train, ncols_train]))
    plot(M(1:10, ii), '-s')
    ylim([0,1])
    colorbar
    colormap('gray')
    title(['label = ', num2str(find(B(:,ii)==1))])
end

%% 1. solve AX=B with LASSO
figure('Position',[100 100 1500 1000])
nRowsFig = 5; nColsFig = 2;

pix_idx = [];
for i = 1:18
    pix_idx = [pix_idx, ((i+4)*28+6):((i+5)*28-5)];
end

% select A with 1 specific label
for ll = 1:10
    A = M_train(:, M_train_l(ll,:)==1);
    A = A(pix_idx, :);
    B_full = zeros(size(A,1), 1); B_full(ll) = 1;
    
    
    % 4. LASSO
    X = lasso(A, B_full, 'Lambda', 0.005);
        
    M = A*X;
    
    subplot(nRowsFig,nColsFig,ll)
    plot(M, '-s')
    %ylim([0,1])
    xlim([1,10])
    title(['label = ', num2str(ll)])
end


%% 2. pixels
clear X

%pix_idx = (8*28):784;

pix_idx = [];
for i = 1:18
    pix_idx = [pix_idx, ((i+4)*28+6):((i+5)*28-5)];
end


figure('Position',[100 100 1500 1000])
nRowsFig = 5; nColsFig = 2;

% select A with 1 single label
for ll = 1:10
    A = M_train(:, M_train_l(ll,:)==1);
    A = A(pix_idx, :);
    B_full = zeros(size(A,1), 1); B_full(ll) = 1;
    
    cvx_begin;
    variable X(size(A,2));
    minimize( norm(X,1) );
    subject to
    A*X == B_full;
    cvx_end;
    
    M = A*X;
    
    subplot(nRowsFig,nColsFig,ll)
    histogram(X)
    title(['label = ', num2str(ll),', optimal value = ', num2str(cvx_optval)])
end


%% 3. apply essential pixels to TEST data
figure('Position',[100 100 1000 1000])
nRowsFig = 3; nColsFig = 4;

%pix_idx = 340:440;

pix_idx = [];
for i = 1:18
    pix_idx = [pix_idx, ((i+4)*28+6):((i+5)*28-5)];
end


for ll = 1:10
    A = M_test(:, M_test_l(ll,:)==1);
    A = A(pix_idx, :);
    B_full = zeros(size(A,1), 1); B_full(ll) = 1;
    
    cvx_begin;
    variable X(size(A,2));
    minimize( norm(X,1) );
    subject to
    A*X == B_full;
    cvx_end;
    
    M = A*X;
    
    subplot(nRowsFig,nColsFig,ll)
    %imagesc(reshape(M(:, randi(size(A,2))), [nrows_train, ncols_train]))
    plot(M, '-s')
    xlim([1,10])
    title(['label = ', num2str(ll)])
end

%% 4. pixels for each digit

ll = 1;
%pix_idx = [(13*28+11):(14*28-12), (14*28+11):(15*28-12)];

A = M_train(:, M_train_l(ll,:)==1);
[~,pix_key] = sort(mean(abs(A),2));
pix_idx = find(pix_key>=772);

A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

figure
subplot(1,2,1)
histogram(X)
title(['label = ', num2str(ll),', optimal value = ', num2str(cvx_optval)])

A = M_test(:, M_test_l(ll,:)==1);
A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

M = A*X;

subplot(1,2,2)
plot(M, '-s')
xlim([1,10])
title(['label = ', num2str(ll)])

%%
%ll = 2;
%ll = 3;
%ll = 4;
%ll = 5;
%ll = 8;
ll = 10; % digit is 0

A = M_train(:, M_train_l(ll,:)==1);

%{
[~,pix_key] = sort(mean(abs(A),2));
pix_idx = find(pix_key>=762);
%}

pix_idx = [];
for i = 1:4
    pix_idx = [pix_idx, ((i+11)*28+13):((i+12)*28-13)];
end

A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

figure
subplot(1,2,1)
histogram(X)
title(['label = ', num2str(ll),', optimal value = ', num2str(cvx_optval)])

A = M_test(:, M_test_l(ll,:)==1);
A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

M = A*X;

subplot(1,2,2)
plot(M, '-s')
xlim([1,10])
title(['label = ', num2str(ll)])

%%
ll = 6;

A = M_train(:, M_train_l(ll,:)==1);
[~,pix_key] = sort(mean(abs(A),2));
pix_idx = find(pix_key>=772);

A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

figure
subplot(1,2,1)
histogram(X)
title(['label = ', num2str(ll),', optimal value = ', num2str(cvx_optval)])

A = M_test(:, M_test_l(ll,:)==1);
A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

M = A*X;

subplot(1,2,2)
plot(M, '-s')
xlim([1,10])
title(['label = ', num2str(ll)])

%%
%ll = 7; 
ll = 9;

A = M_train(:, M_train_l(ll,:)==1);
[~,pix_key] = sort(mean(abs(A),2));
pix_idx = find(pix_key>=772);

A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

figure
subplot(1,2,1)
histogram(X)
title(['label = ', num2str(ll),', optimal value = ', num2str(cvx_optval)])

A = M_test(:, M_test_l(ll,:)==1);
A = A(pix_idx, :);
B_full = zeros(size(A,1), 1); B_full(ll) = 1;

cvx_begin;
variable X(size(A,2));
minimize( norm(X,1) );
subject to
A*X == B_full;
cvx_end;

M = A*X;

subplot(1,2,2)
plot(M, '-s')
xlim([1,10])
title(['label = ', num2str(ll)])
