%% Differential Evolution (DE) and its Variants on Rastrigin Function
% Tác giả: Nhóm 36
% Tài liệu tham khảo: Wikipedia, bài báo nghiên cứu, AI (ChatGPT, Gemini)
% Mục tiêu: So sánh DE gốc và các biến thể (DE/best/1, JADE, CoDE)
% Ngày: 25/11/2025

clear; clc; close all;

%% --- Thông số chung ---
n = 10;              % số chiều
A = 10;              % hằng số hàm Rastrigin
rastrigin = @(x) A*n + sum(x.^2 - A*cos(2*pi*x));
lb = -5.12; ub = 5.12;
pop_size = 60;
F = 0.7; CR = 0.9;
max_iter = 3000;
rng(1); % cố định seed để tái lập kết quả

%% --- Chạy từng thuật toán ---
fprintf("Running DE/rand/1 ...\n");
[hist_rand, score_rand] = DE_rand1(rastrigin, n, lb, ub, pop_size, F, CR, max_iter);

fprintf("Running DE/best/1 ...\n");
[hist_best, score_best] = DE_best1(rastrigin, n, lb, ub, pop_size, F, CR, max_iter);

fprintf("Running JADE ...\n");
[hist_jade, score_jade] = JADE_variant(rastrigin, n, lb, ub, pop_size, max_iter);

fprintf("Running CoDE ...\n");
[hist_code, score_code] = CoDE_variant(rastrigin, n, lb, ub, pop_size, max_iter);

%% --- Vẽ kết quả hội tụ ---
figure;
plot(hist_rand, 'b', 'LineWidth', 2); hold on;
plot(hist_best, 'r', 'LineWidth', 2);
plot(hist_jade, '--g', 'LineWidth', 2);  % nét đứt xanh lá
plot(hist_code, '--k', 'LineWidth', 2);  % nét đứt đen
xlabel('Iteration'); ylabel('Best Fitness (log scale)');
legend('DE/rand/1','DE/best/1','JADE','CoDE');
title('So sánh hội tụ giữa DE gốc và các biến thể');
grid on;

%%
figure;
plot(hist_rand, 'b', 'LineWidth', 2); hold on;
plot(hist_best, 'r', 'LineWidth', 2);
plot(hist_jade, '--g', 'LineWidth', 2);
plot(hist_code, '--k', 'LineWidth', 2);

xlim([2500 3000]);     % zoom trục X
ylim([0 5]);           % để thấy rõ các đường
xlabel('Iteration'); ylabel('Best Fitness (log scale)');
title('Phóng to vùng 2500–3000');
grid on;

%% --- Kết quả cuối ---
fprintf("\n===== KẾT QUẢ TỐT NHẤT =====\n");
fprintf("DE/rand/1 : %.6e\n", score_rand);
fprintf("DE/best/1 : %.6e\n", score_best);
fprintf("JADE      : %.6e\n", score_jade);
fprintf("CoDE      : %.6e\n", score_code);

%% ===================================================================
%% HÀM CON - DE GỐC (DE/rand/1/bin)
function [best_hist, best_score] = DE_rand1(f, n, lb, ub, NP, F, CR, max_iter)
pop = lb + (ub-lb)*rand(NP, n);
score = arrayfun(@(i) f(pop(i,:)), 1:NP)';
best_hist = zeros(max_iter,1);

for iter=1:max_iter
    for i=1:NP
        % --- Mutation ---
        idx = randperm(NP,3);
        while any(idx==i), idx = randperm(NP,3); end
        a=pop(idx(1),:); b=pop(idx(2),:); c=pop(idx(3),:);
        v = a + F*(b - c);
        v = max(min(v,ub),lb); % giới hạn trong biên
        
        % --- Crossover ---
        u = pop(i,:);
        jrand = randi(n);
        for j=1:n
            if rand<CR || j==jrand, u(j)=v(j); end
        end
        
        % --- Selection ---
        fu = f(u);
        if fu < score(i)
            pop(i,:) = u; score(i) = fu;
        end
    end
    best_hist(iter) = min(score);
end
best_score = min(score);
end

%% ===================================================================
%% HÀM CON - DE/best/1/bin
function [best_hist, best_score] = DE_best1(f, n, lb, ub, NP, F, CR, max_iter)
pop = lb + (ub-lb)*rand(NP, n);
score = arrayfun(@(i) f(pop(i,:)), 1:NP)';
best_hist = zeros(max_iter,1);

for iter=1:max_iter
    [~, best_idx] = min(score);
    xbest = pop(best_idx,:);
    for i=1:NP
        idx = randperm(NP,2);
        while any(idx==i), idx = randperm(NP,2); end
        b=pop(idx(1),:); c=pop(idx(2),:);
        v = xbest + F*(b - c);
        v = max(min(v,ub),lb);
        
        u = pop(i,:);
        jrand = randi(n);
        for j=1:n
            if rand<CR || j==jrand, u(j)=v(j); end
        end
        
        fu = f(u);
        if fu < score(i)
            pop(i,:) = u; score(i)=fu;
        end
    end
    best_hist(iter)=min(score);
end
best_score = min(score);
end

%% ===================================================================
%% HÀM CON - JADE (phiên bản rút gọn, dễ hiểu)
function [best_hist, best_score] = JADE_variant(f, n, lb, ub, NP, max_iter)
p = 0.1;     % top 10% để chọn p-best
c = 0.1;     % learning rate
mu_F = 0.5;  % trung bình ban đầu của F
mu_CR = 0.5; % trung bình ban đầu của CR
pop = lb + (ub-lb)*rand(NP, n);
score = arrayfun(@(i) f(pop(i,:)), 1:NP)';
archive = [];
best_hist = zeros(max_iter,1);

for iter=1:max_iter
    [~, sorted_idx] = sort(score);
    pnum = max(2, round(p*NP));
    pbest_set = pop(sorted_idx(1:pnum),:);
    SF=[]; SCR=[];
    
    for i=1:NP
        Fi = min(max(cauchyrnd(mu_F,0.1),0),1);
        CRi = min(max(normrnd(mu_CR,0.1),0),1);
        
        xpbest = pbest_set(randi(pnum),:);
        all_pool = [pop; archive];
        idxs = randperm(size(all_pool,1),2);
        r1 = all_pool(idxs(1),:);
        r2 = all_pool(idxs(2),:);
        
        v = pop(i,:) + Fi*(xpbest - pop(i,:)) + Fi*(r1 - r2);
        v = max(min(v,ub),lb);
        
        u = pop(i,:);
        jrand = randi(n);
        for j=1:n
            if rand<CRi || j==jrand, u(j)=v(j); end
        end
        
        fu = f(u);
        if fu < score(i)
            archive = [archive; pop(i,:)];
            pop(i,:) = u; score(i)=fu;
            SF = [SF; Fi]; SCR = [SCR; CRi];
        end
    end
    
    if ~isempty(SF)
        mu_F = (1-c)*mu_F + c*(mean(SF.^2)/mean(SF));
        mu_CR = (1-c)*mu_CR + c*mean(SCR);
    end
    if size(archive,1)>NP, archive=archive(randperm(size(archive,1),NP),:); end
    best_hist(iter)=min(score);
end
best_score = min(score);
end

function r = cauchyrnd(loc, scale)
r = loc + scale * tan(pi*(rand - 0.5));
end

%% ===================================================================
%% HÀM CON - CoDE (3 chiến lược kết hợp)
function [best_hist, best_score] = CoDE_variant(f, n, lb, ub, NP, max_iter)
F_pool = [1.0, 1.0, 0.8];
CR_pool = [0.1, 0.9, 0.2];
pop = lb + (ub-lb)*rand(NP, n);
score = arrayfun(@(i) f(pop(i,:)), 1:NP)';
best_hist = zeros(max_iter,1);

for iter=1:max_iter
    for i=1:NP
        best_local = inf; best_vec = [];
        for k=1:3
            F = F_pool(k); CR = CR_pool(k);
            idx = randperm(NP,5);
            while any(idx==i), idx = randperm(NP,5); end
            a=pop(idx(1),:); b=pop(idx(2),:); c=pop(idx(3),:);
            d=pop(idx(4),:); e=pop(idx(5),:);
            
            switch k
                case 1, v = a + F*(b - c); % rand/1
                case 2, v = pop(i,:) + F*(a - pop(i,:)) + F*(b - c); % current-to-rand/1
                case 3, v = a + F*(b - c + d - e); % rand/2
            end
            v = max(min(v,ub),lb);
            
            u = pop(i,:);
            jrand = randi(n);
            for j=1:n
                if rand<CR || j==jrand, u(j)=v(j); end
            end
            
            fu = f(u);
            if fu < best_local
                best_local = fu; best_vec = u;
            end
        end
        if best_local < score(i)
            pop(i,:) = best_vec; score(i) = best_local;
        end
    end
    best_hist(iter) = min(score);
end
best_score = min(score);
end
