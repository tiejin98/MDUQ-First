
npy_library_path = '/npy-matlab';  % Replace this with the actual path
addpath(genpath(npy_library_path));

cpd_library_path = '/tensor_toolbox-v3.6';
addpath(genpath(cpd_library_path));
% parpool('local');
function emd_value = compute_emd(P, Q, dist_matrix)
    % P and Q are the histograms (should sum to 1)
    % dist_matrix is the distance between the bins of P and Q (should be non-negative)

    % Ensure that P and Q are row vectors
    P = P(:)'; 
    Q = Q(:)'; 

    % The number of bins (must match between P and Q)
    num_bins = length(P);

    % Flatten the distance matrix to use in the linear programming
    f = dist_matrix(:);

    % Set up the equality constraints
    Aeq = [kron(eye(num_bins), ones(1, num_bins)); kron(ones(1, num_bins), eye(num_bins))];
    beq = [P'; Q'];

    % Set up the boundss
    lb = zeros(num_bins^2, 1);
    ub = [];

    % Solve the linear program
    options = optimoptions('linprog', 'Display', 'none');
    [flow, ~] = linprog(f, [], [], Aeq, beq, lb, ub, options);
    
    % Reshape the flow to match the distance matrix
    flow = reshape(flow, [num_bins, num_bins]);

    % The resulting EMD value is the cost of the optimal flow
    emd_value = sum(sum(flow .* dist_matrix));
end

% Check if writeNPY function exists, if not, install NpyMatlab

pickle = py.importlib.import_module('pickle');
io = py.importlib.import_module('io');
emd_module = py.importlib.import_module('pyemd');


seman_file_path = 'semantic_simmat.pkl';
know_file_path = 'knowledge_simmat.pkl';

% Extra pickle files (new matrices)
seman_file_path_extra = 'semantic_simmat_disagree.pkl';
know_file_path_extra = 'knowledge_simmat_disagree.pkl';
seman_file = io.open(seman_file_path, 'rb');
know_file = io.open(know_file_path, 'rb');
seman = pickle.load(seman_file);
know = pickle.load(know_file);
seman_file.close();
know_file.close();

seman_file_extra = io.open(seman_file_path_extra, 'rb');
know_file_extra = io.open(know_file_path_extra, 'rb');
seman_extra = pickle.load(seman_file_extra);
know_extra = pickle.load(know_file_extra);
seman_file_extra.close();
know_file_extra.close();


seman_keys = cellfun(@char, cell(py.list(seman.keys())), 'UniformOutput', false);
know_keys   = cellfun(@char, cell(py.list(know.keys())), 'UniformOutput', false);
seman_extra_keys = cellfun(@char, cell(py.list(seman_extra.keys())), 'UniformOutput', false);
know_extra_keys  = cellfun(@char, cell(py.list(know_extra.keys())), 'UniformOutput', false);


% Initialize an empty structure to store the final results
final = containers.Map();

% Iterate through the keys in the seman dict
for i = 1:length(know_keys)
    key = know_keys{i};
    
    % Check if the key exists in the know dictionary
    if ~any(strcmp(seman_keys, key))
        continue;  % Skip to the next key if it doesn't exist in know
    end
    
    % Extract the matrices from the seman and know dicts
    seman_matrix = double(py.numpy.array(seman{key}));
    know_matrix = double(py.numpy.array(know{key}));
    % Stack the matrices along the third dimension (equivalent to axis=0 in Python)
    final_matrix = cat(3, seman_matrix, know_matrix);
    
    % Store the final matrix in the final struct with the same key
    final(key) = final_matrix;
end

keys = final.keys;  % Get all the keys (as a cell array)
num_keys = length(keys);  % Get the number of keys

keys_list = {};
values_list = {};
res = containers.Map;
total = 0;

% Generate random matrix once, assuming all final_matrix have the same shape
shape = size(final(keys{1}));  % Access using containers.Map syntax
random_matrix = randn(shape); % Random matrix is generated once outside the loop
random_matrix = tensor(random_matrix);
h = waitbar(0, 'Processing...');
tic;
total_iterations = min(1000, num_keys) * length(1:20);  % Total number of iterations for both loops
current_iteration = 0;
normR = norm(random_matrix);
for k = 1:min(1000, num_keys)
    key = keys{k};
    final_matrix = final(key);  % Access containers.Map using parentheses
    final_matrix = tensor(final_matrix);
    recon_key = [];
    recon_random_key = [];
    normX = norm(final_matrix);
    for i = 1:20
        flag = false;
        rank = [i,i,2];
        final_factors = tucker_als(tensor(final_matrix), rank,'printitn', 0);
        random_factors = tucker_als(tensor(random_matrix), rank,'printitn', 0);
        
        normresidual = sqrt( normX^2 + norm(final_factors)^2 - 2 * innerprod(final_factors,final_factors));             
        fit = 1 - (normresidual / normX); %fraction explained by model
        recon_key = [recon_key, fit];
        normresidual = sqrt( normR^2 + norm(random_factors)^2 - 2 * innerprod(random_matrix,random_factors));
        fit = 1 - (normresidual / normR); %fraction explained by model
        recon_random_key = [recon_random_key, fit];

        flag = true;

        current_iteration = current_iteration + 1;
        elapsed_time = toc;
        remaining_time = (elapsed_time / current_iteration) * (total_iterations - current_iteration);
        
        % Format the remaining time in HH:MM:SS format
        estimated_time_str = datestr(seconds(remaining_time), 'HH:MM:SS');
        
        % Update the progress bar with estimated remaining time
        waitbar(current_iteration / total_iterations, h, ...
            sprintf('Processing key %d of %d, time remaining: %s', ...
            k, min(1000, num_keys), estimated_time_str));
    end
    
    % Store the reconstruction results
    recon_keys{k}        = key;              
    recon_values{k}      = recon_key;   
    
    total = total + 1;
    waitbar(k / min(1000, num_keys), h);
end

recon_res  = [recon_keys; recon_values];

save('results_tucker.mat', 'recon_res', '-v7');

