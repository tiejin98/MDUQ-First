npy_library_path = '/npy-matlab';  % Replace this with the actual path
addpath(genpath(npy_library_path));

cpd_library_path = '/tensor_toolbox-v3.6';
addpath(genpath(cpd_library_path));


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

    % Set up the bounds
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

final = containers.Map();

% Iterate through keys available in the original know dictionary
for i = 1:length(know_keys)
    key = know_keys{i};
    
    % Check that the key exists in all four dictionaries
    if ~any(strcmp(seman_keys, key)) || ~any(strcmp(seman_extra_keys, key)) || ~any(strcmp(know_extra_keys, key))
        continue;  % Skip if key is missing in any of the extra dictionaries
    end
    
    % Extract matrices from the original dictionaries
    seman_matrix = double(py.numpy.array(seman{key}));
    know_matrix  = double(py.numpy.array(know{key}));
    
    % Extract matrices from the extra dictionaries
    seman_matrix_extra = double(py.numpy.array(seman_extra{key}));
    know_matrix_extra  = double(py.numpy.array(know_extra{key}));

    % Form a tensor with 4 slices along the third dimension (20x20x4)
    final_matrix = cat(3, seman_matrix, know_matrix,seman_matrix_extra,know_matrix_extra);
    
    % Store the 20x20x4 tensor in the final map under the current key
    final(key) = final_matrix;
end

keys = final.keys;          % Get all keys from the final map (cell array)
num_keys = length(keys);    % Number of keys

keys_list = {};
values_list = {};
total = 0;

% Generate a random tensor once (using the shape from one final matrix)
shape = size(final(keys{1}));  % Should now be [20, 20, 4]
random_matrix = randn(shape);
random_matrix = tensor(random_matrix);

h = waitbar(0, 'Processing...');
tic;
total_iterations = min(1000, num_keys) * 10;
current_iteration = 0;
normR = norm(random_matrix);
N = min(1000, num_keys);
recon_keys         = cell(1, N);     
recon_values       = cell(1, N);     

for k = 1:min(1000, num_keys)
    key = keys{k};
    final_matrix = final(key);  % Get the 20x20x4 tensor for this key
    final_matrix = tensor(final_matrix);
    
    recon_key = [];
    recon_random_key = [];
    normX = norm(final_matrix);
    
    for i = 10:20
        % Perform CP decomposition on the tensor with i components
        final_factors = cp_als(final_matrix, i, 'printitn', 0);
        % Also perform CP decomposition on the random tensor
        random_factors = cp_als(random_matrix, i, 'printitn', 0);

        % Compute the reconstruction fit for final_matrix
        normresidual = sqrt( normX^2 + norm(final_factors)^2 - 2 * innerprod(final_factors, final_factors) );
        fit = 1 - (normresidual / normX); % Fraction explained by the model
        recon_key = [recon_key, fit];
        
        % Compute the reconstruction fit for the random tensor
        normresidual_rand = sqrt( normR^2 + norm(random_factors)^2 - 2 * innerprod(random_matrix, random_factors) );
        fit_rand = 1 - (normresidual_rand / normR);
        recon_random_key = [recon_random_key, fit_rand];

        current_iteration = current_iteration + 1;
        elapsed_time = toc;
        remaining_time = (elapsed_time / current_iteration) * (total_iterations - current_iteration);
        
        % Format the remaining time in HH:MM:SS format
        estimated_time_str = datestr(seconds(remaining_time), 'HH:MM:SS');
        
        % Update the progress bar with the current status and estimated remaining time
        waitbar(current_iteration / total_iterations, h, ...
            sprintf('Processing key %d of %d, time remaining: %s', k, min(1000, num_keys), estimated_time_str));
    end
    
    % Store the reconstruction results for this key
    recon_keys{k}        = key;              
    recon_values{k}      = recon_key;   

    total = total + 1;
    waitbar(k / min(1000, num_keys), h);
    
    % Optional: save intermediate results after processing the first key

end
recon_res  = [recon_keys; recon_values];

close(h);

% Save the final reconstruction results (you can update the filename as desired)
save('results_total_cp.mat', 'recon_res','-v7');
