function [num_mag, n_ims, n_rows, n_cols, I] = read_images(fid)
num_mag = fread(fid, 1, 'int', 'ieee-be'); % magic#
n_ims = fread(fid, 1, 'int', 'ieee-be'); % #images
n_rows = fread(fid, 1, 'int', 'ieee-be'); % #rows
n_cols = fread(fid, 1, 'int', 'ieee-be'); % #columns
I = zeros(n_rows*n_cols, n_ims);

for i = 1:n_ims
    I_i = fread(fid, [n_rows, n_cols], 'int8', 'ieee-be')';
    I(:, i) = reshape(I_i, [n_rows*n_cols, 1]);
end

return