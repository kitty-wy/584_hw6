function [num_mag, n_las, I] = read_labels(fid)
num_mag = fread(fid, 1, 'int', 'ieee-be'); % magic#
n_las = fread(fid, 1, 'int', 'ieee-be'); % #labels
l = fread(fid, n_las, 'uint8', 'ieee-be'); % labels
I = zeros(10, n_las);

for i = 1:n_las
    if l(i) == 0
        I(10, i) = 1;
    else % label == 1,2,...,9
        I(l(i), i) = 1;
    end
end

return