function [pks, locs] = SBLpeaks_1D(gamma, Nsources)
%
% [pks, locs] = SBLpeaks_1D(gamma, Nsources)
%
% fast alternative for findpeaks in 1D case
%

% output variables
pks = zeros(Nsources,1);
locs = zeros(Nsources,1);

% zero padding on the boundary
gamma_new = zeros(length(gamma)+2,1);
gamma_new(2:end-1) = gamma;

[~, Ilocs]= sort(gamma,'descend');

% current number of peaks found
npeaks = 0;

for ii = 1:length(Ilocs)
    
    % local patch area surrounding the current array entry i.e. (r,c)
    local_patch = gamma_new(Ilocs(ii):Ilocs(ii)+2);
    
    % zero the center
    local_patch(2) = 0;
    
    if sum(sum(gamma(Ilocs(ii)) > local_patch)) == 3
        npeaks = npeaks + 1;
        
        pks(npeaks) = gamma(Ilocs(ii));
        locs(npeaks) = Ilocs(ii);
        
        % if found sufficient peaks, break
        if npeaks == Nsources
            break;
        end
    end
    
end

% if Nsources not found
if npeaks ~= Nsources
    pks(npeaks+1:Nsources) = [];
    locs(npeaks+1:Nsources) = [];
end

end
