function edges = boundaryMask(labels, inBounds)
%BOUNDARYMASK Bool edge mask between differing labels, brain-adjacent only.
%   EDGES = BOUNDARYMASK(LABELS, INBOUNDS) marks a 1-pixel edge where two
%   neighbouring pixels carry different labels and at least one of them is
%   annotated brain (label > 0 and in-bounds), so the empty-background
%   frame is not outlined.
%
%   Mirrors engine/atlas.py::boundary_mask.
%
%   LABELS, INBOUNDS: (H, W) arrays, same size. INBOUNDS logical.
%   EDGES: (H, W) logical.
    brain = (labels > 0) & inBounds;
    edges = false(size(labels));

    % Horizontal edges (between col i and i+1) -- mark left pixel only (1px)
    diffH = labels(:, 1:end-1) ~= labels(:, 2:end);
    keepH = diffH & (brain(:, 1:end-1) | brain(:, 2:end));
    edges(:, 1:end-1) = edges(:, 1:end-1) | keepH;

    % Vertical edges (between row i and i+1) -- mark top pixel only (1px)
    diffV = labels(1:end-1, :) ~= labels(2:end, :);
    keepV = diffV & (brain(1:end-1, :) | brain(2:end, :));
    edges(1:end-1, :) = edges(1:end-1, :) | keepV;
end
