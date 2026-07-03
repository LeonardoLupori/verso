function [o, u, v] = anchoringToVectors(anchoring)
%ANCHORINGTOVECTORS Split a 9-element anchoring vector into origin + direction vectors.
%   [O, U, V] = ANCHORINGTOVECTORS(ANCHORING) splits the 9-element
%   [ox oy oz ux uy uz vx vy vz] anchoring vector (atlas voxel space,
%   matching project-verso.json / QuickNII) into three 1x3 row vectors.
%
%   Mirrors engine/anchoring.py::anchoring_to_vectors.
    anchoring = double(anchoring(:)');
    if numel(anchoring) ~= 9
        error("verso:anchoringToVectors:badSize", ...
            "anchoring must have 9 elements, got %d", numel(anchoring));
    end
    o = anchoring(1:3);
    u = anchoring(4:6);
    v = anchoring(7:9);
end
