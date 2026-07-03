function [lrIdx, apIdx, dvIdx] = quickniiVoxelIndices(lrC, apC, dvC)
%QUICKNIIVOXELINDICES VisuAlign/QUINT-matching voxel selection (0-based).
%   [LRIDX, APIDX, DVIDX] = QUICKNIIVOXELINDICES(LRC, APC, DVC) takes
%   continuous atlas coordinates (in BrainGlobe's raw, un-reordered
%   (AP, DV, LR) array convention) and returns the 0-based voxel indices
%   VisuAlign/PyNutil would sample: floor(LR), ceil(AP), ceil(DV).
%
%   AP and DV use ceil (not floor) because BrainGlobe's raw array has AP/DV
%   reversed relative to QuickNII's convention; this asymmetric floor/ceil
%   split reproduces VisuAlign/PyNutil's sampling exactly (~0.5-voxel
%   boundary offset otherwise). See .claude/matlab-port.md and
%   engine/atlas.py::_quicknii_floor_indices for the full derivation.
%
%   Outputs are still 0-based, continuous-valued doubles (not yet clipped
%   or cast to integer array subscripts) — callers bounds-check, clip, and
%   add 1 before indexing into a MATLAB array.
    lrIdx = floor(lrC);
    apIdx = ceil(apC);
    dvIdx = ceil(dvC);
end
