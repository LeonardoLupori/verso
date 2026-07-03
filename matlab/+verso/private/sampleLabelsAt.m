function [labels, inBounds] = sampleLabelsAt(atlas, grid)
%SAMPLELABELSAT Sample the annotation volume at arbitrary atlas voxel coordinates.
%   [LABELS, INBOUNDS] = SAMPLELABELSAT(ATLAS, GRID) samples ATLAS.Annotation
%   (as returned by loadAtlasVolume) at the atlas voxel coordinates in GRID,
%   an (H, W, 3) array with GRID(:,:,1)=LR, GRID(:,:,2)=AP, GRID(:,:,3)=DV
%   (QuickNII voxel order, matching the anchoring formula).
%
%   LABELS: (H, W) int32, valid everywhere (clamped) -- caller decides what
%       to do with out-of-bounds pixels using INBOUNDS.
%   INBOUNDS: (H, W) logical -- true where the voxel is inside the atlas volume.
%
%   Mirrors engine/atlas.py::AtlasVolume.sample_labels_at.
    [lrIdx, apIdx, dvIdx] = quickniiVoxelIndices(grid(:, :, 1), grid(:, :, 2), grid(:, :, 3));

    sz = size(atlas.Annotation);
    apMax = sz(1); dvMax = sz(2); lrMax = sz(3);
    inBounds = (apIdx >= 0) & (apIdx < apMax) ...
        & (dvIdx >= 0) & (dvIdx < dvMax) ...
        & (lrIdx >= 0) & (lrIdx < lrMax);

    apC = min(max(apIdx, 0), apMax - 1);
    dvC = min(max(dvIdx, 0), dvMax - 1);
    lrC = min(max(lrIdx, 0), lrMax - 1);

    linIdx = sub2ind(sz, apC + 1, dvC + 1, lrC + 1);
    labels = atlas.Annotation(linIdx);
end
