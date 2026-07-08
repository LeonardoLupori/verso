function [gray, inBounds] = sampleReferenceAt(atlas, grid)
%SAMPLEREFERENCEAT Sample the reference/template volume at arbitrary atlas voxel coordinates.
%   [GRAY, INBOUNDS] = SAMPLEREFERENCEAT(ATLAS, GRID) samples
%   ATLAS.Reference (as returned by loadAtlasVolume) at the atlas voxel
%   coordinates in GRID, an (H, W, 3) array with GRID(:,:,1)=LR,
%   GRID(:,:,2)=AP, GRID(:,:,3)=DV (anchoring voxel order).
%
%   GRAY: (H, W) uint8, scaled by ATLAS.RefScale; 0 outside the atlas volume.
%   INBOUNDS: (H, W) logical -- true where the voxel is inside the atlas volume.
%
%   Mirrors engine/atlas.py::AtlasVolume.sample_reference_at.
    [lrIdx, apIdx, dvIdx] = sampleVoxelIndices(grid(:, :, 1), grid(:, :, 2), grid(:, :, 3));

    sz = size(atlas.Reference);
    apMax = sz(1); dvMax = sz(2); lrMax = sz(3);
    inBounds = (apIdx >= 0) & (apIdx < apMax) ...
        & (dvIdx >= 0) & (dvIdx < dvMax) ...
        & (lrIdx >= 0) & (lrIdx < lrMax);

    apC = min(max(apIdx, 0), apMax - 1);
    dvC = min(max(dvIdx, 0), dvMax - 1);
    lrC = min(max(lrIdx, 0), lrMax - 1);

    linIdx = sub2ind(sz, apC + 1, dvC + 1, lrC + 1);
    grayVals = min(max(atlas.Reference(linIdx) * atlas.RefScale, 0), 255);
    gray = uint8(grayVals);
    gray(~inBounds) = 0;
end
