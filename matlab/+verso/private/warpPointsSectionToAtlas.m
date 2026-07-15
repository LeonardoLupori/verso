function out = warpPointsSectionToAtlas(pointsNorm, srcPx, dstPx, workW, workH)
%WARPPOINTSSECTIONTOATLAS Map section-space points into atlas-space via the Delaunay warp.
%   OUT = WARPPOINTSSECTIONTOATLAS(POINTSNORM, SRCPX, DSTPX, WORKW, WORKH)
%   triangulates on the *dst* (section) anchors and interpolates *src*
%   (atlas) coordinates for each input point. Points outside the convex
%   hull pass through unchanged (clipped to [0, 1]).
%
%   Mirrors engine/warping.py::warp_points_section_to_atlas exactly
%   (including the VisuAlign-parity aspect-ratio pre-scaling and the four
%   invisible corner anchors) -- see .claude/matlab-port.md.
%
%   POINTSNORM: (M, 2) normalised section-space points in [0, 1].
%   SRCPX, DSTPX: (N, 2) atlas-space / section-space control points, in
%       working-resolution pixels (may be empty, i.e. 0x2, for no warp).
%   WORKW, WORKH: working image dimensions in pixels (normalises control
%       points; does not affect POINTSNORM, already normalised).
%
%   OUT: (M, 2) normalised atlas-space points.
    [pts, srcNorm, dstNorm, scale] = prepareWarp(pointsNorm, srcPx, dstPx, workW, workH);
    if isempty(dstNorm) || npAllClose(srcNorm, dstNorm)
        out = min(max(pts, 0.0), 1.0);
        return
    end
    % Triangulate the section (dst) anchors; interpolate the atlas (src) coords.
    out = barycentricMap(withCorners(dstNorm), withCorners(srcNorm), pts, scale);
end
