function out = warpPointsAtlasToSection(pointsNorm, srcPx, dstPx, workW, workH)
%WARPPOINTSATLASTOSECTION Map atlas-space points into section-space via the Delaunay warp.
%   OUT = WARPPOINTSATLASTOSECTION(POINTSNORM, SRCPX, DSTPX, WORKW, WORKH)
%   is the mirror of WARPPOINTSSECTIONTOATLAS: triangulates on the *src*
%   (atlas) anchors and interpolates *dst* (section) coordinates. Points
%   outside the convex hull pass through unchanged (clipped to [0, 1]).
%
%   Mirrors engine/warping.py::warp_points_atlas_to_section exactly -- see
%   .claude/matlab-port.md.
%
%   POINTSNORM: (M, 2) normalised atlas-space points in [0, 1].
%   SRCPX, DSTPX: (N, 2) atlas-space / section-space control points, in
%       working-resolution pixels (may be empty, i.e. 0x2, for no warp).
%   WORKW, WORKH: working image dimensions in pixels.
%
%   OUT: (M, 2) normalised section-space points.
    [pts, srcNorm, dstNorm, scale] = prepareWarp(pointsNorm, srcPx, dstPx, workW, workH);
    if isempty(srcNorm) || npAllClose(srcNorm, dstNorm)
        out = min(max(pts, 0.0), 1.0);
        return
    end
    % Triangulate the atlas (src) anchors; interpolate the section (dst) coords.
    out = barycentricMap(withCorners(srcNorm), withCorners(dstNorm), pts, scale);
end
