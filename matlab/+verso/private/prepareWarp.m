function [pts, srcNorm, dstNorm, scale] = prepareWarp(pointsNorm, srcPx, dstPx, workW, workH)
%PREPAREWARP Normalise query + control points and compute the triangulation scale.
%   [PTS, SRCNORM, DSTNORM, SCALE] = PREPAREWARP(POINTSNORM, SRCPX, DSTPX, WORKW,
%   WORKH) is the shared front-end for the two warp directions: it reshapes the
%   query points, normalises the control points to [0, 1]^2 by the working image
%   size, and returns the [aspect, 1] anisotropy factor applied before
%   triangulating (see .claude/matlab-port.md and warping.py::_tri_scale).
%
%   Mirrors engine/warping.py::_prepare_warp exactly.
%
%   POINTSNORM: (M, 2) normalised query points in [0, 1].
%   SRCPX, DSTPX: (N, 2) atlas-space / section-space control points, in
%       working-resolution pixels (may be empty, i.e. 0x2, for no warp).
%   WORKW, WORKH: working image dimensions in pixels.
    pts = reshape(double(pointsNorm), [], 2);
    wh = [double(workW), double(workH)];
    srcNorm = reshape(double(srcPx), [], 2) ./ wh;
    dstNorm = reshape(double(dstPx), [], 2) ./ wh;
    scale = [workW / workH, 1.0];
end
