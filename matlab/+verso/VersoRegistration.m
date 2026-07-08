classdef VersoRegistration < handle
    %VERSOREGISTRATION Convert pixels to Allen CCFv3 atlas coordinates for a VERSO project.
    %   MATLAB port of engine/registration.py::VersoRegistration. Reads a
    %   native project-verso.json (self-contained for coordinate work) and
    %   provides the same three operations:
    %
    %       r = verso.VersoRegistration("my_experiment/project-verso.json");
    %       xyz = r.coord_image_to_atlas("s001", [1200 3400; 1500 3600]);
    %       res = r.coord_atlas_to_image(xyz);
    %       labels = r.image_to_atlas("s001", "Kind", "annotation");
    %
    %   A slice is addressed by its section id, or by the original image's
    %   file stem or basename (same resolution rules as the Python side).
    %
    %   Atlas volumes (annotation/reference) are read from a local BrainGlobe
    %   atlas cache folder, resolved (and downloaded if necessary) the first
    %   time image_to_atlas is called -- see the AtlasDir constructor
    %   argument and .claude/matlab-port.md.
    %
    %   Requires MATLAB with the Image Processing Toolbox (tiffreadVolume).

    properties (Access = private)
        ResolutionUm
        AtlasShapeVal
        AtlasNameVal
        AtlasDirOverride
        Snapshots       % 1xN struct array of snapshots, project order (parallel to IdsList)
        IdsList         % cellstr, project order
        AtlasVol        % [] until first needed by image_to_atlas
        ChunkPixelBudget = 2000000  % image_to_atlas row-chunk size (overridable in tests)
    end

    methods
        function obj = VersoRegistration(projectPath, opts)
            %VERSOREGISTRATION Load a project from its native JSON.
            %   R = VERSO.VERSOREGISTRATION(PROJECTPATH) loads
            %   project-verso.json and builds the coordinate snapshot.
            %
            %   R = VERSO.VERSOREGISTRATION(PROJECTPATH, "AtlasDir", DIR)
            %   uses DIR (a folder containing annotation.tiff/reference.tiff/
            %   structures.json/metadata.json) instead of the default
            %   BrainGlobe cache lookup/download.
            arguments
                projectPath (1, 1) string
                opts.AtlasDir (1, 1) string = ""
            end
            proj = loadProjectJson(projectPath);
            obj.ResolutionUm = proj.ResolutionUm;
            obj.AtlasShapeVal = proj.AtlasShape;
            obj.AtlasNameVal = proj.AtlasName;
            obj.Snapshots = proj.Snapshots;
            obj.IdsList = proj.Ids;
            obj.AtlasDirOverride = opts.AtlasDir;
            obj.AtlasVol = [];
        end

        function ids = ids(obj)
            %IDS Return the section ids in project order.
            ids = obj.IdsList;
        end

        function n = numSections(obj)
            %NUMSECTIONS Return the number of sections in the project.
            n = numel(obj.IdsList);
        end

        function tf = hasSection(obj, key)
            %HASSECTION True if KEY resolves to a section (id, stem, or basename).
            try
                obj.resolveSlice(key);
                tf = true;
            catch
                tf = false;
            end
        end

        function [coords, inside] = coord_image_to_atlas(obj, sliceKey, xy, opts)
            %COORD_IMAGE_TO_ATLAS Map image pixels on SLICEKEY to atlas coordinates.
            %   COORDS = R.COORD_IMAGE_TO_ATLAS(SLICEKEY, XY) maps an (N,2)
            %   array of image pixels to an (N,3) array of atlas coordinates.
            %
            %   Name-value options:
            %       Space ("full" | "working", default "full")
            %       Units ("voxel" | "um" | "mm", default "voxel")
            %
            %   [COORDS, INSIDE] = ... also returns an (N,1) logical mask,
            %   true where the pixel falls within the section frame.
            %
            %   Mirrors engine/registration.py::coord_image_to_atlas.
            arguments
                obj
                sliceKey (1, 1) string
                xy (:, 2) double
                opts.Space (1, 1) string {mustBeMember(opts.Space, ["full", "working"])} = "full"
                opts.Units (1, 1) string {mustBeMember(opts.Units, ["voxel", "um", "mm"])} = "voxel"
            end
            snap = obj.getSnapshot(sliceKey);
            if ~snap.Aligned
                error("verso:VersoRegistration:notAligned", ...
                    "Section '%s' has no alignment; cannot map pixels.", snap.Id);
            end

            if opts.Space == "full"
                px = xy(:, 1) * snap.WorkW / snap.FullW;
                py = xy(:, 2) * snap.WorkH / snap.FullH;
            else
                px = xy(:, 1);
                py = xy(:, 2);
            end

            if snap.FlipH
                px = snap.WorkW - px;
            end
            if snap.FlipV
                py = snap.WorkH - py;
            end

            s = px / snap.WorkW;
            t = py / snap.WorkH;
            st = [s, t];
            if ~isempty(snap.SrcPx)
                uv = warpPointsSectionToAtlas(st, snap.SrcPx, snap.DstPx, snap.WorkW, snap.WorkH);
            else
                uv = st;
            end

            voxel = snap.O + uv(:, 1) .* snap.U + uv(:, 2) .* snap.V;
            coords = obj.toUnits(voxel, opts.Units);
            inside = (s >= 0.0 & s <= 1.0) & (t >= 0.0 & t <= 1.0);
        end

        function res = coord_atlas_to_image(obj, xyz, opts)
            %COORD_ATLAS_TO_IMAGE Back-project atlas voxels to image pixels.
            %   RES = R.COORD_ATLAS_TO_IMAGE(XYZ) matches each row of the
            %   (N,3) atlas-voxel array XYZ to the nearest section whose
            %   footprint covers it, returning a scalar struct:
            %       section_id  (N,1) string, "" where uncovered
            %       xy          (N,2) double, nan where uncovered
            %       distance    (N,1) double, inf where uncovered
            %       valid       (N,1) logical
            %
            %   Name-value options:
            %       Space ("full" | "working", default "full")
            %       Units ("voxel" | "um" | "mm", default "voxel")
            %       MaxDistance (default Inf) -- voxels farther than this
            %           (in Units) from the matched plane are marked invalid.
            %
            %   Mirrors engine/registration.py::coord_atlas_to_image.
            arguments
                obj
                xyz (:, 3) double
                opts.Space (1, 1) string {mustBeMember(opts.Space, ["full", "working"])} = "full"
                opts.Units (1, 1) string {mustBeMember(opts.Units, ["voxel", "um", "mm"])} = "voxel"
                opts.MaxDistance (1, 1) double = Inf
            end
            n = size(xyz, 1);
            bestDist = inf(n, 1);
            sectionId = repmat("", n, 1);
            xy = nan(n, 2);

            for k = 1:numel(obj.Snapshots)
                snap = obj.Snapshots(k);
                if ~snap.Aligned
                    continue
                end
                normal = cross(snap.U, snap.V);
                normalNorm = norm(normal);
                if normalNorm == 0.0
                    continue
                end
                normal = normal / normalNorm;

                rel = xyz - snap.O;               % (n, 3)
                dist = abs(rel * normal');         % (n, 1)

                pinvUV = pinv([snap.U; snap.V]');  % (2, 3), matches np.linalg.pinv(column_stack([u,v]))
                uv = rel * pinvUV';                % (n, 2)
                insideMask = uv(:, 1) >= 0.0 & uv(:, 1) <= 1.0 ...
                    & uv(:, 2) >= 0.0 & uv(:, 2) <= 1.0;

                better = insideMask & (dist < bestDist);
                if ~any(better)
                    continue
                end

                stBetter = warpPointsAtlasToSection( ...
                    uv(better, :), snap.SrcPx, snap.DstPx, snap.WorkW, snap.WorkH);
                px = stBetter(:, 1) * snap.WorkW;
                py = stBetter(:, 2) * snap.WorkH;
                if snap.FlipH
                    px = snap.WorkW - px;
                end
                if snap.FlipV
                    py = snap.WorkH - py;
                end
                if opts.Space == "full"
                    px = px * snap.FullW / snap.WorkW;
                    py = py * snap.FullH / snap.WorkH;
                end

                bestDist(better) = dist(better);
                sectionId(better) = string(snap.Id);
                xy(better, 1) = px;
                xy(better, 2) = py;
            end

            covered = isfinite(bestDist);
            distance = obj.toUnits(bestDist, opts.Units);
            valid = covered;
            if ~isinf(opts.MaxDistance)
                valid = valid & (distance <= opts.MaxDistance);
            end

            res = struct("section_id", sectionId, "xy", xy, "distance", distance, "valid", valid);
        end

        function [img, inBounds] = image_to_atlas(obj, sliceKey, opts)
            %IMAGE_TO_ATLAS Resample an atlas volume onto SLICEKEY's own pixel grid.
            %   IMG = R.IMAGE_TO_ATLAS(SLICEKEY) resamples the annotation
            %   volume onto the section's full-resolution pixel grid,
            %   accounting for the section's affine anchoring, nonlinear
            %   (Delaunay) warp, and preprocessing flips.
            %
            %   Name-value options:
            %       Kind ("annotation" | "template" | "boundary", default "annotation")
            %       Space ("full" | "working", default "full")
            %
            %   Returns, matching KIND:
            %       "annotation" -- (H,W) int32 region-ID array (0 = background/out-of-atlas)
            %       "template"   -- (H,W) uint8 grayscale array (0 outside the atlas volume)
            %       "boundary"   -- (H,W) logical edge mask
            %
            %   [IMG, INBOUNDS] = ... also returns an (H,W) logical mask,
            %   true where the pixel's atlas voxel lies inside the atlas volume.
            %
            %   Mirrors engine/registration.py::image_to_atlas.
            arguments
                obj
                sliceKey (1, 1) string
                opts.Kind (1, 1) string {mustBeMember(opts.Kind, ["annotation", "template", "boundary"])} = "annotation"
                opts.Space (1, 1) string {mustBeMember(opts.Space, ["full", "working"])} = "full"
            end
            snap = obj.getSnapshot(sliceKey);
            if ~snap.Aligned
                error("verso:VersoRegistration:notAligned", ...
                    "Section '%s' has no alignment; cannot map pixels.", snap.Id);
            end

            if opts.Space == "full"
                outW = snap.FullW;
                outH = snap.FullH;
            else
                outW = snap.WorkW;
                outH = snap.WorkH;
            end

            atlas = obj.getAtlasVolume();

            needsLabels = opts.Kind ~= "template";
            if needsLabels
                labels = zeros(outH, outW, "int32");
            end
            if opts.Kind == "template"
                gray = zeros(outH, outW, "uint8");
            end
            inBounds = false(outH, outW);

            % Row-chunked so full-resolution images (tens of thousands of px
            % per side) don't require an all-at-once (H*W, 2) warp-lookup buffer.
            rowsPerChunk = max(1, floor(obj.ChunkPixelBudget / outW));
            xs = ((0:outW - 1) + 0.5) / outW;

            row0 = 0;
            while row0 < outH
                row1 = min(outH - 1, row0 + rowsPerChunk - 1);   % 0-based, inclusive
                rows0 = (row0:row1)';                             % 0-based row indices
                ys = (rows0 + 0.5) / outH;

                [ssGrid, ttGrid] = meshgrid(xs, ys);   % (chunkRows, outW)
                if snap.FlipH
                    ssGrid = 1.0 - ssGrid;
                end
                if snap.FlipV
                    ttGrid = 1.0 - ttGrid;
                end

                st = [ssGrid(:), ttGrid(:)];
                if ~isempty(snap.SrcPx)
                    uv = warpPointsSectionToAtlas(st, snap.SrcPx, snap.DstPx, snap.WorkW, snap.WorkH);
                else
                    uv = st;
                end

                voxelFlat = snap.O + uv(:, 1) .* snap.U + uv(:, 2) .* snap.V;  % (chunkRows*outW, 3)
                chunkRows = numel(rows0);
                voxelGrid = reshape(voxelFlat, chunkRows, outW, 3);

                rowsIdx = rows0 + 1;  % MATLAB 1-based row range for this chunk
                if opts.Kind == "template"
                    [chunkGray, chunkInside] = sampleReferenceAt(atlas, voxelGrid);
                    gray(rowsIdx, :) = chunkGray;
                else
                    [chunkLabels, chunkInside] = sampleLabelsAt(atlas, voxelGrid);
                    chunkLabels(~chunkInside) = 0;
                    labels(rowsIdx, :) = chunkLabels;
                end
                inBounds(rowsIdx, :) = chunkInside;

                row0 = row1 + 1;
            end

            if opts.Kind == "boundary"
                img = boundaryMask(labels, inBounds);
            elseif opts.Kind == "template"
                img = gray;
            else
                img = labels;
            end
        end
    end

    methods (Hidden)
        function setAtlasVolumeForTesting(obj, atlasVol)
            %SETATLASVOLUMEFORTESTING Inject a fake atlas volume (test-only).
            %   Bypasses resolveAtlasDir/loadAtlasVolume (and therefore any
            %   network access) so unit tests can exercise image_to_atlas
            %   offline against a small synthetic atlas. ATLASVOL must have
            %   the same fields as loadAtlasVolume's return value (Annotation,
            %   Reference, RefScale, ColorTable, ResolutionUm, Shape).
            obj.AtlasVol = atlasVol;
        end

        function setChunkPixelBudgetForTesting(obj, budget)
            %SETCHUNKPIXELBUDGETFORTESTING Shrink the image_to_atlas chunk size (test-only).
            %   Lowers the row-chunk pixel budget so image_to_atlas takes the
            %   multi-chunk path on small test images, exercising the chunk
            %   loop without allocating a full-resolution image. BUDGET is a
            %   positive pixel count.
            obj.ChunkPixelBudget = budget;
        end
    end

    methods (Static, Hidden)
        % Thin forwarders exposing the +verso/private numeric primitives to the
        % cross-language parity test (matlab/tests/tParity.m). Same test-hook
        % pattern as the setAtlasVolumeForTesting methods above: Hidden, so they
        % are not part of the public API -- they only let the pure primitives be
        % checked directly against the Python golden fixture.
        function out = warpSectionToAtlasForTesting(pointsNorm, srcPx, dstPx, workW, workH)
            out = warpPointsSectionToAtlas(pointsNorm, srcPx, dstPx, workW, workH);
        end

        function out = warpAtlasToSectionForTesting(pointsNorm, srcPx, dstPx, workW, workH)
            out = warpPointsAtlasToSection(pointsNorm, srcPx, dstPx, workW, workH);
        end

        function ouv = anchoringVectorsForTesting(anchoring)
            % Returns a 3x3 matrix with rows [o; u; v].
            [o, u, v] = anchoringToVectors(anchoring);
            ouv = [o; u; v];
        end

        function idx = sampleVoxelIndicesForTesting(coords)
            % coords: (M,3) continuous [lr ap dv] -> (M,3) [lr ap dv] indices.
            [lr, ap, dv] = sampleVoxelIndices(coords(:, 1), coords(:, 2), coords(:, 3));
            idx = [lr, ap, dv];
        end
    end

    methods (Access = private)
        function snap = getSnapshot(obj, key)
            idx = obj.resolveSlice(key);
            snap = obj.Snapshots(idx);
        end

        function idx = resolveSlice(obj, key)
            key = char(key);
            exact = find(strcmp(obj.IdsList, key), 1);
            if ~isempty(exact)
                idx = exact;
                return
            end

            byStem = [];
            byName = [];
            for k = 1:numel(obj.Snapshots)
                [~, stem, ext] = fileparts(obj.Snapshots(k).OriginalPath);
                name = [stem, ext];
                if strcmp(stem, key)
                    byStem(end + 1) = k; %#ok<AGROW>
                end
                if strcmp(name, key)
                    byName(end + 1) = k; %#ok<AGROW>
                end
            end

            if ~isempty(byStem)
                matches = byStem;
            else
                matches = byName;
            end

            if numel(matches) == 1
                idx = matches(1);
            elseif isempty(matches)
                error("verso:VersoRegistration:noSuchSlice", ...
                    "No section matches '%s'. Available ids: %s", key, strjoin(obj.IdsList, ", "));
            else
                error("verso:VersoRegistration:ambiguousSlice", ...
                    "Slice '%s' is ambiguous; candidate ids: %s", key, strjoin(obj.IdsList(matches), ", "));
            end
        end

        function atlas = getAtlasVolume(obj)
            if isempty(obj.AtlasVol)
                atlasDir = resolveAtlasDir(obj.AtlasNameVal, obj.AtlasDirOverride);
                obj.AtlasVol = loadAtlasVolume(atlasDir);
            end
            atlas = obj.AtlasVol;
        end

        function out = toUnits(obj, voxel, units)
            switch units
                case "voxel"
                    out = voxel;
                case "um"
                    out = voxel * obj.ResolutionUm;
                otherwise  % "mm"
                    out = voxel * obj.ResolutionUm / 1000.0;
            end
        end
    end
end
