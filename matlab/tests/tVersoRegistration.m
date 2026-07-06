classdef tVersoRegistration < matlab.unittest.TestCase
    %TVERSOREGISTRATION Offline tests for verso.VersoRegistration.
    %   Mirrors tests/engine/test_registration.py's approach: builds small
    %   project-verso.json fixtures on disk (no real BrainGlobe atlas needed
    %   for the coordinate-only tests) and, for image_to_atlas, injects a
    %   small synthetic atlas volume via the Hidden test hook
    %   setAtlasVolumeForTesting, so tests run offline and fast.
    %
    %   Run with: runtests('matlab/tests')

    properties (Constant)
        AtlasAP = 20
        AtlasDV = 16
        AtlasLR = 24
        ResolutionUm = 25.0
    end

    methods (Test)
        function roundtripNoControlPoints(testCase)
            proj = testCase.projectFile({testCase.section("s1", 10.0)});
            reg = verso.VersoRegistration(proj);

            p = [30.0 20.0; 70.0 50.0];
            xyz = reg.coord_image_to_atlas("s1", p);
            testCase.verifySize(xyz, [2 3]);

            res = reg.coord_atlas_to_image(xyz);
            testCase.verifyEqual(res.section_id, ["s1"; "s1"]);
            testCase.verifyEqual(res.distance, [0; 0], "AbsTol", 1e-6);
            testCase.verifyEqual(res.xy, p, "AbsTol", 1e-6);
            testCase.verifyTrue(all(res.valid));
        end

        function roundtripWithFlips(testCase)
            proj = testCase.projectFile({testCase.section("s1", 10.0, "FlipH", true, "FlipV", true)});
            reg = verso.VersoRegistration(proj);

            p = [30.0 20.0];
            xyz = reg.coord_image_to_atlas("s1", p);
            res = reg.coord_atlas_to_image(xyz);
            testCase.verifyEqual(res.section_id(1), "s1");
            testCase.verifyEqual(res.xy, p, "AbsTol", 1e-6);
        end

        function roundtripWorkingSpace(testCase)
            proj = testCase.projectFile({testCase.section("s1", 10.0)});
            reg = verso.VersoRegistration(proj);

            p = [12.0 8.0];
            xyz = reg.coord_image_to_atlas("s1", p, "Space", "working");
            res = reg.coord_atlas_to_image(xyz, "Space", "working");
            testCase.verifyEqual(res.xy, p, "AbsTol", 1e-6);
        end

        function roundtripWithControlPoints(testCase)
            cps = [ ...
                struct("src_x", 10.0, "src_y", 8.0, "dst_x", 14.0, "dst_y", 6.0), ...
                struct("src_x", 30.0, "src_y", 20.0, "dst_x", 26.0, "dst_y", 24.0), ...
                struct("src_x", 40.0, "src_y", 12.0, "dst_x", 38.0, "dst_y", 16.0)];
            sec = testCase.section("s1", 10.0, "ControlPoints", cps, "Work", [48 32], "Full", [48 32]);
            proj = testCase.projectFile({sec});
            reg = verso.VersoRegistration(proj);

            p = [[cps.dst_x]', [cps.dst_y]'];
            xyz = reg.coord_image_to_atlas("s1", p);

            anch = testCase.canonicalAnchoring(10.0, 1);
            expected = zeros(numel(cps), 3);
            for k = 1:numel(cps)
                [o, u, v] = testCase.anchoringVectors(anch);
                expected(k, :) = o + (cps(k).src_x / 48) * u + (cps(k).src_y / 32) * v;
            end
            testCase.verifyEqual(xyz, expected, "AbsTol", 1e-6);

            res = reg.coord_atlas_to_image(xyz);
            testCase.verifyEqual(res.section_id, repmat("s1", 3, 1));
            testCase.verifyEqual(res.xy, p, "AbsTol", 1e-6);
        end

        function nearestSectionPicksCloserPlane(testCase)
            proj = testCase.projectFile({testCase.section("s1", 8.0), testCase.section("s2", 14.0)});
            reg = verso.VersoRegistration(proj);

            res = reg.coord_atlas_to_image([12.0 9.0 8.0]);
            testCase.verifyEqual(res.section_id(1), "s1");
            testCase.verifyEqual(res.distance(1), 1.0, "AbsTol", 1e-6);
            testCase.verifyTrue(res.valid(1));

            res2 = reg.coord_atlas_to_image([12.0 12.0 8.0]);
            testCase.verifyEqual(res2.section_id(1), "s2");
            testCase.verifyEqual(res2.distance(1), 2.0, "AbsTol", 1e-6);
        end

        function voxelOutsideAllFootprintsIsInvalid(testCase)
            proj = testCase.projectFile({testCase.section("s1", 8.0)});
            reg = verso.VersoRegistration(proj);

            res = reg.coord_atlas_to_image([100.0 8.0 8.0]);
            testCase.verifyEqual(res.section_id(1), "");
            testCase.verifyFalse(res.valid(1));
            testCase.verifyFalse(isfinite(res.distance(1)));
            testCase.verifyTrue(all(isnan(res.xy(1, :))));
        end

        function unitsForward(testCase)
            proj = testCase.projectFile({testCase.section("s1", 10.0)});
            reg = verso.VersoRegistration(proj);

            p = [30.0 20.0];
            vox = reg.coord_image_to_atlas("s1", p);
            testCase.verifyEqual( ...
                reg.coord_image_to_atlas("s1", p, "Units", "um"), vox * testCase.ResolutionUm, "AbsTol", 1e-6);
            testCase.verifyEqual( ...
                reg.coord_image_to_atlas("s1", p, "Units", "mm"), vox * testCase.ResolutionUm / 1000.0, "AbsTol", 1e-9);
        end

        function returnValidFlagsOutOfFrame(testCase)
            proj = testCase.projectFile({testCase.section("s1", 10.0)});  % full == [96 64]
            reg = verso.VersoRegistration(proj);

            p = [30.0 20.0; 999.0 20.0; -5.0 10.0];
            [coords, inside] = reg.coord_image_to_atlas("s1", p);
            testCase.verifySize(coords, [3 3]);
            testCase.verifyEqual(inside, [true; false; false]);
        end

        function sliceResolverByIdStemAndBasename(testCase)
            proj = testCase.projectFile({ ...
                testCase.section("s1", 8.0, "OriginalPath", "/data/IMG_1.tif"), ...
                testCase.section("s2", 14.0, "OriginalPath", "/data/IMG_2.tif")});
            reg = verso.VersoRegistration(proj);

            testCase.verifyTrue(reg.hasSection("s1"));
            testCase.verifyTrue(reg.hasSection("IMG_1"));
            testCase.verifyTrue(reg.hasSection("IMG_2.tif"));
            testCase.verifyFalse(reg.hasSection("nope"));
        end

        function badSpaceAndUnitsRaise(testCase)
            proj = testCase.projectFile({testCase.section("s1", 10.0)});
            reg = verso.VersoRegistration(proj);
            testCase.verifyError(@() reg.coord_image_to_atlas("s1", [1.0 1.0], "Space", "nope"), ...
                "MATLAB:validators:mustBeMember");
            testCase.verifyError(@() reg.coord_image_to_atlas("s1", [1.0 1.0], "Units", "parsecs"), ...
                "MATLAB:validators:mustBeMember");
        end

        function imageToAtlasAnnotationMatchesPointwiseLookup(testCase)
            sec = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32]);
            proj = testCase.projectFile({sec});
            reg = verso.VersoRegistration(proj);
            reg.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            labels = reg.image_to_atlas("s1", "Kind", "annotation");
            testCase.verifySize(labels, [64 96]);
            testCase.verifyClass(labels, "int32");

            cols = [5, 50, 90];
            rows = [5, 10, 60];
            pts = [cols' + 0.5, rows' + 0.5];
            xyz = reg.coord_image_to_atlas("s1", pts);
            expected = testCase.expectedLabelsAt(xyz);
            actual = zeros(1, 3, "int32");
            for k = 1:3
                actual(k) = labels(rows(k) + 1, cols(k) + 1);
            end
            testCase.verifyEqual(actual, expected);
        end

        function imageToAtlasAnnotationWithWarp(testCase)
            cps = [ ...
                struct("src_x", 10.0, "src_y", 8.0, "dst_x", 20.0, "dst_y", 8.0), ...
                struct("src_x", 30.0, "src_y", 20.0, "dst_x", 20.0, "dst_y", 20.0), ...
                struct("src_x", 10.0, "src_y", 25.0, "dst_x", 20.0, "dst_y", 25.0)];
            secWarp = testCase.section("s1", 10.0, "ControlPoints", cps, "Work", [48 32], "Full", [48 32]);
            secFlat = testCase.section("s1", 10.0, "Work", [48 32], "Full", [48 32]);

            regWarp = verso.VersoRegistration(testCase.projectFile({secWarp}));
            regWarp.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());
            regFlat = verso.VersoRegistration(testCase.projectFile({secFlat}));
            regFlat.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            labelsWarp = regWarp.image_to_atlas("s1", "Kind", "annotation");
            labelsFlat = regFlat.image_to_atlas("s1", "Kind", "annotation");

            testCase.verifyEqual(size(labelsWarp), size(labelsFlat));
            testCase.verifyNotEqual(labelsWarp, labelsFlat);
        end

        function imageToAtlasTemplate(testCase)
            sec = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32]);
            reg = verso.VersoRegistration(testCase.projectFile({sec}));
            reg.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            [gray, inBounds] = reg.image_to_atlas("s1", "Kind", "template");
            testCase.verifySize(gray, [64 96]);
            testCase.verifyClass(gray, "uint8");
            testCase.verifySize(inBounds, [64 96]);
            testCase.verifyTrue(all(gray(~inBounds) == 0));
        end

        function imageToAtlasBoundary(testCase)
            sec = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32]);
            reg = verso.VersoRegistration(testCase.projectFile({sec}));
            reg.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            boundary = reg.image_to_atlas("s1", "Kind", "boundary");
            labels = reg.image_to_atlas("s1", "Kind", "annotation");

            testCase.verifyClass(boundary, "logical");
            testCase.verifyEqual(size(boundary), size(labels));
            testCase.verifyTrue(any(boundary, "all"));
        end

        function imageToAtlasWorkingSpace(testCase)
            sec = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32]);
            reg = verso.VersoRegistration(testCase.projectFile({sec}));
            reg.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            labels = reg.image_to_atlas("s1", "Kind", "annotation", "Space", "working");
            testCase.verifySize(labels, [32 48]);
        end

        function imageToAtlasBadKindRaises(testCase)
            reg = verso.VersoRegistration(testCase.projectFile({testCase.section("s1", 10.0)}));
            testCase.verifyError(@() reg.image_to_atlas("s1", "Kind", "nope"), ...
                "MATLAB:validators:mustBeMember");
        end

        function imageToAtlasChunkingMatchesSingleChunk(testCase)
            % Gap 2: a shrunken chunk budget forces the multi-chunk row loop;
            % its output must be identical to the single-chunk (default) path.
            sec = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32]);
            proj = testCase.projectFile({sec});

            regSingle = verso.VersoRegistration(proj);
            regSingle.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            regChunked = verso.VersoRegistration(proj);
            regChunked.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());
            regChunked.setChunkPixelBudgetForTesting(100);  % 100/96 -> 1 row per chunk => 64 chunks

            for kind = ["annotation", "template", "boundary"]
                [imgSingle, boundsSingle] = regSingle.image_to_atlas("s1", "Kind", kind);
                [imgChunked, boundsChunked] = regChunked.image_to_atlas("s1", "Kind", kind);
                testCase.verifyEqual(imgChunked, imgSingle, ...
                    sprintf("chunked %s output differs from single-chunk", kind));
                testCase.verifyEqual(boundsChunked, boundsSingle, ...
                    sprintf("chunked %s inBounds differs from single-chunk", kind));
            end
        end

        function coordImageToAtlasUnalignedRaises(testCase)
            % Gap 3: a degenerate plane (u x v == 0) is not aligned.
            degenerate = [0 10 0, 24 0 0, 0 0 0];
            proj = testCase.projectFile({testCase.section("s1", 10.0, "Anchoring", degenerate)});
            reg = verso.VersoRegistration(proj);
            testCase.verifyError(@() reg.coord_image_to_atlas("s1", [10.0 10.0]), ...
                "verso:VersoRegistration:notAligned");
        end

        function imageToAtlasUnalignedRaises(testCase)
            % Gap 3: image_to_atlas rejects an unaligned section too.
            degenerate = [0 10 0, 24 0 0, 0 0 0];
            proj = testCase.projectFile({testCase.section("s1", 10.0, "Anchoring", degenerate)});
            reg = verso.VersoRegistration(proj);
            reg.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());
            testCase.verifyError(@() reg.image_to_atlas("s1"), ...
                "verso:VersoRegistration:notAligned");
        end

        function coordAtlasToImageSkipsUnaligned(testCase)
            % Gap 3: unaligned sections are skipped in the nearest-plane search.
            degenerate = [0 10 0, 24 0 0, 0 0 0];
            proj = testCase.projectFile({testCase.section("s1", 10.0, "Anchoring", degenerate)});
            reg = verso.VersoRegistration(proj);

            res = reg.coord_atlas_to_image([12.0 9.0 10.0]);
            testCase.verifyEqual(res.section_id(1), "");
            testCase.verifyFalse(res.valid(1));
            testCase.verifyFalse(isfinite(res.distance(1)));
        end

        function coordAtlasToImageMaxDistance(testCase)
            % Gap 4: voxels farther than MaxDistance from the matched plane are
            % covered (matched) but marked invalid.
            proj = testCase.projectFile({testCase.section("s1", 8.0)});
            reg = verso.VersoRegistration(proj);

            % One AP unit off the plane at position 8 -> distance 1 voxel.
            xyz = [12.0 9.0 8.0];
            near = reg.coord_atlas_to_image(xyz, "MaxDistance", 2.0);
            testCase.verifyEqual(near.section_id(1), "s1");
            testCase.verifyEqual(near.distance(1), 1.0, "AbsTol", 1e-6);
            testCase.verifyTrue(near.valid(1));

            far = reg.coord_atlas_to_image(xyz, "MaxDistance", 0.5);
            testCase.verifyEqual(far.section_id(1), "s1");   % still matched
            testCase.verifyFalse(far.valid(1));              % but out of tolerance

            % MaxDistance is expressed in the requested Units.
            farUm = reg.coord_atlas_to_image(xyz, "Units", "um", "MaxDistance", testCase.ResolutionUm * 0.5);
            testCase.verifyFalse(farUm.valid(1));
            nearUm = reg.coord_atlas_to_image(xyz, "Units", "um", "MaxDistance", testCase.ResolutionUm * 2.0);
            testCase.verifyTrue(nearUm.valid(1));
        end

        function imageToAtlasRespectsFlips(testCase)
            % Gap 5: flips change which atlas voxel each output pixel samples,
            % and the flipped result stays consistent with the coordinate path.
            secFlat = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32]);
            secFlip = testCase.section("s1", 10.0, "Full", [96 64], "Work", [48 32], ...
                "FlipH", true);

            regFlat = verso.VersoRegistration(testCase.projectFile({secFlat}));
            regFlat.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());
            regFlip = verso.VersoRegistration(testCase.projectFile({secFlip}));
            regFlip.setAtlasVolumeForTesting(testCase.fakeAtlasVolume());

            labelsFlat = regFlat.image_to_atlas("s1", "Kind", "annotation");
            labelsFlip = regFlip.image_to_atlas("s1", "Kind", "annotation");
            testCase.verifyNotEqual(labelsFlip, labelsFlat);

            % The fake atlas splits regions 1|2 along LR; a horizontal flip
            % mirrors columns, so the flipped image equals the flat one mirrored.
            testCase.verifyEqual(labelsFlip, fliplr(labelsFlat));

            % And each output pixel still agrees with a pointwise coordinate lookup.
            cols = [5, 50, 90];
            rows = [5, 10, 60];
            pts = [cols' + 0.5, rows' + 0.5];
            xyz = regFlip.coord_image_to_atlas("s1", pts);
            expected = testCase.expectedLabelsAt(xyz);
            actual = zeros(1, 3, "int32");
            for k = 1:3
                actual(k) = labelsFlip(rows(k) + 1, cols(k) + 1);
            end
            testCase.verifyEqual(actual, expected);
        end

        function idsAndNumSections(testCase)
            % Gap 6: the ids()/numSections() accessors report project order.
            proj = testCase.projectFile({ ...
                testCase.section("s1", 8.0), testCase.section("s2", 14.0)});
            reg = verso.VersoRegistration(proj);
            testCase.verifyEqual(reg.ids(), {'s1', 's2'});
            testCase.verifyEqual(reg.numSections(), 2);
        end

        function emptyProjectHasNoSections(testCase)
            % Gap 6: a project with zero sections builds and reports empty.
            reg = verso.VersoRegistration(testCase.projectFile({}));
            testCase.verifyEqual(reg.numSections(), 0);
            testCase.verifyEmpty(reg.ids());
            testCase.verifyFalse(reg.hasSection("s1"));
        end
    end

    methods (Access = private)
        function anchoring = canonicalAnchoring(testCase, position, axis)
            % Mirrors AtlasVolume.canonical_plane_anchoring(position, axis=1): coronal
            % plane spanning the full LR x DV extent at the given AP position.
            if nargin < 3
                axis = 1;
            end
            if axis == 1
                o = [0.0, position, 0.0];
                u = [double(testCase.AtlasLR), 0.0, 0.0];
                v = [0.0, 0.0, double(testCase.AtlasDV)];
            else
                error("unsupported axis in test helper");
            end
            anchoring = [o, u, v];
        end

        function [o, u, v] = anchoringVectors(~, anchoring)
            o = anchoring(1:3);
            u = anchoring(4:6);
            v = anchoring(7:9);
        end

        function s = section(testCase, id, position, varargin)
            p = inputParser;
            addParameter(p, "Axis", 1);
            addParameter(p, "FlipH", false);
            addParameter(p, "FlipV", false);
            addParameter(p, "ControlPoints", struct("src_x", {}, "src_y", {}, "dst_x", {}, "dst_y", {}));
            addParameter(p, "Work", [48 32]);
            addParameter(p, "Full", [96 64]);
            addParameter(p, "OriginalPath", id + ".tif");
            addParameter(p, "Anchoring", []);
            parse(p, varargin{:});
            r = p.Results;

            if isempty(r.Anchoring)
                anchoring = testCase.canonicalAnchoring(position, r.Axis);
            else
                anchoring = r.Anchoring;
            end

            s = struct();
            s.id = id;
            s.slice_index = floor(position);
            s.original_path = char(r.OriginalPath);
            s.thumbnail_path = char(id + ".ome.tif");
            s.resolution_original_wh = r.Full;
            s.resolution_thumbnail_wh = r.Work;
            s.preprocessing = struct( ...
                "flip_horizontal", r.FlipH, "flip_vertical", r.FlipV, "slice_mask_path", "");
            s.alignment = struct( ...
                "anchoring", anchoring, "position_mm", 0, "status", "complete", "source", "manual");
            if isempty(r.ControlPoints)
                cpsOut = {};
            else
                cpsOut = num2cell(r.ControlPoints);
            end
            s.warp = struct("control_points", {cpsOut}, "status", "in_progress");
        end

        function path = projectFile(testCase, sections)
            proj = struct();
            proj.version = "1.3";
            proj.name = "t";
            proj.atlas = struct( ...
                "name", "fake", "source", "brainglobe", ...
                "resolution_um", testCase.ResolutionUm, ...
                "shape", [testCase.AtlasAP, testCase.AtlasDV, testCase.AtlasLR]);
            proj.interpolation_axis = "AP";
            proj.channels = {};
            proj.cp_size = 10;
            proj.cp_shape = "Cross";
            proj.cp_color = "#fff500";
            proj.working_scale = 0.5;
            proj.sections = sections;

            tmpDir = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            path = fullfile(tmpDir, "project-verso.json");
            fid = fopen(path, "w");
            fwrite(fid, jsonencode(proj));
            fclose(fid);
            path = string(path);
        end

        function vol = fakeAtlasVolume(testCase)
            % Two regions split along LR (x) at the midpoint, so warping and
            % flipping visibly change which region a pixel samples.
            ann = ones(testCase.AtlasAP, testCase.AtlasDV, testCase.AtlasLR, "int32");
            half = floor(testCase.AtlasLR / 2);
            ann(:, :, (half + 1):end) = 2;

            ref = reshape(0:(testCase.AtlasAP * testCase.AtlasDV * testCase.AtlasLR - 1), ...
                testCase.AtlasAP, testCase.AtlasDV, testCase.AtlasLR);
            ref = double(ref);
            refMax = max(ref(:));
            if refMax > 0
                refScale = 255.0 / refMax;
            else
                refScale = 1.0;
            end

            colorTable = containers.Map("KeyType", "double", "ValueType", "any");
            colorTable(0) = uint8([0 0 0]);
            colorTable(1) = uint8([255 0 0]);
            colorTable(2) = uint8([0 255 0]);

            vol = struct( ...
                "Annotation", ann, "Reference", ref, "RefScale", refScale, ...
                "ColorTable", colorTable, "ResolutionUm", testCase.ResolutionUm, ...
                "Shape", [testCase.AtlasAP, testCase.AtlasDV, testCase.AtlasLR]);
        end

        function expected = expectedLabelsAt(testCase, xyz)
            lr = floor(xyz(:, 1));
            ap = ceil(xyz(:, 2));
            dv = ceil(xyz(:, 3));
            apMax = testCase.AtlasAP; dvMax = testCase.AtlasDV; lrMax = testCase.AtlasLR;
            inside = (ap >= 0 & ap < apMax) & (dv >= 0 & dv < dvMax) & (lr >= 0 & lr < lrMax);
            half = floor(lrMax / 2);
            lrClamped = min(max(lr, 0), lrMax - 1);
            expected = int32(zeros(size(xyz, 1), 1));
            for k = 1:size(xyz, 1)
                if inside(k)
                    if lrClamped(k) >= half
                        expected(k) = 2;
                    else
                        expected(k) = 1;
                    end
                end
            end
            expected = expected';
        end
    end
end
