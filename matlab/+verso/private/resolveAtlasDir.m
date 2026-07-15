function atlasDir = resolveAtlasDir(atlasName, atlasDirOverride)
%RESOLVEATLASDIR Find (or download) the BrainGlobe atlas folder for ATLASNAME.
%   ATLASDIR = RESOLVEATLASDIR(ATLASNAME, ATLASDIROVERRIDE) resolves the
%   folder containing annotation.tiff / reference.tiff / structures.json /
%   metadata.json for the given BrainGlobe atlas name:
%
%   1. If ATLASDIROVERRIDE is non-empty, validate and use it directly.
%   2. Otherwise look for an already-downloaded copy under the standard
%      BrainGlobe cache directory ({brainglobe_dir}/{atlasName}_v*).
%   3. Otherwise download it from the BrainGlobe/GIN remote and extract it
%      into the cache directory (see downloadBrainglobeAtlas).
%
%   See .claude/matlab-port.md for the full cache-format / download-protocol
%   reference this mirrors from brainglobe_atlasapi.
    if nargin < 2
        atlasDirOverride = "";
    end
    atlasDirOverride = string(atlasDirOverride);

    if strlength(atlasDirOverride) > 0
        atlasDir = atlasDirOverride;
        iValidateAtlasDir(atlasDir);
        return
    end

    brainglobeDir = iDefaultBrainglobeDir();
    if ~isfolder(brainglobeDir)
        mkdir(brainglobeDir);
    end

    candidates = dir(fullfile(brainglobeDir, atlasName + "_v*"));
    candidates = candidates([candidates.isdir]);

    if numel(candidates) > 1
        names = strjoin({candidates.name}, ", ");
        error("verso:resolveAtlasDir:ambiguous", ...
            "Multiple versions of atlas '%s' found in %s: %s. " + ...
            "Remove all but one, or pass AtlasDir explicitly.", ...
            atlasName, brainglobeDir, names);
    elseif numel(candidates) == 1
        atlasDir = string(fullfile(candidates(1).folder, candidates(1).name));
        return
    end

    atlasDir = downloadBrainglobeAtlas(atlasName, brainglobeDir);
end

function iValidateAtlasDir(atlasDir)
    required = ["annotation.tiff", "structures.json", "metadata.json"];
    for r = required
        if ~isfile(fullfile(atlasDir, r))
            error("verso:resolveAtlasDir:missingFile", ...
                "Atlas directory '%s' is missing required file '%s' " + ...
                "(expected the standard BrainGlobe atlas folder layout).", ...
                atlasDir, r);
        end
    end
end

function d = iDefaultBrainglobeDir()
    configDir = string(getenv("BRAINGLOBE_CONFIG_DIR"));
    if configDir == ""
        configDir = fullfile(iUserHomeDir(), ".config", "brainglobe");
    end
    confPath = fullfile(configDir, "bg_config.conf");

    d = "";
    if isfile(confPath)
        d = iReadBrainglobeDirFromConf(confPath);
    end
    if d == ""
        d = string(fullfile(iUserHomeDir(), ".brainglobe"));
    end
end

function d = iReadBrainglobeDirFromConf(confPath)
    % Index-based loop (not "for line = lines'") -- splitlines' output
    % orientation should not be relied on for for-loop iteration semantics.
    d = "";
    lines = splitlines(string(fileread(confPath)));
    inSection = false;
    for k = 1:numel(lines)
        t = strtrim(lines(k));
        if startsWith(t, "[")
            inSection = strcmpi(t, "[default_dirs]");
            continue
        end
        if inSection && contains(t, "=")
            parts = split(t, "=");
            key = strtrim(parts(1));
            if strcmpi(key, "brainglobe_dir")
                d = strtrim(strjoin(parts(2:end), "="));
                return
            end
        end
    end
end

function h = iUserHomeDir()
    h = char(java.lang.System.getProperty("user.home"));
end
