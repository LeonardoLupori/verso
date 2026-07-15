function atlasDir = downloadBrainglobeAtlas(atlasName, brainglobeDir)
%DOWNLOADBRAINGLOBEATLAS Fetch and extract a BrainGlobe atlas from the GIN remote.
%   ATLASDIR = DOWNLOADBRAINGLOBEATLAS(ATLASNAME, BRAINGLOBEDIR) mirrors
%   brainglobe_atlasapi's own download_extract_file():
%     1. Fetch last_versions.conf (an INI file, [atlases] section) to
%        resolve ATLASNAME's current version.
%     2. Download {atlasName}_v{X.Y}.tar.gz.
%     3. Extract it into BRAINGLOBEDIR.
%   See .claude/matlab-port.md for the exact URLs/format this depends on.
    baseUrl = "https://gin.g-node.org/brainglobe/atlases/raw/master/";

    version = iResolveRemoteVersion(baseUrl, atlasName);

    archiveName = atlasName + "_v" + version + ".tar.gz";
    archiveUrl = baseUrl + archiveName;
    tempArchive = fullfile(tempdir, "verso_" + archiveName);

    try
        websave(tempArchive, archiveUrl);
    catch err
        error("verso:downloadBrainglobeAtlas:downloadFailed", ...
            "Failed to download atlas '%s' from %s (%s). Check your internet " + ...
            "connection, or download it once via Python " + ...
            "(BrainGlobeAtlas('%s')) and pass its folder as AtlasDir.", ...
            atlasName, archiveUrl, err.message, atlasName);
    end

    cleanupObj = onCleanup(@() iDeleteIfExists(tempArchive));
    untar(tempArchive, brainglobeDir);

    atlasDir = string(fullfile(brainglobeDir, atlasName + "_v" + version));
    if ~isfolder(atlasDir)
        error("verso:downloadBrainglobeAtlas:extractionMismatch", ...
            "Downloaded and extracted '%s', but the expected folder %s was " + ...
            "not created. The atlas package layout may have changed.", ...
            archiveName, atlasDir);
    end
end

function version = iResolveRemoteVersion(baseUrl, atlasName)
    confUrl = baseUrl + "last_versions.conf";
    try
        confText = webread(confUrl);
    catch err
        error("verso:downloadBrainglobeAtlas:noConnection", ...
            "Could not reach %s to resolve the latest atlas version (%s). " + ...
            "Check your internet connection, or pass AtlasDir explicitly.", ...
            confUrl, err.message);
    end

    confText = string(confText);
    lines = splitlines(confText);
    inSection = false;
    version = "";
    % Index-based loop (not "for line = lines'") -- splitlines' output
    % orientation should not be relied on for for-loop iteration semantics.
    for k = 1:numel(lines)
        t = strtrim(lines(k));
        if startsWith(t, "[")
            inSection = strcmpi(t, "[atlases]");
            continue
        end
        if inSection && contains(t, "=")
            parts = split(t, "=");
            key = strtrim(parts(1));
            if strcmpi(key, atlasName)
                version = strtrim(strjoin(parts(2:end), "="));
                break
            end
        end
    end

    if version == ""
        error("verso:downloadBrainglobeAtlas:unknownAtlas", ...
            "Atlas '%s' was not found in %s -- check the name matches " + ...
            "project-verso.json's atlas.name exactly.", atlasName, confUrl);
    end
end

function iDeleteIfExists(p)
    if isfile(p)
        delete(p);
    end
end
