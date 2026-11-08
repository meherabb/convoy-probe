# PLACEHOLDER
# 4.1 Model code: repositories cloned at pinned commits and imported from source.
#     No requirements.txt is installed (see Section 0.3); minimal dependencies were
#     installed under the constraints file in Section 0.3.2.
REPOS = {
    "dust3r":       dict(url="https://github.com/naver/dust3r.git",
                         commit="4c24a6ebf04809f2cfe59915e51779c8984aaa40", submodules=True,
                         syspath=["", "croco"], position="front"),
    "mast3r":       dict(url="https://github.com/naver/mast3r.git",
                         commit="f5209afc300cec36239a7ac992263f36847bbba0", submodules=True,
                         syspath=[""], position="back"),      # after dust3r: shares naver/dust3r code
    "vggt":         dict(url="https://github.com/facebookresearch/vggt.git",
                         commit="a288dd0f14786c93483e45524328726ab7b1b4ce", submodules=False,
                         syspath=[""], position="front"),
    "pi3":          dict(url="https://github.com/yyfz/Pi3.git",
                         commit="9fa3ddb3f8d53041f8b2738df404f62223bbaa7b", submodules=False,
                         syspath=[""], position="front"),
    "map-anything": dict(url="https://github.com/facebookresearch/map-anything.git",
                         commit="3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9", submodules=False,
                         syspath=[""], position="front"),
}

def _git(args, cwd=None, timeout=900):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-400:]

REPO_STATUS = {}
with stage("clone_repos"):
    for name, spec in REPOS.items():
        d = os.path.join(REPO_DIR, name)
        note = []
        if not os.path.isdir(os.path.join(d, ".git")):
            shutil.rmtree(d, ignore_errors=True)
            ok, msg = _git(["clone", "--filter=blob:none", spec["url"], d])
            if not ok:
                REPO_STATUS[name] = dict(ok=False, commit=None, note=f"clone failed: {msg}")
                continue
        ok, msg = _git(["checkout", "-q", spec["commit"]], cwd=d)
        note.append("pinned commit" if ok else f"pinned commit unavailable, using HEAD ({msg[:80]})")
        if spec["submodules"]:
            ok_s, msg_s = _git(["submodule", "update", "--init", "--recursive"], cwd=d)
            if not ok_s:
                note.append(f"submodules failed: {msg_s[:120]}")
        _, head = _git(["rev-parse", "HEAD"], cwd=d)
        for sub in spec["syspath"]:
            p = os.path.join(d, sub) if sub else d
            if p not in sys.path:
                if spec["position"] == "front":
                    sys.path.insert(0, p)
                else:
                    sys.path.append(p)
        REPO_STATUS[name] = dict(ok=True, commit=head.strip()[:12], note="; ".join(note))
importlib.invalidate_caches()
for k, v in REPO_STATUS.items():
    print(f"{k:<14}{'PASS' if v['ok'] else 'FAIL':<6}{str(v['commit']):<14}{v['note']}")