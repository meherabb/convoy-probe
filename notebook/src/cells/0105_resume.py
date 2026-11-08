# PLACEHOLDER
# 0.5 Stage checkpoints. Every expensive stage (rendering, real-data preparation, optical flow,
#     each model on each testbed) is archived as checkpoints/<stage>.zip the moment it finishes and
#     appended at once to convoy_checkpoints.zip, so that archive always holds every stage finished
#     so far, even if the session stops later. The paper's figures, tables and results go into a
#     second archive, convoy_paper_outputs.zip (Section 13). A later session that attaches the archive
#     (or this run's output) restores each stage whose fingerprint matches, byte for byte.
import zipfile, zlib

CKPT_FORMAT = "convoy-checkpoint/1"
_STORED_EXT = (".png", ".jpg", ".jpeg", ".npz", ".zip", ".gz")      # already compressed: store as is
_STAGE_REG_DIR = os.path.join(MANIFEST_DIR, "stages")
os.makedirs(_STAGE_REG_DIR, exist_ok=True)
STAGE_LOG = []
SEARCH_ROOTS = []
for _r in ["/kaggle/input", os.environ.get("CONVOY_RESUME_FROM", ""), "/content/drive/MyDrive/convoy_checkpoints"]:
    if _r and os.path.exists(_r) and os.path.abspath(_r) not in SEARCH_ROOTS:
        SEARCH_ROOTS.append(os.path.abspath(_r))


def canonical(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def stage_fingerprint(name, family, config=None, upstream=()):
    rec = dict(format=CKPT_FORMAT, stage=name, version=STAGE_VERSIONS[family], seed=GLOBAL_SEED,
               config=canonical(config), upstream=sorted(str(u) for u in upstream))
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:20]


def _arcname(path):
    p = os.path.abspath(path)
    for tag, root in (("S", SCRATCH_DIR), ("O", OUTPUT_DIR)):
        r = os.path.abspath(root)
        if p.startswith(r + os.sep):
            return tag + "/" + os.path.relpath(p, r).replace(os.sep, "/")
    raise ValueError(f"{path} is outside the scratch and output folders")


def _destination(arc):
    tag, rel = arc.split("/", 1)
    root = os.path.abspath({"S": SCRATCH_DIR, "O": OUTPUT_DIR}[tag])
    dst = os.path.normpath(os.path.join(root, rel))
    if not dst.startswith(root + os.sep):
        raise ValueError("unsafe path in a checkpoint: " + arc)
    return dst


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _zip_meta(path):
    try:
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("stage.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


# ---- the progressive archive convoy_checkpoints.zip -----------------------------------------------
_ARCHIVE = dict(names=None, stale=set(), enabled=True, note="")
_KAGGLE_OUTPUT_CAP = 19.5 * 2**30                    # Kaggle keeps at most 20 GB of /kaggle/working


def _dir_bytes(root):
    tot = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            try:
                tot += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return tot


def _crc_of(path):
    c = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def _archive_names():
    if _ARCHIVE["names"] is None:
        _ARCHIVE["names"] = {}
        if os.path.exists(ZIP_PATH):
            try:
                with zipfile.ZipFile(ZIP_PATH) as zf:
                    _ARCHIVE["names"] = {i.filename: (i.file_size, i.CRC) for i in zf.infolist()}
            except (zipfile.BadZipFile, OSError):
                os.remove(ZIP_PATH)                  # an interrupted append: start the archive again
    return _ARCHIVE["names"]


def archive_add(path, arcname, compress=False):
    # append one file to convoy_checkpoints.zip unless an identical copy is already in it
    if not _ARCHIVE["enabled"] or not os.path.exists(path):
        return
    names = _archive_names()
    size = os.path.getsize(path)
    if arcname in names:
        if names[arcname][0] != size or names[arcname][1] != _crc_of(path):
            _ARCHIVE["stale"].add(arcname)            # superseded in this folder: the archive is rebuilt at the end
        return
    if ENV_NAME == "kaggle" and _dir_bytes(ROOT_WORK) + size > _KAGGLE_OUTPUT_CAP:
        _ARCHIVE.update(enabled=False, note="stopped: the duplicate copy would exceed Kaggle's 20 GB output limit "
                                            "(checkpoints/ alone is the resume point)")
        print("[archive] " + _ARCHIVE["note"])
        return
    with zipfile.ZipFile(ZIP_PATH, "a", allowZip64=True) as zf:
        zf.write(path, arcname, compress_type=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED)
        names[arcname] = (size, zf.getinfo(arcname).CRC)


def archive_stage(zip_path):
    archive_add(zip_path, "checkpoints/" + os.path.basename(zip_path))


def _extract(zf):
    for info in zf.infolist():
        if info.filename == "stage.json" or info.is_dir():
            continue
        dst = _destination(info.filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zf.open(info) as src, open(dst + ".part", "wb") as out:
            shutil.copyfileobj(src, out, 1 << 22)
        os.replace(dst + ".part", dst)


class _PreviousRuns:
    # index of the stage archives found in attached data (lazy: built on first use)
    def __init__(self):
        self.entries = None

    def _add(self, meta, loc):
        if isinstance(meta, dict) and meta.get("format") == CKPT_FORMAT:
            self.entries.setdefault(meta["stage"], []).append((loc, meta))

    def _scan_zip(self, path):
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                if "stage.json" in names:
                    self._add(json.loads(zf.read("stage.json")), ("zip", path, None))
                    return
                for n in names:
                    if re.search(r"(^|/)checkpoints/[^/]+\.zip$", n):
                        with zf.open(n) as fh, zipfile.ZipFile(fh) as inner:
                            if "stage.json" in inner.namelist():
                                self._add(json.loads(inner.read("stage.json")), ("nested", path, n))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError):
            pass

    def build(self):
        self.entries = {}
        for root in SEARCH_ROOTS:
            for dirpath, dirnames, filenames in os.walk(root):
                if os.path.abspath(dirpath).startswith(os.path.abspath(CKPT_DIR)) or \
                   dirpath[len(root):].count(os.sep) >= 7:
                    dirnames[:] = []
                    continue
                if "stage.json" in filenames:
                    self._add(_read_json(os.path.join(dirpath, "stage.json")), ("dir", dirpath, None))
                    dirnames[:] = []                     # an extracted stage archive: nothing further below
                for fn in filenames:
                    if fn.lower().endswith(".zip"):
                        self._scan_zip(os.path.join(dirpath, fn))

    def find(self, name, fp):
        if self.entries is None:
            self.build()
        for loc, meta in self.entries.get(name, []):
            if meta.get("fingerprint") == fp:
                return loc, meta
        return None

    def summary(self):
        if self.entries is None:
            self.build()
        return {k: len(v) for k, v in sorted(self.entries.items())}


PREVIOUS = _PreviousRuns()


def _fmt_seconds(sec):
    sec = float(sec or 0)
    return f"{sec:.0f} s" if sec < 90 else f"{sec / 60:.1f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


def _where(kind, path, member=None):
    # where a checkpoint came from, in words; only the archive or folder name is shown (an attached
    # dataset's mount path can contain the account name, which this notebook never prints)
    if kind == "checkpoints":
        return "this run's checkpoints/ folder"
    if kind == "session":
        return "this session"
    return f"the attached data ({os.path.basename(path.rstrip(os.sep))})"


_SESSION_SOURCE = {}                                  # stage -> where it came from in this session


def _original_seconds(meta):
    ex = meta.get("extra") or {}
    return ex.get("seconds") if isinstance(ex.get("seconds"), (int, float)) else meta.get("compute_seconds", 0.0)


class Stage:
    """One resumable unit of work: restore it if a matching checkpoint exists, else compute and commit.

    Every call of done() and commit() prints one line, so each cell states plainly whether its work
    was restored from a checkpoint (nothing is re-run) or is being computed in this session:
        [checkpoint] RESTORED  <stage>  from <where>: no need to re-run (...)
        [checkpoint] RUNNING   <stage>  no matching checkpoint ...: running it now
        [checkpoint] SAVED     <stage>  computed in ...; archived in checkpoints/ and convoy_checkpoints.zip
    """

    def __init__(self, name, family, config=None, upstream=()):
        self.name, self.family = name, family
        self.fp = stage_fingerprint(name, family, config, upstream)
        self.zip_path = os.path.join(CKPT_DIR, _safe(name) + ".zip")
        self.reg_path = os.path.join(_STAGE_REG_DIR, _safe(name) + ".json")
        self.t0 = time.time()

    def _present(self, meta):
        try:
            return all(os.path.exists(_destination(a)) and os.path.getsize(_destination(a)) == s for a, s in meta["files"])
        except (KeyError, ValueError):
            return False

    def _register(self, meta):
        with open(self.reg_path, "w") as f:
            json.dump(meta, f, indent=1)

    def _say(self, word, text):
        print(f"[checkpoint] {word:<9} {self.name:<22} {text}", flush=True)

    def _log(self, status, source, meta):
        mib = round(sum(s for _, s in meta.get("files", [])) / 2**20, 1)
        took = float(_original_seconds(meta) or 0.0) if status == "computed" else time.time() - self.t0
        STAGE_LOG.append(dict(stage=self.name, status=status, source=source, fingerprint=self.fp,
                              files=len(meta.get("files", [])), MiB=mib, seconds=round(took, 1),
                              original_seconds=round(float(_original_seconds(meta) or 0.0), 1)))
        prev = _SESSION_SOURCE.get(self.name)
        if status == "kept":
            if prev and prev != "computed":
                self._say("RESTORED", f"restored earlier in this session from {prev}: no need to re-run")
            else:
                self._say("COMPLETE", ("computed earlier in this session" if prev else "already complete in this folder")
                                      + ": no need to re-run")
            return
        _SESSION_SOURCE[self.name] = source if status == "restored" else "computed"
        if status == "restored":
            self._say("RESTORED", f"from {source}: no need to re-run ({len(meta.get('files', []))} files, {mib:,.0f} MiB, "
                                  f"{_fmt_seconds(time.time() - self.t0)} to restore; computing it took "
                                  f"{_fmt_seconds(_original_seconds(meta))})")
        else:
            self._say("SAVED", f"computed in {_fmt_seconds(took)}; archived in checkpoints/{os.path.basename(self.zip_path)}"
                               + (" and convoy_checkpoints.zip" if _ARCHIVE["enabled"] else ""))

    def _not_found(self):
        if PREVIOUS.entries is None:
            PREVIOUS.build()
        others = [m_ for _, m_ in PREVIOUS.entries.get(self.name, []) if m_.get("fingerprint") != self.fp]
        old = _zip_meta(self.zip_path)
        if old and old.get("fingerprint") != self.fp:
            others.append(old)
        why = (f"; {len(others)} archived version(s) of this stage were found but were made with other settings or "
               f"code, so they are not used" if others else "")
        self._say("RUNNING", f"no matching checkpoint in checkpoints/ or the attached data{why}: running it now")

    def _write_zip(self, meta):
        tmp = self.zip_path + ".part"
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            zf.writestr("stage.json", json.dumps(meta, indent=1))
            for arc, _ in meta["files"]:
                ct = zipfile.ZIP_STORED if arc.lower().endswith(_STORED_EXT) else zipfile.ZIP_DEFLATED
                zf.write(_destination(arc), arc, compress_type=ct)
        os.replace(tmp, self.zip_path)
        archive_stage(self.zip_path)

    def available(self):
        # where this stage could be taken from, without extracting anything (None: must be computed)
        meta = _read_json(self.reg_path)
        if meta and meta.get("fingerprint") == self.fp and self._present(meta):
            return "this session"
        meta = _zip_meta(self.zip_path)
        if meta and meta.get("fingerprint") == self.fp:
            return _where("checkpoints", self.zip_path)
        hit = PREVIOUS.find(self.name, self.fp)
        return _where(*hit[0]) if hit else None

    def done(self):
        # True if the stage's outputs are in place (kept or restored); False if it must be computed.
        # Order: this session -> this run's checkpoints/ -> any attached data (Kaggle inputs, CONVOY_RESUME_FROM)
        self.t0 = time.time()
        meta = _read_json(self.reg_path)
        if meta and meta.get("fingerprint") == self.fp and self._present(meta):
            if _zip_meta(self.zip_path) is None:
                self._write_zip(meta)
            else:
                archive_stage(self.zip_path)
            self._log("kept", _where("session", ""), meta)
            return True
        meta = _zip_meta(self.zip_path)
        if meta and meta.get("fingerprint") == self.fp:
            with zipfile.ZipFile(self.zip_path) as zf:
                _extract(zf)
            if self._present(meta):
                self._register(meta); self._log("restored", _where("checkpoints", self.zip_path), meta)
                archive_stage(self.zip_path)
                return True
        hit = PREVIOUS.find(self.name, self.fp)
        if hit:
            (kind, path, member), meta = hit
            if kind == "zip":
                with zipfile.ZipFile(path) as zf:
                    _extract(zf)
                if os.path.abspath(path) != os.path.abspath(self.zip_path):
                    shutil.copyfile(path, self.zip_path + ".part"); os.replace(self.zip_path + ".part", self.zip_path)
            elif kind == "nested":
                with zipfile.ZipFile(path) as outer:
                    with outer.open(member) as fh, zipfile.ZipFile(fh) as zf:
                        _extract(zf)
                    with outer.open(member) as fh, open(self.zip_path + ".part", "wb") as out:
                        shutil.copyfileobj(fh, out, 1 << 22)
                os.replace(self.zip_path + ".part", self.zip_path)
            else:
                for arc, _ in meta["files"]:
                    dst = _destination(arc)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(os.path.join(path, *arc.split("/")), dst)
                self._write_zip(meta)
            if self._present(meta):
                self._register(meta); self._log("restored", _where(kind, path, member), meta)
                archive_stage(self.zip_path)
                return True
        self._not_found()
        return False

    def commit(self, paths, extra=None):
        files = []
        for p in paths if isinstance(paths, (list, tuple)) else [paths]:
            if os.path.isdir(p):
                for dp, _, fns in os.walk(p):
                    files += [os.path.join(dp, fn) for fn in sorted(fns) if not fn.endswith(".part")]
            elif os.path.exists(p):
                files.append(p)
        files = sorted(set(files))
        meta = dict(format=CKPT_FORMAT, stage=self.name, family=self.family, fingerprint=self.fp,
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    files=[[_arcname(f), os.path.getsize(f)] for f in files], extra=extra or {},
                    compute_seconds=round(time.time() - self.t0, 1))
        self._write_zip(meta)
        self._register(meta)
        self._log("computed", "this session", meta)
        return meta


def stage_table(prefix=None):
    df = pd.DataFrame(STAGE_LOG)
    if prefix and len(df):
        df = df[df["stage"].str.startswith(prefix)]
    return df


# ---- the second archive, download links and an optional off-session copy (v6.2) --------------------
_PAPER_SUBDIRS = ("figures", "tables", "results", "manifest", "logs")
PAPER_README = (
    "convoy_paper_outputs.zip: everything needed to check the paper's numbers, figures and tables.\n"
    "  figures/   every figure as vector PDF and 300-dpi PNG, with the plotted data (*_data.json)\n"
    "  tables/    formatted tables as CSV, JSON and booktabs LaTeX (T*.csv / .json / .tex)\n"
    "  results/   unformatted results of every analysis (estimates, intervals, bins, claim audit)\n"
    "  manifest/  idea lock, amendments, decisions, run configuration, model and data status, stage log,\n"
    "             figure captions (figures.json), table captions (tables.json), RAM log, file checksums\n"
    "  logs/      environment, installation, testbed self-tests, ground-truth validation, model status\n"
    "The resume point is the other archive, convoy_checkpoints.zip (attach it as a Kaggle dataset).\n")


def build_paper_zip(note=""):
    tmp, n = PAPER_ZIP + ".part", 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for sub in _PAPER_SUBDIRS:
            for dp, _, fns in os.walk(os.path.join(OUTPUT_DIR, sub)):
                for fn in sorted(fns):
                    p = os.path.join(dp, fn)
                    zf.write(p, "paper_outputs/" + os.path.relpath(p, OUTPUT_DIR).replace(os.sep, "/"),
                             compress_type=zipfile.ZIP_STORED if fn.lower().endswith(_STORED_EXT) else zipfile.ZIP_DEFLATED)
                    n += 1
        zf.writestr("paper_outputs/README.txt", PAPER_README + note)
    os.replace(tmp, PAPER_ZIP)
    return n


def offer_downloads(paths, auto=False):
    # Download links in the cell output. A browser can only receive a file while it is connected to
    # the session; auto=True also asks it to start the downloads (best effort). For a run that must
    # survive a closed browser, use Save Version -> Save & Run All (the files then stay in the
    # version's Output tab) and/or the dataset copy below.
    items = [p for p in paths if p and os.path.exists(p)]
    if not items:
        return
    try:
        from IPython.display import display as _disp, HTML as _HTML, Javascript as _JS
    except ImportError:
        return
    rows = []
    for p in items:
        rel = os.path.relpath(p, os.getcwd())
        href = rel if not rel.startswith("..") else p
        rows.append(f'<li><a href="{href}" download="{os.path.basename(p)}" target="_blank">{os.path.basename(p)}</a> '
                    f'({os.path.getsize(p) / 2**20:,.0f} MiB)</li>')
    _disp(_HTML("<p>Download (right-click and save if the browser does not start it):</p><ul>" + "".join(rows) + "</ul>"))
    if auto:
        names = [os.path.basename(p) for p in items]
        _disp(_JS("setTimeout(function(){var n=%s;document.querySelectorAll('a').forEach(function(a){"
                  "if(n.indexOf(a.textContent.trim())>=0){try{a.click();}catch(e){}}});},1500);" % json.dumps(names)))


def _kaggle_api_ready():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        _s = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
        return bool(os.environ["KAGGLE_USERNAME"] and os.environ["KAGGLE_KEY"])
    except Exception:
        return False


def backup_to_dataset(paths, message):
    # Copies the archives into a PRIVATE Kaggle dataset <user>/<dataset_backup_slug> with the Kaggle API,
    # so they survive a closed browser or a stopped session and can be attached directly next time.
    # Active only on Kaggle with the secrets KAGGLE_USERNAME and KAGGLE_KEY attached (Add-ons -> Secrets).
    if not RUN_CONFIG.get("dataset_backup") or ENV_NAME != "kaggle" or not _kaggle_api_ready():
        return "not configured (attach the secrets KAGGLE_USERNAME and KAGGLE_KEY to enable it)"
    import subprocess
    user, slug = os.environ["KAGGLE_USERNAME"], RUN_CONFIG["dataset_backup_slug"]
    d = os.path.join(SCRATCH_DIR, "dataset_backup")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for p in paths:
        if os.path.exists(p):
            try:
                os.link(p, os.path.join(d, os.path.basename(p)))
            except OSError:
                shutil.copy2(p, d)
    with open(os.path.join(d, "dataset-metadata.json"), "w") as f:
        json.dump(dict(title="Convoy ambiguity checkpoints", id=f"{user}/{slug}", licenses=[{"name": "CC0-1.0"}]), f)
    cli = shutil.which("kaggle")
    if cli is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], capture_output=True, timeout=600)
        cli = shutil.which("kaggle") or "kaggle"
    note = ""
    for cmd in ([cli, "datasets", "version", "-p", d, "-m", message[:200], "-q"], [cli, "datasets", "create", "-p", d, "-q"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        except Exception as e:                        # never stop the run for a backup
            note = f"{type(e).__name__}: {e}"
            continue
        if r.returncode == 0:
            shutil.rmtree(d, ignore_errors=True)
            return f"copied to your private Kaggle dataset '{slug}' ({cmd[2]})"
        note = (r.stderr or r.stdout).strip()[-300:]
    shutil.rmtree(d, ignore_errors=True)
    return "dataset copy failed: " + note.replace(user, "<account>")         # the account name is never printed


print(f"Checkpoint folder: {CKPT_DIR}")
print("=" * 100)
if SEARCH_ROOTS:
    _found = PREVIOUS.summary()
    _labels = ["/kaggle/input" if r == "/kaggle/input" else "CONVOY_RESUME_FROM" if r == os.path.abspath(os.environ.get("CONVOY_RESUME_FROM", "-"))
               else "Google Drive" for r in SEARCH_ROOTS]
    print(f"RESUME CHECK: searched {', '.join(_labels)} for checkpoints of a previous run.")
    if _found:
        def _src_label(kind, path):
            # nested: the outer archive; loose or extracted stage archives: the folder that holds them
            path = path.rstrip(os.sep)
            if kind != "nested" and os.path.basename(os.path.dirname(path)) == "checkpoints":
                return os.path.basename(os.path.dirname(os.path.dirname(path))) + "/checkpoints/"
            return os.path.basename(path)
        _src = {}
        for _lst in PREVIOUS.entries.values():
            for (_kind, _path, _m), _ in _lst:
                _lab = _src_label(_kind, _path)
                _src[_lab] = _src.get(_lab, 0) + 1
        print(f"  found {sum(_found.values())} stage checkpoint(s) for {len(_found)} stage(s) in: "
              + "; ".join(f"{k} ({v})" for k, v in sorted(_src.items())))
        print("  Each stage is restored when its cell is reached, if its fingerprint (settings, code version and inputs)\n"
              "  matches this run; the cell then prints 'RESTORED ... no need to re-run'. A stage without a match prints\n"
              "  'RUNNING ... running it now' and is computed and saved.")
        print(textwrap.fill(", ".join(_found), width=110, initial_indent="  stages found: ", subsequent_indent="    "))
    else:
        print("  no stage checkpoints in the attached data: every stage runs in this session (and is saved as it finishes).")
else:
    print("RESUME CHECK: nothing attached; every stage runs in this session (and is saved as it finishes).\n"
          "  To resume later, attach convoy_checkpoints.zip from this run's output as a Kaggle dataset (Section 0.5).")
print("=" * 100)