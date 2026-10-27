# E9: independent RadImageNet ResNet-50 arm for the verified E2 parent.
#
# Adapted 2026-08-11 from the V52 cell in the public Kaggle competition notebook
# prvsiyan/rsna-knee-read-the-report-then-the-knee (latest source SHA-256
# b54aa529f38dc6f594478e7975d86459ddbff898453a2c2633f4c3be4b909e61).
# Kaggle's public-code rule deems public Competition Code open-source; Meta Kaggle
# documents public notebooks under Apache-2.0. Changes here remove the unavailable B3
# arm, pin the public E2 OOF bundle, add per-target/no-regression gates, and preserve E2
# byte-for-byte on every failure. This module is appended as the final notebook cell;
# its imports and DICOM helpers are supplied by the parent notebook.
SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
N_SLOT = len(SLOTS)
CACHE_SLICES = 8
TIME_BUDGET = 8.72 * 3600
IMG = CACHE_IMG = 224
# Match the released RadImageNet model's full-frame pretraining. Setting a crop
# larger than every acquisition disables the optional physical crop in read_slot.
CROP_MM = 10_000.0
SLICE_BAND = (0.12, 0.88)
# The audited public OOF was produced after the notebook's legacy member group, which
# left this process-global pixel contract active. Pin it instead of inheriting whichever
# E2 group happened to run last. This uses public preprocessing code only; no legacy
# checkpoint or unknown-license asset is attached.
RULES = dict(RULES_LEGACY)
TOKEN_DIM = 2048             # official ResNet-50 global-average feature
HEAD_DIM = 512
PINNED_HEADS_SHA256 = "0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa"
PINNED_REMOTE_AUDIT_SHA256 = "267f948078710d3ca8a6f0de4ce0a5e75e850e1f08d36451549a137d878a6fe8"
PINNED_PUBLIC_DIAGNOSTIC_SHA256 = "0f2f82fb40f0570d6766f73b0d7f51489df6d0faa8fd6e4c0f45bcd6c4c7b283"
PINNED_E9B_CONTRACT_SHA256 = "6777c0a0ba7dd044752fac948752dc39e9ca35b3c280fe74ee6d27a5865d87e7"
# E10 repairs E9's censored search: its alpha grid stopped at 0.25 and four of five outer
# folds selected that ceiling, so the deployed 0.20 was a boundary artifact rather than an
# optimum. The ladder, the two-source per-target gains and every deployable weight map live
# in the hash-pinned contract; this cell recomputes the remote half in-kernel before use.
# The contract also carries the held-out form of E10's own weight choice: selecting the rung
# on four grouped folds and scoring the fifth picks 0.60 (public) and 0.70 (v15) in all five
# outer folds, and never picks 0.20. So every deployable rung at or below 0.35 is below what
# honest selection would choose, which is the answer to "you tuned on the 58 gold rows".
PINNED_E10_CONTRACT_SHA256 = "219c91f40905181c222e2966b3fed01a96570ddfb64862357d5fd6cad500cd45"
E10_CONFIG = "uniform_035"
E10_PRESERVED_TARGETS = ["Baker's", "Fracture"]

# E11 trains a third arm whose diversity is in the pixels rather than in the weights. The
# existing arm reads three fat-suppressed slots at full frame; every one of E2's twenty
# members reads one DINOv2 recipe. Nothing in the portfolio has yet looked at a
# non-fat-suppressed series, where meniscal and ligament morphology is conventionally read,
# and nothing has given RadImageNet a physically normalised field of view. E11 changes both:
# three non-suppressed slots plus one suppressed anchor, cropped to 130 mm, which the parent
# notebook establishes is below the acquired field of view of 99.6% of series while still
# containing the joint. It is a training mode only; it never touches the submission.
ARM_MODE = "e11"
E11_SLOTS = [
    ("SAG_NOFS", "Sagittal", None, False),
    ("COR_NOFS", "Coronal", None, False),
    ("AX_NOFS", "Axial", None, False),
    ("SAG_FS", "Sagittal", None, True),
]
E11_CROP_MM = 130.0
E11_CACHE_SLICES = 8
E11_IMG = 224
# Availability of non-suppressed series per plane is unmeasured, so a fill floor rather than
# the parent's 90% rule: below this the run has found something structurally wrong, above it
# an empty slot is simply masked out of the token set like any other absent series.
E11_MIN_FILL = 0.45


def _v52_as_bool(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "T", "YES", "Y"}:
        return True
    if text in {"0", "FALSE", "F", "NO", "N"}:
        return False
    try:
        number = float(text)
        return True if number == 1 else False if number == 0 else None
    except Exception:
        return None


def audit_official_sequence_metadata(inferred, official):
    """Audit metadata agreement without changing the checkpoint pixel contract."""
    needed = {"SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression"}
    if inferred.empty or official.empty or not needed.issubset(official.columns):
        return
    inferred_flags = inferred[["SeriesInstanceUID", "fluid", "fatsat"]].copy()
    official_flags = official[
        ["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression"]
    ].copy()
    official_flags["official_fluid"] = official_flags["Fluid_Sensitive"].map(
        _v52_as_bool
    )
    official_flags["official_fatsat"] = official_flags["Fat_Suppression"].map(
        _v52_as_bool
    )
    merged = inferred_flags.merge(
        official_flags[["SeriesInstanceUID", "official_fluid", "official_fatsat"]],
        on="SeriesInstanceUID",
        how="inner",
    )
    for inferred_col, official_col, name in [
        ("fluid", "official_fluid", "Fluid_Sensitive"),
        ("fatsat", "official_fatsat", "Fat_Suppression"),
    ]:
        valid = merged[official_col].notna() & merged[inferred_col].notna()
        if valid.any():
            agreement = (
                merged.loc[valid, inferred_col].astype(bool).to_numpy()
                == merged.loc[valid, official_col].astype(bool).to_numpy()
            ).mean()
            log(
                f"V52 metadata audit {name}: {agreement:.1%} agreement "
                f"on {int(valid.sum())} series"
            )


def find_input_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    raise FileNotFoundError(name)


def find_input_dir(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        if Path(root).name == name:
            return Path(root)
    raise FileNotFoundError(name)


def make_targets(train):
    """Three independent public report teachers; image-read gold always wins."""
    uid = "StudyInstanceUID"
    sources = [
        pd.read_csv(find_input_file("report_labels_v2.csv")),
        pd.read_csv(find_input_file("llm_labels_v2.csv")),
        pd.read_csv(find_input_file("labels_llm_gpt56sol.csv")),
    ]
    cube = []
    for frame in sources:
        if frame[uid].duplicated().any():
            raise ValueError("duplicate study in report-label source")
        aligned = train[[uid]].merge(frame[[uid] + TARGETS], on=uid, how="left")
        cube.append(aligned[TARGETS].to_numpy(float))
    cube = np.stack(cube)
    available = np.isfinite(cube).sum(0)
    if np.any(available < 2):
        raise ValueError("fewer than two report teachers for a study/target")
    y = np.nanmean(cube, axis=0).astype(np.float32)
    disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = np.clip(2.0 * np.abs(y - .5), 0, 1)
    w = (.15 + .85 * (.65 * agreement + .35 * certainty)).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    w[gold] = 3.0
    return y, w, gold


def report_groups(train):
    report = (train.Report.fillna("").astype(str).str.lower()
              .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.array([hashlib.sha256(x.encode()).hexdigest()[:24] for x in report])


def _v52_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_radimagenet(device):
    """Strictly load the official RadImageNet ResNet-50 PyTorch checkpoint."""
    from torchvision.models import resnet50

    checkpoint = find_input_file("ResNet50.pt")
    expected_checkpoint = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
    observed_checkpoint = _v52_sha256(checkpoint)
    if observed_checkpoint != expected_checkpoint:
        raise RuntimeError(f"RadImageNet checkpoint drift: {observed_checkpoint}")

    class RadImageNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                *list(resnet50(weights=None).children())[:-2]
            )

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = RadImageNetEncoder()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not state or not all(str(key).startswith("backbone.") for key in state):
        raise RuntimeError("unexpected RadImageNet state-dict namespace")
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 23_508_032:
        raise RuntimeError(f"unexpected RadImageNet parameter count {parameter_count}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    log(
        f"RadImageNet strict load: {parameter_count:,} params; "
        f"inference GPUs={max(1, gpu_count)}"
    )
    return model


@torch.inference_mode()
def encode_radimagenet(cache, slot_mask, device):
    """Encode acquired slices with the official [-1, 1] RadImageNet contract."""
    n, slots, slices, h, w = cache.shape
    features = np.zeros((n, slots * slices, TOKEN_DIM), np.float16)
    token_mask = np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model = load_radimagenet(device)
    if device.type == "cuda":
        batch = 192 if torch.cuda.device_count() > 1 else 96
    else:
        batch = 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0:b0 + batch]
        x = torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        if feat.shape[1:] != (TOKEN_DIM,):
            raise RuntimeError(f"unexpected RadImageNet feature shape {tuple(feat.shape)}")
        features.reshape(-1, TOKEN_DIM)[ix] = (
            feat.float().cpu().numpy().astype(np.float16)
        )
        if b0 % (batch * 100) == 0:
            log(f"RadImageNet encoded {b0}/{len(valid)} acquired slices")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, token_mask.astype(np.float32)


class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(TOKEN_DIM),
                                     nn.Linear(TOKEN_DIM, HEAD_DIM), nn.GELU())
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * .01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, HEAD_DIM) * .01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4), nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(), nn.Dropout(.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        # No study should be empty, but keep MHA numerically defined if one is.
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token,
                                     key_padding_mask=key_padding,
                                     need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(TARGETS), -1)
        fused = self.fuse(torch.cat(
            [attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def macro_auc(y, pred):
    from sklearn.metrics import roc_auc_score
    hard = (np.asarray(y) >= .5).astype(np.uint8)
    values = [roc_auc_score(hard[:, j], pred[:, j])
              for j in range(hard.shape[1]) if np.unique(hard[:, j]).size == 2]
    return float(np.mean(values))


def _v52_target_auc(y, pred):
    from sklearn.metrics import roc_auc_score
    hard = (np.asarray(y) >= .5).astype(np.uint8)
    return {
        target: float(roc_auc_score(hard[:, index], pred[:, index]))
        for index, target in enumerate(TARGETS)
    }


@torch.inference_mode()
def predict_head(model, features, masks, indices, device, batch=64):
    model.eval()
    pred = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0:b0 + batch]
        x = torch.from_numpy(features[ix]).to(device)
        m = torch.from_numpy(masks[ix]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred.append(torch.sigmoid(model(x, m)).float().cpu())
    return torch.cat(pred).numpy()


def train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset
    class Rows(Dataset):
        def __init__(self, indices): self.indices = np.asarray(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, k):
            i = self.indices[k]
            return features[i], masks[i], y[i], weights[i]
    model = FoundationQueryHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    generator = torch.Generator().manual_seed(SEED + 100 + fold)
    loader = DataLoader(Rows(train_idx), batch_size=48, shuffle=True,
                        generator=generator, num_workers=2, pin_memory=True,
                        persistent_workers=True)
    best, best_auc, stale = None, -1.0, 0
    for epoch in range(24):
        model.train()
        for x, m, target, weight in loader:
            x, m = x.to(device), m.to(device)
            target, weight = target.to(device), weight.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = F.binary_cross_entropy_with_logits(logits, target,
                                                          reduction="none")
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = predict_head(model, features, masks, val_idx, device)
        score = macro_auc(y[val_idx], pred)
        log(f"fold {fold} epoch {epoch}: grouped weak-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5: break
    return best, best_auc


def _v52_rank_columns(values):
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64))
    return frame.rank(method="average", pct=True).to_numpy(np.float64)


def _v52_validate_submission(frame, expected_ids):
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("V52 submission schema drift")
    ids = frame["StudyInstanceUID"].astype(str).tolist()
    if ids != list(map(str, expected_ids)) or len(ids) != len(set(ids)):
        raise RuntimeError("V52 submission study identity/order drift")
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("V52 submission values are invalid")


def main_v52():
    import shutil
    from sklearn.model_selection import GroupKFold

    output = Path("/kaggle/working/rsna_rad_e9")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e9_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "evidence_boundary": (
            "All OOF values are local diagnostics on 58 official image labels; "
            "they are not Kaggle competition scores. E2 remains the primary unless "
            "strict artifact, OOF, inference, and submission gates all pass."
        ),
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0 (Kaggle-hosted weight metadata)",
        "encoder_sha256": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "base_oof_sha256": "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c",
        "parent": "E2 captured 20-member DINOv2 rank ensemble",
        "blend_contract": "rank columns independently, then 80% E2 plus 20% RadImageNet",
        "pixel_rules": dict(RULES),
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("V52 RadImageNet experiment requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = 8.72 * 3600 - elapsed
        audit["elapsed_before_v52_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 2.0 * 3600:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
        train_series = pd.read_csv(
            ROOT / "train_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        if len(train) != 4407:
            raise RuntimeError(f"unexpected train study count {len(train)}")
        plane = dict(zip(train_series.SeriesInstanceUID, train_series.Anatomical_Plane))
        headers = annotate(walk("train_series"))
        audit_official_sequence_metadata(headers, train_series)
        studies, pixels, slot_mask = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, "train-v52 "), "train-v52"
        )
        by_uid = {str(uid): i for i, uid in enumerate(studies)}
        missing = [uid for uid in train.StudyInstanceUID if uid not in by_uid]
        if missing:
            raise RuntimeError(f"{len(missing)} train studies absent from cache")
        order = np.array([by_uid[uid] for uid in train.StudyInstanceUID], dtype=np.int64)
        pixels, slot_mask = pixels[order], slot_mask[order]
        train_token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
        if train_token_count < int(0.90 * len(train) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient acquired train slices: {train_token_count}")
        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()

        y, weights, gold = make_targets(train)
        if int(gold.sum()) != 58:
            raise RuntimeError(f"expected 58 fully gold studies, observed {int(gold.sum())}")
        groups = report_groups(train)
        if len(np.unique(groups)) < 4000:
            raise RuntimeError("unexpected report-group collapse")

        splits = list(GroupKFold(5).split(features, groups=groups))
        fold_id = np.full(len(train), -1, dtype=np.int8)
        folds = []
        oof = np.zeros_like(y, dtype=np.float32)
        for fold, (tr, va) in enumerate(splits):
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"report leakage in fold {fold}")
            fold_id[va] = fold
            state, score = train_fold(
                features, token_mask, y, weights, tr, va, fold, device
            )
            if state is None:
                raise RuntimeError(f"fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            oof[va] = predict_head(head, features, token_mask, va, device)
            folds.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
            del head
            torch.cuda.empty_cache()
        if (fold_id < 0).any() or not np.isfinite(oof).all():
            raise RuntimeError("incomplete V52 OOF")

        weak_auc = macro_auc(y, oof)
        gold_auc = macro_auc(y[gold], oof[gold])
        log(f"V52 RadImageNet OOF weak macro AUC {weak_auc:.5f}")
        log(f"V52 RadImageNet OOF gold macro AUC {gold_auc:.5f} on 58 studies")
        torch.save(
            {
                "version": "v52-radimagenet-resnet50-official-1",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "encoder_source_commit": audit["encoder_source_commit"],
                "img": IMG,
                "slices_per_plane": CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "weak_oof_auc": weak_auc,
                "gold_oof_auc": gold_auc,
            },
            output / "v52_radimagenet_heads.pt",
        )
        oof_frame = pd.DataFrame(oof, columns=TARGETS)
        oof_frame.insert(0, "StudyInstanceUID", train.StudyInstanceUID)
        oof_frame["fold"] = fold_id
        oof_frame["is_gold"] = gold.astype(np.uint8)
        oof_frame.to_csv(output / "v52_oof.csv", index=False)

        base_npz = find_input_file("oof.npz")
        observed_base_hash = _v52_sha256(base_npz)
        if observed_base_hash != audit["base_oof_sha256"]:
            raise RuntimeError(f"E2 OOF artifact drift: {observed_base_hash}")
        with np.load(base_npz, allow_pickle=False) as base_bundle:
            expected_members = {"ids", "pred", "y_derived", "gold_mask", "targets"}
            if set(base_bundle.files) != expected_members:
                raise RuntimeError(f"unexpected E2 OOF members: {base_bundle.files}")
            base_ids = base_bundle["ids"].astype(str)
            base_targets = base_bundle["targets"].astype(str).tolist()
            base_gold = base_bundle["gold_mask"].astype(bool)
            base_prediction = base_bundle["pred"].astype(np.float64)
        if base_targets != TARGETS:
            raise RuntimeError("E2 OOF target order drift")
        if not np.array_equal(base_ids, train.StudyInstanceUID.astype(str).to_numpy()):
            raise RuntimeError("E2 OOF study order drift")
        if not np.array_equal(base_gold, gold):
            raise RuntimeError("E2 OOF gold mask differs from official train.csv")
        train_rows = np.flatnonzero(gold)
        if len(train_rows) != 58:
            raise RuntimeError(f"expected 58 E2 gold rows, observed {len(train_rows)}")
        gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
        exact_public = base_prediction[gold]
        rad = oof[gold].astype(np.float64)
        if not all(np.isfinite(x).all() for x in (gold_y, exact_public, rad)):
            raise RuntimeError("non-finite aligned E2/RadImageNet OOF value")

        base_rank = _v52_rank_columns(exact_public)
        rad_rank = _v52_rank_columns(rad)
        base_score = macro_auc(gold_y, base_rank)
        rad_score = macro_auc(gold_y, rad_rank)
        alpha_grid = np.array([0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25])
        gold_folds = fold_id[train_rows]
        if sorted(np.unique(gold_folds).tolist()) != [0, 1, 2, 3, 4]:
            raise RuntimeError("gold rows do not cover all five grouped folds")
        nested = np.zeros_like(base_rank)
        choices = []
        outer_train_scores = []
        for outer in range(5):
            tr = gold_folds != outer
            va = ~tr
            scored = []
            for alpha in alpha_grid:
                blend = (1.0 - alpha) * base_rank[tr] + alpha * rad_rank[tr]
                score = macro_auc(gold_y[tr], blend) - 0.01 * float(alpha)
                scored.append(float(score))
            best = max(range(len(alpha_grid)), key=lambda i: (scored[i], -alpha_grid[i]))
            alpha = float(alpha_grid[best])
            choices.append(alpha)
            outer_train_scores.append(scored)
            nested[va] = (1.0 - alpha) * base_rank[va] + alpha * rad_rank[va]
        nested_score = macro_auc(gold_y, nested)
        # Deployment weight is fixed by the independently scored public 0.906 mechanism.
        # The 58 gold rows may veto it and measure fold stability, but do not tune it.
        final_alpha = 0.20
        final_oof = (1.0 - final_alpha) * base_rank + final_alpha * rad_rank
        final_score = macro_auc(gold_y, final_oof)
        grid_scores = {
            f"{alpha:.3f}": macro_auc(
                gold_y, (1.0 - alpha) * base_rank + alpha * rad_rank
            )
            for alpha in alpha_grid
        }
        base_target_scores = _v52_target_auc(gold_y, base_rank)
        rad_target_scores = _v52_target_auc(gold_y, rad_rank)
        final_target_scores = _v52_target_auc(gold_y, final_oof)
        target_deltas = {
            target: final_target_scores[target] - base_target_scores[target]
            for target in TARGETS
        }
        target_regressions = {
            target: delta for target, delta in target_deltas.items() if delta < -1e-12
        }
        positive_folds = int(sum(alpha > 0 for alpha in choices))
        supported = bool(
            final_alpha > 0
            and positive_folds >= 3
            and nested_score >= base_score + 0.001
            and final_score >= base_score + 0.001
            and not target_regressions
        )
        audit["oof"] = {
            "rows": 58,
            "weak_macro_auc": weak_auc,
            "rad_gold_macro_auc": rad_score,
            "e2_macro_auc": base_score,
            "outer_fold_choices": choices,
            "outer_fold_penalized_train_scores": outer_train_scores,
            "nested_blend_macro_auc": nested_score,
            "final_alpha": final_alpha,
            "final_descriptive_macro_auc": final_score,
            "full_grid_macro_auc": grid_scores,
            "positive_outer_folds": positive_folds,
            "per_target": {
                target: {
                    "e2_auc": base_target_scores[target],
                    "radimagenet_auc": rad_target_scores[target],
                    "blend_auc": final_target_scores[target],
                    "blend_delta": target_deltas[target],
                }
                for target in TARGETS
            },
            "target_regressions": target_regressions,
            "gold_fold_counts": {
                str(fold): int((gold_folds == fold).sum()) for fold in range(5)
            },
            "selection_supported": supported,
        }
        audit["train_available_slice_tokens"] = train_token_count
        audit["head_count"] = len(folds)
        if not supported:
            audit["status"] = "OOF_REJECTED_E2_PRESERVED"
            log(
                f"V52 rejected by nested OOF: base={base_score:.5f}, "
                f"nested={nested_score:.5f}, final={final_score:.5f}, choices={choices}"
            )
            return

        del features, token_mask
        gc.collect()
        test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
        test_series = pd.read_csv(
            ROOT / "test_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        test_plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
        test_headers = annotate(walk("test_series"))
        audit_official_sequence_metadata(test_headers, test_series)
        test_studies, test_pixels, test_slot_mask = build_cache(
            pick_slots(test_headers, test_plane),
            test_plane,
            lat_of(test_headers, "test-v52 "),
            "test-v52",
        )
        test_by_uid = {str(uid): i for i, uid in enumerate(test_studies)}
        test_missing = [uid for uid in test.StudyInstanceUID if uid not in test_by_uid]
        if test_missing:
            raise RuntimeError(f"{len(test_missing)} test studies absent from cache")
        test_order = np.array([test_by_uid[uid] for uid in test.StudyInstanceUID])
        test_pixels = test_pixels[test_order]
        test_slot_mask = test_slot_mask[test_order]
        test_token_count = int(
            np.repeat(test_slot_mask[:, :, None], CACHE_SLICES, 2).sum()
        )
        if test_token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient acquired test slices: {test_token_count}")
        test_features, test_token_mask = encode_radimagenet(
            test_pixels, test_slot_mask, device
        )
        del test_pixels, test_slot_mask, test_headers
        gc.collect()

        fold_predictions = []
        all_test = np.arange(len(test), dtype=np.int64)
        for record in folds:
            head = FoundationQueryHead().to(device)
            head.load_state_dict(record["state_dict"], strict=True)
            fold_predictions.append(
                predict_head(head, test_features, test_token_mask, all_test, device)
            )
            del head
            torch.cuda.empty_cache()
        if len(fold_predictions) != 5:
            raise RuntimeError("test inference did not use all five heads")
        rad_test = np.mean(np.stack(fold_predictions), axis=0)
        if not np.isfinite(rad_test).all():
            raise RuntimeError("non-finite RadImageNet test prediction")

        baseline = pd.read_csv(preserved, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(baseline, test.StudyInstanceUID)
        rad_frame = pd.DataFrame(rad_test, columns=TARGETS)
        rad_frame.insert(0, "StudyInstanceUID", test.StudyInstanceUID)
        _v52_validate_submission(rad_frame, test.StudyInstanceUID)
        rad_frame.to_csv(output / "submission_rad_only.csv", index=False)
        baseline_rank = _v52_rank_columns(baseline[TARGETS].to_numpy())
        rad_test_rank = _v52_rank_columns(rad_test)
        selected_path = None
        for alpha in alpha_grid[1:]:
            candidate = baseline.copy()
            candidate[TARGETS] = (
                (1.0 - alpha) * baseline_rank + alpha * rad_test_rank
            )
            _v52_validate_submission(candidate, test.StudyInstanceUID)
            path = output / f"submission_e2_rad_{int(round(1000 * alpha)):03d}.csv"
            candidate.to_csv(path, index=False)
            if abs(float(alpha) - final_alpha) < 1e-12:
                selected_path = path
        if selected_path is None or not selected_path.is_file():
            raise RuntimeError("selected V52 blend artifact is absent")
        selected = pd.read_csv(selected_path, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(selected, test.StudyInstanceUID)
        audit["test_studies"] = len(test)
        audit["test_available_slice_tokens"] = test_token_count
        audit["test_head_count"] = len(fold_predictions)
        audit["selected_path"] = str(selected_path)
        audit["selected_sha256"] = _v52_sha256(selected_path)
        audit["fallback_sha256"] = _v52_sha256(preserved)
        shutil.copy2(selected_path, primary)
        if _v52_sha256(primary) != audit["selected_sha256"]:
            raise RuntimeError("primary V52 copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        log(
            f"E9 selected alpha={final_alpha:.3f}; "
            f"nested={nested_score:.5f} vs E2 OOF={base_score:.5f}"
        )
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E9 preserves E2: {audit['error']}")
    finally:
        if audit.get("status") != "CANDIDATE_SELECTED" and preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _v52_load_pinned_e9b():
    """Load the public v15 heads and reconstruct the dual-OOF target gate."""
    heads_path = find_input_file("v52_radimagenet_heads.pt")
    remote_path = find_input_file("rad_e9_audit.json")
    public_path = find_input_file("public_oof_diagnostic.json")
    contract_path = find_input_file("e9b_contract.json")
    expected_hashes = {
        heads_path: PINNED_HEADS_SHA256,
        remote_path: PINNED_REMOTE_AUDIT_SHA256,
        public_path: PINNED_PUBLIC_DIAGNOSTIC_SHA256,
        contract_path: PINNED_E9B_CONTRACT_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = _v52_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned E9b artifact drift for {path.name}: {observed}")

    remote = json.loads(remote_path.read_text())
    public = json.loads(public_path.read_text())
    contract = json.loads(contract_path.read_text())
    if remote.get("status") != "OOF_REJECTED_E2_PRESERVED":
        raise RuntimeError(f"unexpected v15 audit status {remote.get('status')}")
    if remote.get("primary_sha256") != (
        "f9fb57b7bac8489a5d5285b3984b06df57f142572be6417eac6341c43e96707a"
    ):
        raise RuntimeError("v15 did not preserve the exact E2 visible artifact")
    remote_oof = remote.get("oof", {})
    public_target = public.get("per_target", {})
    remote_target = remote_oof.get("per_target", {})
    if set(public_target) != set(TARGETS) or set(remote_target) != set(TARGETS):
        raise RuntimeError("E9b diagnostic target set drift")
    if int(remote.get("head_count", -1)) != 5:
        raise RuntimeError("v15 remote audit does not contain five heads")
    if int(remote_oof.get("positive_outer_folds", -1)) != 5:
        raise RuntimeError("v15 remote outer-fold support drift")
    if int(public.get("positive_outer_folds", -1)) != 5:
        raise RuntimeError("public outer-fold support drift")

    selected_targets = [
        target for target in TARGETS
        if float(public_target[target]["blend_delta"]) > 1e-12
        and float(remote_target[target]["blend_delta"]) > 1e-12
    ]
    preserved_targets = [target for target in TARGETS if target not in selected_targets]
    if selected_targets != contract.get("selected_targets"):
        raise RuntimeError(f"E9b selected-target contract drift: {selected_targets}")
    if preserved_targets != contract.get("preserved_targets"):
        raise RuntimeError(f"E9b preserved-target contract drift: {preserved_targets}")
    if len(selected_targets) != 10 or preserved_targets != ["Baker's", "Fracture"]:
        raise RuntimeError("E9b requires the ten-target dual-OOF intersection")
    alpha = float(contract.get("alpha", -1))
    if abs(alpha - 0.20) > 1e-12:
        raise RuntimeError(f"E9b alpha drift: {alpha}")

    def selective_macro(records, base_key, blend_key):
        return float(np.mean([
            float(records[target][blend_key] if target in selected_targets
                  else records[target][base_key])
            for target in TARGETS
        ]))

    public_base = float(public["base_gold_macro_auc"])
    remote_base = float(remote_oof["e2_macro_auc"])
    public_selective = selective_macro(public_target, "base_auc", "blend_auc")
    remote_selective = selective_macro(remote_target, "e2_auc", "blend_auc")
    if public_selective < public_base + 0.001:
        raise RuntimeError("E9b public selective gate no longer improves E2")
    if remote_selective < remote_base + 0.001:
        raise RuntimeError("E9b remote selective gate no longer improves E2")

    payload = torch.load(heads_path, map_location="cpu", weights_only=True)
    expected_payload = {
        "version": "v52-radimagenet-resnet50-official-1",
        "targets": TARGETS,
        "encoder_sha256": (
            "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
        ),
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "img": 224,
        "slices_per_plane": 8,
        "feature": "global_average_pool",
    }
    for key, expected in expected_payload.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"pinned E9b head contract drift for {key}")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("pinned E9b payload requires five folds")
    if sorted(int(record.get("fold", -1)) for record in folds) != list(range(5)):
        raise RuntimeError("pinned E9b fold identity drift")
    if any(not isinstance(record.get("state_dict"), dict) for record in folds):
        raise RuntimeError("pinned E9b state dictionary is absent")
    if abs(float(payload.get("weak_oof_auc", -1)) - 0.8278261335825697) > 1e-12:
        raise RuntimeError("pinned E9b weak OOF drift")
    if abs(float(payload.get("gold_oof_auc", -1)) - 0.8543239133509962) > 1e-12:
        raise RuntimeError("pinned E9b gold OOF drift")
    return {
        "payload": payload,
        "folds": folds,
        "alpha": alpha,
        "selected_targets": selected_targets,
        "preserved_targets": preserved_targets,
        "public_base": public_base,
        "public_selective": public_selective,
        "remote_base": remote_base,
        "remote_selective": remote_selective,
        "remote_outer_fold_choices": remote_oof["outer_fold_choices"],
    }


def _v52_e10_remote_ladder(contract):
    """Recompute this account's half of the ladder from artifacts the kernel can read.

    The contract carries per-target gains for two independent RadImageNet OOF runs. Only the
    public run is unverifiable here, so its numbers stay data. The remote run is rebuilt from
    the attached OOF table, the pinned E2 OOF bundle and the official labels, and must match
    the contract exactly or E10 refuses to deploy.
    """
    train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    oof_path = find_input_file("v52_oof.csv")
    observed = _v52_sha256(oof_path)
    if observed != contract["remote_oof_sha256"]:
        raise RuntimeError(f"E10 remote OOF drift: {observed}")
    rad_frame = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
    if rad_frame.columns.tolist() != ["StudyInstanceUID", *TARGETS, "fold", "is_gold"]:
        raise RuntimeError("E10 remote OOF schema drift")
    aligned = train[["StudyInstanceUID"]].merge(
        rad_frame, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    if aligned[TARGETS].isna().any().any():
        raise RuntimeError("E10 remote OOF does not cover every official train study")

    base_npz = find_input_file("oof.npz")
    with np.load(base_npz, allow_pickle=False) as bundle:
        if bundle["targets"].astype(str).tolist() != TARGETS:
            raise RuntimeError("E10 E2 OOF target order drift")
        if not np.array_equal(
            bundle["ids"].astype(str), train.StudyInstanceUID.astype(str).to_numpy()
        ):
            raise RuntimeError("E10 E2 OOF study order drift")
        if not np.array_equal(bundle["gold_mask"].astype(bool), gold):
            raise RuntimeError("E10 E2 gold mask differs from official train.csv")
        base_prediction = bundle["pred"].astype(np.float64)

    # Rank within the scored rows, matching both the E9b parent and test-time deployment
    # where the ranked population and the scored population are the same studies.
    base = _v52_rank_columns(base_prediction[gold])
    rad = _v52_rank_columns(aligned[TARGETS].to_numpy(np.float64)[gold])
    gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
    if len(gold_y) != 58 or not np.isfinite(base).all() or not np.isfinite(rad).all():
        raise RuntimeError("E10 gold alignment is incomplete or non-finite")
    reference = _v52_target_auc(gold_y, base)
    if abs(
        float(np.mean([reference[t] for t in TARGETS]))
        - float(contract["base_gold_macro_auc"])
    ) > 1e-9:
        raise RuntimeError("E10 base gold diagnostic drift")

    rebuilt = {}
    for key in contract["ladder"]:
        alpha = float(key)
        scores = _v52_target_auc(gold_y, (1.0 - alpha) * base + alpha * rad)
        rebuilt[key] = {t: scores[t] - reference[t] for t in TARGETS}
    pinned_remote = contract["per_target_ladder_delta"]["remote_v15"]
    if set(rebuilt) != set(pinned_remote):
        raise RuntimeError("E10 ladder key drift")
    for key, deltas in rebuilt.items():
        for target, delta in deltas.items():
            if abs(delta - float(pinned_remote[key][target])) > 1e-9:
                raise RuntimeError(
                    f"E10 recomputed remote gain disagrees at {key}/{target}: {delta}"
                )
    if any(abs(reference[t] - float(contract["per_target_base_auc"][t])) > 1e-9 for t in TARGETS):
        raise RuntimeError("E10 per-target base AUC drift")
    return rebuilt, reference


def _v52_load_e10():
    """Validate the E10 contract, then return the weight map the kernel will deploy."""
    heads_path = find_input_file("v52_radimagenet_heads.pt")
    remote_path = find_input_file("rad_e9_audit.json")
    contract_path = find_input_file("e10_contract.json")
    for path, expected in (
        (heads_path, PINNED_HEADS_SHA256),
        (remote_path, PINNED_REMOTE_AUDIT_SHA256),
        (contract_path, PINNED_E10_CONTRACT_SHA256),
    ):
        observed = _v52_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned E10 artifact drift for {path.name}: {observed}")

    contract = json.loads(contract_path.read_text())
    if contract.get("version") != "e10-alpha-ladder-2":
        raise RuntimeError(f"unexpected E10 contract version {contract.get('version')}")
    if contract.get("targets") != TARGETS:
        raise RuntimeError("E10 contract target order drift")
    remote = json.loads(remote_path.read_text())
    if remote.get("status") != "OOF_REJECTED_E2_PRESERVED":
        raise RuntimeError(f"unexpected v15 audit status {remote.get('status')}")
    if remote.get("primary_sha256") != (
        "f9fb57b7bac8489a5d5285b3984b06df57f142572be6417eac6341c43e96707a"
    ):
        raise RuntimeError("v15 did not preserve the exact E2 visible artifact")
    if int(remote.get("head_count", -1)) != 5:
        raise RuntimeError("v15 remote audit does not contain five heads")

    rebuilt, base_auc = _v52_e10_remote_ladder(contract)
    public_ladder = contract["per_target_ladder_delta"]["public"]
    configuration = contract["configurations"].get(E10_CONFIG)
    if configuration is None:
        raise RuntimeError(f"E10 contract has no configuration {E10_CONFIG!r}")
    alpha_map = {t: float(configuration["alpha_map"][t]) for t in TARGETS}
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alpha_map.values()):
        raise RuntimeError("E10 weight outside the unit interval")
    preserved = sorted(t for t, alpha in alpha_map.items() if alpha == 0.0)
    if preserved != sorted(E10_PRESERVED_TARGETS):
        raise RuntimeError(f"E10 preserved-target drift: {preserved}")

    # Two-tier gate. The scored objective is macro AUC, so the binding requirement is that
    # the deployed map raise the macro in BOTH independent runs -- the public numbers as
    # pinned data, this account's numbers as recomputed above. A per-target "never harm any
    # single label" rule is strictly stronger than that objective and would veto rungs that
    # trade a small loss on one label for a large gain on another, so it is enforced only for
    # configurations that actually claim it. Whichever claim the contract makes is verified;
    # a configuration cannot quietly assert dual-positivity it no longer has.
    claims_dual_positive = bool(configuration["all_dual_positive"])
    observed_dual_positive = True
    for target, alpha in alpha_map.items():
        if alpha == 0.0:
            continue
        key = f"{alpha:.2f}"
        if key not in rebuilt:
            raise RuntimeError(f"E10 weight {key} is outside the audited ladder")
        gains = (float(public_ladder[key][target]), float(rebuilt[key][target]))
        if not all(gain > 0 for gain in gains):
            observed_dual_positive = False
            if claims_dual_positive:
                raise RuntimeError(
                    f"E10 dual-source gate rejects {target} at {key}: {gains}"
                )
    if observed_dual_positive != claims_dual_positive:
        raise RuntimeError(
            f"E10 contract claims all_dual_positive={claims_dual_positive} for "
            f"{E10_CONFIG!r} but recomputation observes {observed_dual_positive}"
        )

    macro = {}
    for name, ladder in (("public", public_ladder), ("remote_v15", rebuilt)):
        total = 0.0
        for target, alpha in alpha_map.items():
            gain = 0.0 if alpha == 0.0 else float(ladder[f"{alpha:.2f}"][target])
            total += float(base_auc[target]) + gain
        macro[name] = total / len(TARGETS)
        if macro[name] <= float(contract["base_gold_macro_auc"]):
            raise RuntimeError(
                f"E10 macro gate rejects {E10_CONFIG!r}: {name} macro {macro[name]} "
                f"does not beat base {contract['base_gold_macro_auc']}"
            )
        if abs(macro[name] - float(configuration["descriptive_macro"][name])) > 1e-9:
            raise RuntimeError(
                f"E10 recomputed {name} macro {macro[name]} disagrees with the contract "
                f"value {configuration['descriptive_macro'][name]}"
            )

    payload = torch.load(heads_path, map_location="cpu", weights_only=True)
    expected_payload = {
        "version": "v52-radimagenet-resnet50-official-1",
        "targets": TARGETS,
        "encoder_sha256": (
            "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
        ),
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "img": 224,
        "slices_per_plane": 8,
        "feature": "global_average_pool",
    }
    for key, expected in expected_payload.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"pinned E10 head contract drift for {key}")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("pinned E10 payload requires five folds")
    if sorted(int(record.get("fold", -1)) for record in folds) != list(range(5)):
        raise RuntimeError("pinned E10 fold identity drift")
    if any(not isinstance(record.get("state_dict"), dict) for record in folds):
        raise RuntimeError("pinned E10 state dictionary is absent")
    if abs(float(payload.get("gold_oof_auc", -1)) - 0.8543239133509962) > 1e-12:
        raise RuntimeError("pinned E10 gold OOF drift")
    return {
        "payload": payload,
        "folds": folds,
        "contract": contract,
        "configuration": E10_CONFIG,
        "alpha_map": alpha_map,
        "preserved_targets": sorted(E10_PRESERVED_TARGETS),
        "diagnostic_macro": configuration["diagnostic_macro"],
        "recomputed_macro": macro,
        "all_dual_positive": claims_dual_positive,
        "rationale": configuration["rationale"],
        "recomputed_remote_ladder": rebuilt,
    }


def main_v52_pinned_e9b():
    """Inference-only E9b from hash-pinned v15 heads; preserve E2 on any failure."""
    import shutil

    output = Path("/kaggle/working/rsna_rad_e9b")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e9b_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "mode": "pinned_v15_heads_inference_only",
        "evidence_boundary": (
            "OOF values are diagnostics, not Kaggle scores. The fixed public 20-percent "
            "vote is applied only to targets improving in two independent OOF runs."
        ),
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0",
        "encoder_sha256": (
            "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
        ),
        "heads_sha256": PINNED_HEADS_SHA256,
        "parent": "E2 captured 20-member DINOv2 rank ensemble",
        "pixel_rules": dict(RULES),
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("E9b RadImageNet inference requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = 8.72 * 3600 - elapsed
        audit["elapsed_before_e9b_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 45 * 60:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        pinned = _v52_load_pinned_e9b()
        audit["oof_gate"] = {
            "alpha": pinned["alpha"],
            "selected_targets": pinned["selected_targets"],
            "preserved_targets": pinned["preserved_targets"],
            "public_e2_macro_auc": pinned["public_base"],
            "public_selective_macro_auc": pinned["public_selective"],
            "remote_e2_macro_auc": pinned["remote_base"],
            "remote_selective_macro_auc": pinned["remote_selective"],
            "remote_outer_fold_choices": pinned["remote_outer_fold_choices"],
            "selection_supported": True,
        }

        test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
        test_series = pd.read_csv(
            ROOT / "test_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
        headers = annotate(walk("test_series"))
        audit_official_sequence_metadata(headers, test_series)
        studies, pixels, slot_mask = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, "test-e9b "), "test-e9b"
        )
        by_uid = {str(uid): index for index, uid in enumerate(studies)}
        missing = [uid for uid in test.StudyInstanceUID if uid not in by_uid]
        if missing:
            raise RuntimeError(f"{len(missing)} test studies absent from E9b cache")
        order = np.asarray([by_uid[uid] for uid in test.StudyInstanceUID], dtype=np.int64)
        pixels, slot_mask = pixels[order], slot_mask[order]
        token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
        if token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient acquired E9b test slices: {token_count}")
        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()

        all_test = np.arange(len(test), dtype=np.int64)
        fold_predictions = []
        for record in pinned["folds"]:
            head = FoundationQueryHead().to(device)
            head.load_state_dict(record["state_dict"], strict=True)
            fold_predictions.append(
                predict_head(head, features, token_mask, all_test, device)
            )
            del head
            torch.cuda.empty_cache()
        if len(fold_predictions) != 5:
            raise RuntimeError("E9b test inference did not use all five heads")
        rad_test = np.mean(np.stack(fold_predictions), axis=0)
        if rad_test.shape != (len(test), len(TARGETS)) or not np.isfinite(rad_test).all():
            raise RuntimeError(f"invalid E9b prediction shape/value: {rad_test.shape}")

        baseline = pd.read_csv(preserved, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(baseline, test.StudyInstanceUID)
        rad_frame = pd.DataFrame(rad_test, columns=TARGETS)
        rad_frame.insert(0, "StudyInstanceUID", test.StudyInstanceUID)
        _v52_validate_submission(rad_frame, test.StudyInstanceUID)
        rad_frame.to_csv(output / "submission_rad_only.csv", index=False)
        baseline_rank = _v52_rank_columns(baseline[TARGETS].to_numpy())
        rad_rank = _v52_rank_columns(rad_test)
        selected = baseline.copy()
        alpha = pinned["alpha"]
        for target in pinned["selected_targets"]:
            index = TARGETS.index(target)
            selected[target] = (
                (1.0 - alpha) * baseline_rank[:, index] + alpha * rad_rank[:, index]
            )
        for target in pinned["preserved_targets"]:
            if not np.array_equal(
                selected[target].to_numpy(), baseline[target].to_numpy()
            ):
                raise RuntimeError(f"E9b failed to preserve {target}")
        _v52_validate_submission(selected, test.StudyInstanceUID)
        selected_path = output / "submission_e2_rad_robust_200.csv"
        selected.to_csv(selected_path, index=False)
        selected = pd.read_csv(selected_path, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(selected, test.StudyInstanceUID)

        audit.update({
            "test_studies": len(test),
            "test_available_slice_tokens": token_count,
            "test_head_count": len(fold_predictions),
            "selected_path": str(selected_path),
            "selected_sha256": _v52_sha256(selected_path),
            "fallback_sha256": _v52_sha256(preserved),
        })
        shutil.copy2(selected_path, primary)
        if _v52_sha256(primary) != audit["selected_sha256"]:
            raise RuntimeError("primary E9b copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        log(
            f"E9b selected alpha={alpha:.3f} on "
            f"{len(pinned['selected_targets'])} dual-OOF-stable targets; "
            "Baker's and Fracture preserve E2"
        )
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E9b preserves E2: {audit['error']}")
    finally:
        if audit.get("status") != "CANDIDATE_SELECTED" and preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _v52_rad_test_predictions(pinned, test, device, tag):
    """Five-head RadImageNet test prediction on the official test tree."""
    test_series = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    headers = annotate(walk("test_series"))
    audit_official_sequence_metadata(headers, test_series)
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane, lat_of(headers, f"{tag} "), tag
    )
    by_uid = {str(uid): index for index, uid in enumerate(studies)}
    missing = [uid for uid in test.StudyInstanceUID if uid not in by_uid]
    if missing:
        raise RuntimeError(f"{len(missing)} test studies absent from {tag} cache")
    order = np.asarray([by_uid[uid] for uid in test.StudyInstanceUID], dtype=np.int64)
    pixels, slot_mask = pixels[order], slot_mask[order]
    token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
    if token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f"insufficient acquired {tag} test slices: {token_count}")
    features, token_mask = encode_radimagenet(pixels, slot_mask, device)
    del pixels, slot_mask, headers
    gc.collect()

    rows = np.arange(len(test), dtype=np.int64)
    predictions = []
    for record in pinned["folds"]:
        head = FoundationQueryHead().to(device)
        head.load_state_dict(record["state_dict"], strict=True)
        predictions.append(predict_head(head, features, token_mask, rows, device))
        del head
        torch.cuda.empty_cache()
    if len(predictions) != 5:
        raise RuntimeError(f"{tag} test inference did not use all five heads")
    rad_test = np.mean(np.stack(predictions), axis=0)
    if rad_test.shape != (len(test), len(TARGETS)) or not np.isfinite(rad_test).all():
        raise RuntimeError(f"invalid {tag} prediction shape/value: {rad_test.shape}")
    return rad_test, token_count, len(predictions)


def main_v52_e10():
    """Deploy the audited E10 weight map; preserve the E2 parent on any failure."""
    import shutil

    output = Path("/kaggle/working/rsna_rad_e10")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e10_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "mode": "pinned_v15_heads_inference_only",
        "experiment": "E10",
        "configuration": E10_CONFIG,
        "evidence_boundary": (
            "OOF values are 58-study diagnostics on official train labels, not Kaggle "
            "scores. E10 widens the weight ladder that E9 truncated at 0.25 and votes only "
            "where two independent OOF runs agree at the deployed weight."
        ),
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0",
        "heads_sha256": PINNED_HEADS_SHA256,
        "contract_sha256": PINNED_E10_CONTRACT_SHA256,
        "parent": "E2 captured 20-member DINOv2 rank ensemble",
        "pixel_rules": dict(RULES),
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("E10 RadImageNet inference requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = 8.72 * 3600 - elapsed
        audit["elapsed_before_e10_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 45 * 60:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        pinned = _v52_load_e10()
        audit["weight_gate"] = {
            "configuration": pinned["configuration"],
            "alpha_map": pinned["alpha_map"],
            "preserved_targets": pinned["preserved_targets"],
            "diagnostic_macro": pinned["diagnostic_macro"],
            "recomputed_macro": pinned["recomputed_macro"],
            "all_dual_positive": pinned["all_dual_positive"],
            "base_gold_macro_auc": pinned["contract"]["base_gold_macro_auc"],
            "rationale": pinned["rationale"],
            "remote_ladder_recomputed_in_kernel": True,
        }

        test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
        rad_test, token_count, head_count = _v52_rad_test_predictions(
            pinned, test, device, "test-e10"
        )

        baseline = pd.read_csv(preserved, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(baseline, test.StudyInstanceUID)
        rad_frame = pd.DataFrame(rad_test, columns=TARGETS)
        rad_frame.insert(0, "StudyInstanceUID", test.StudyInstanceUID)
        _v52_validate_submission(rad_frame, test.StudyInstanceUID)
        rad_frame.to_csv(output / "submission_rad_only.csv", index=False)
        baseline_rank = _v52_rank_columns(baseline[TARGETS].to_numpy())
        rad_rank = _v52_rank_columns(rad_test)

        # Materialise every audited rung so the ladder is inspectable from one run; only the
        # configured map is promoted to the visible submission.
        written = {}
        for name, configuration in sorted(pinned["contract"]["configurations"].items()):
            frame = baseline.copy()
            for target, alpha in configuration["alpha_map"].items():
                alpha = float(alpha)
                if alpha > 0:
                    index = TARGETS.index(target)
                    frame[target] = (
                        (1.0 - alpha) * baseline_rank[:, index] + alpha * rad_rank[:, index]
                    )
            for target, alpha in configuration["alpha_map"].items():
                if float(alpha) == 0.0 and not np.array_equal(
                    frame[target].to_numpy(), baseline[target].to_numpy()
                ):
                    raise RuntimeError(f"E10 failed to preserve {target} in {name}")
            _v52_validate_submission(frame, test.StudyInstanceUID)
            path = output / f"submission_e10_{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = _v52_sha256(path)
        audit["ladder_sha256"] = written

        selected_path = output / f"submission_e10_{pinned['configuration']}.csv"
        selected = pd.read_csv(selected_path, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(selected, test.StudyInstanceUID)
        for target in pinned["preserved_targets"]:
            if not np.array_equal(
                selected[target].to_numpy(), baseline[target].to_numpy()
            ):
                raise RuntimeError(f"E10 promoted file does not preserve {target}")
        audit.update({
            "test_studies": len(test),
            "test_available_slice_tokens": token_count,
            "test_head_count": head_count,
            "selected_path": str(selected_path),
            "selected_sha256": _v52_sha256(selected_path),
            "fallback_sha256": _v52_sha256(preserved),
        })
        shutil.copy2(selected_path, primary)
        if _v52_sha256(primary) != audit["selected_sha256"]:
            raise RuntimeError("primary E10 copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        voted = sorted(t for t, alpha in pinned["alpha_map"].items() if alpha > 0)
        log(
            f"E10 promoted {pinned['configuration']} over {len(voted)} dual-OOF targets; "
            f"{', '.join(pinned['preserved_targets'])} preserve E2"
        )
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E10 preserves E2: {audit['error']}")
    finally:
        if audit.get("status") != "CANDIDATE_SELECTED" and preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _v52_e11_availability(headers, plane_map):
    """Count studies offering each (plane, fat-suppression) pair before any slot is picked.

    The parent arm reads only fat-suppressed series and never had to ask how many studies
    carry a non-suppressed one. E11 depends on that answer, so it is measured and logged
    rather than assumed: a slot nobody can fill is a masked column, and a run that produced
    one silently would look like a weak arm instead of an absent input.
    """
    frame = headers[["StudyInstanceUID", "SeriesInstanceUID", "fatsat"]].copy()
    frame["plane"] = frame.SeriesInstanceUID.map(plane_map)
    total = frame.StudyInstanceUID.nunique()
    table = {}
    for plane in ("Sagittal", "Coronal", "Axial"):
        for fatsat in (True, False):
            selected = frame[(frame.plane == plane) & (frame.fatsat == bool(fatsat))]
            studies = selected.StudyInstanceUID.nunique()
            key = f"{plane}_{'FS' if fatsat else 'NOFS'}"
            table[key] = {
                "studies": int(studies),
                "fraction": float(studies / total) if total else 0.0,
                "series": int(len(selected)),
            }
            log(f"E11 availability {key}: {studies}/{total} studies, {len(selected)} series")
    return table


def main_v52_e11():
    """Train a third arm on a deliberately different pixel recipe. Never ships a candidate."""
    import shutil
    from sklearn.model_selection import GroupKFold

    output = Path("/kaggle/working/rsna_rad_e11")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e11_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "mode": "e11-diverse-recipe-training-only",
        "evidence_boundary": (
            "Every value here is a local out-of-fold diagnostic on 58 official image "
            "labels. It is not a Kaggle score, and this mode never replaces the parent "
            "submission under any outcome."
        ),
        "recipe": {
            "slots": [list(slot) for slot in E11_SLOTS],
            "crop_mm": E11_CROP_MM,
            "cache_slices": E11_CACHE_SLICES,
            "img": E11_IMG,
            "differs_from_parent_arm": (
                "parent reads 3 fat-suppressed slots at full frame; this reads 3 "
                "non-suppressed slots plus 1 suppressed anchor at a 130 mm physical crop"
            ),
        },
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0 (Kaggle-hosted weight metadata)",
        "encoder_sha256": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("E11 training requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = TIME_BUDGET - elapsed
        audit["elapsed_before_e11_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 2.5 * 3600:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        # Point the shared pixel path at the E11 recipe. These are the same process globals
        # the parent notebook's readers consult, so the override has to happen before any
        # slot is picked or any pixel is decoded, and nothing after this point may assume
        # the parent arm's values.
        globals().update(
            SLOTS=list(E11_SLOTS),
            N_SLOT=len(E11_SLOTS),
            CACHE_SLICES=int(E11_CACHE_SLICES),
            IMG=int(E11_IMG),
            CACHE_IMG=int(E11_IMG),
            CROP_MM=float(E11_CROP_MM),
        )
        log(
            f"E11 recipe: {[s[0] for s in E11_SLOTS]} at {E11_CROP_MM:.0f} mm, "
            f"{E11_IMG} px, {E11_CACHE_SLICES} slices/slot"
        )

        train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
        train_series = pd.read_csv(
            ROOT / "train_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        if len(train) != 4407:
            raise RuntimeError(f"unexpected train study count {len(train)}")
        plane = dict(zip(train_series.SeriesInstanceUID, train_series.Anatomical_Plane))
        headers = annotate(walk("train_series"))
        audit["availability"] = _v52_e11_availability(headers, plane)

        studies, pixels, slot_mask = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, "train-e11 "), "train-e11"
        )
        by_uid = {str(uid): i for i, uid in enumerate(studies)}
        missing = [uid for uid in train.StudyInstanceUID if uid not in by_uid]
        if missing:
            raise RuntimeError(f"{len(missing)} train studies absent from cache")
        order = np.array([by_uid[uid] for uid in train.StudyInstanceUID], dtype=np.int64)
        pixels, slot_mask = pixels[order], slot_mask[order]
        fill = float(slot_mask.mean())
        per_slot = {
            name: float(slot_mask[:, k].mean())
            for k, (name, _, _, _) in enumerate(E11_SLOTS)
        }
        audit["slot_fill"] = per_slot
        audit["overall_fill"] = fill
        for name, value in per_slot.items():
            log(f"E11 slot fill {name}: {value:.1%}")
        if fill < E11_MIN_FILL:
            raise RuntimeError(f"E11 slot fill {fill:.1%} below the {E11_MIN_FILL:.0%} floor")

        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()

        y, weights, gold = make_targets(train)
        if int(gold.sum()) != 58:
            raise RuntimeError(f"expected 58 fully gold studies, observed {int(gold.sum())}")
        groups = report_groups(train)
        if len(np.unique(groups)) < 4000:
            raise RuntimeError("unexpected report-group collapse")

        splits = list(GroupKFold(5).split(features, groups=groups))
        fold_id = np.full(len(train), -1, dtype=np.int8)
        folds = []
        oof = np.zeros_like(y, dtype=np.float32)
        for fold, (tr, va) in enumerate(splits):
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"report leakage in fold {fold}")
            fold_id[va] = fold
            state, score = train_fold(
                features, token_mask, y, weights, tr, va, fold, device
            )
            if state is None:
                raise RuntimeError(f"fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            oof[va] = predict_head(head, features, token_mask, va, device)
            folds.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
            del head
            torch.cuda.empty_cache()
        if (fold_id < 0).any() or not np.isfinite(oof).all():
            raise RuntimeError("incomplete E11 OOF")

        weak_auc = macro_auc(y, oof)
        gold_auc = macro_auc(y[gold], oof[gold])
        audit["weak_oof_auc"] = float(weak_auc)
        audit["gold_oof_auc"] = float(gold_auc)
        log(f"E11 OOF weak macro AUC {weak_auc:.5f}")
        log(f"E11 OOF gold macro AUC {gold_auc:.5f} on 58 studies")

        # The question E11 exists to answer is not whether this arm is strong on its own but
        # whether it says something the portfolio does not already know. Both halves are
        # measured against the same 58 rows and the same rank basis the deployed blend uses.
        base_npz = find_input_file("oof.npz")
        with np.load(base_npz, allow_pickle=False) as bundle:
            if bundle["targets"].astype(str).tolist() != TARGETS:
                raise RuntimeError("E11 E2 OOF target order drift")
            if not np.array_equal(
                bundle["ids"].astype(str), train.StudyInstanceUID.astype(str).to_numpy()
            ):
                raise RuntimeError("E11 E2 OOF study order drift")
            base_prediction = bundle["pred"].astype(np.float64)
        base = _v52_rank_columns(base_prediction[gold])
        new = _v52_rank_columns(oof[gold].astype(np.float64))
        gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
        reference = _v52_target_auc(gold_y, base)
        audit["e2_base_gold_macro"] = float(np.mean([reference[t] for t in TARGETS]))
        ladder = {}
        for alpha in (0.20, 0.35, 0.50):
            scores = _v52_target_auc(gold_y, (1.0 - alpha) * base + alpha * new)
            ladder[f"{alpha:.2f}"] = {
                "macro": float(np.mean([scores[t] for t in TARGETS])),
                "per_target_delta": {t: float(scores[t] - reference[t]) for t in TARGETS},
            }
            log(f"E11 blend alpha={alpha:.2f} gold macro {ladder[f'{alpha:.2f}']['macro']:.5f}")
        audit["blend_vs_e2"] = ladder

        try:
            parent_oof = pd.read_csv(
                find_input_file("v52_oof.csv"), dtype={"StudyInstanceUID": str}
            )
            aligned = train[["StudyInstanceUID"]].merge(
                parent_oof, on="StudyInstanceUID", how="left", validate="one_to_one"
            )
            parent = _v52_rank_columns(aligned[TARGETS].to_numpy(np.float64)[gold])
            audit["correlation_with_parent_arm"] = {
                t: float(np.corrcoef(parent[:, i], new[:, i])[0, 1])
                for i, t in enumerate(TARGETS)
            }
            log(
                "E11 mean rank correlation with the parent arm: "
                f"{np.mean(list(audit['correlation_with_parent_arm'].values())):.3f}"
            )
        except FileNotFoundError:
            audit["correlation_with_parent_arm"] = None

        torch.save(
            {
                "version": "e11-radimagenet-resnet50-diverse-1",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "slots": [list(slot) for slot in E11_SLOTS],
                "crop_mm": E11_CROP_MM,
                "img": E11_IMG,
                "slices_per_plane": E11_CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "weak_oof_auc": float(weak_auc),
                "gold_oof_auc": float(gold_auc),
            },
            output / "v52_e11_heads.pt",
        )
        oof_frame = pd.DataFrame(oof, columns=TARGETS)
        oof_frame.insert(0, "StudyInstanceUID", train.StudyInstanceUID)
        oof_frame["fold"] = fold_id
        oof_frame["is_gold"] = gold.astype(np.uint8)
        oof_frame.to_csv(output / "v52_e11_oof.csv", index=False)
        audit["status"] = "E11_TRAINED_E2_PRESERVED"
        audit["heads_sha256"] = _v52_sha256(output / "v52_e11_heads.pt")
        audit["oof_sha256"] = _v52_sha256(output / "v52_e11_oof.csv")
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E11 preserves E2: {audit['error']}")
    finally:
        if preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if ARM_MODE == "e11":
    main_v52_e11()
else:
    try:
        find_input_file("v52_radimagenet_heads.pt")
    except FileNotFoundError:
        main_v52()
    else:
        try:
            find_input_file("e10_contract.json")
        except FileNotFoundError:
            main_v52_pinned_e9b()
        else:
            main_v52_e10()

