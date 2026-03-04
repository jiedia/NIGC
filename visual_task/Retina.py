import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time

# ========= 1. Build multi-scale DoG filters (ON/OFF) =========
def make_dog_kernel(ksize, sigma_center, sigma_surround):
    """Generate a 2D DoG kernel: G_center - G_surround"""
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    center = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_center ** 2))
    surround = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_surround ** 2))

    center /= center.sum()
    surround /= surround.sum()

    dog = center - surround
    return dog.astype(np.float32)

def build_retina_filters(scales=(1.5, 3.0, 6.0), surround_ratio=1.6):
    """
    Returns a list; each element is a dict:
    {
        "name": "scaleX_ON" / "scaleX_OFF",
        "kernel": 2D convolution kernel (np.float32)
    }
    """
    filters = []
    for s in scales:
        ksize = int(6 * s + 1)  # window of ~3 sigma
        dog = make_dog_kernel(ksize, s, s * surround_ratio)
        filters.append({"name": f"scale{s:.1f}_ON", "kernel": dog})
        filters.append({"name": f"scale{s:.1f}_OFF", "kernel": -dog})
    return filters

# ========= 2. Compute optical flow (DSGC directional channels) =========
def compute_dsgc_directional_channels(prev_frame, curr_frame, num_directions=8):
    """
    Compute direction-selective ganglion cell (DSGC) responses; returns multiple directional channels.
    Optical flow gives per-pixel motion direction; bins are used to form directional channels.
    """
    # Convert to grayscale (input assumed BGR)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Ensure image dimensions match
    if prev_gray.shape != curr_gray.shape:
        raise ValueError(f"Input image dimensions do not match: {prev_gray.shape} vs {curr_gray.shape}")

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) 
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # magnitude and angle of flow

    # Assign each pixel to a directional channel by angle
    # cv2.cartToPolar returns angle in [0, 2*pi); cover full range
    directional_channels = np.zeros((magnitude.shape[0], magnitude.shape[1], num_directions), dtype=np.float32)

    for i in range(num_directions):
        # Divide [0, 2*pi) into num_directions bins
        angle_start = i * 2 * np.pi / num_directions
        angle_end = (i + 1) * 2 * np.pi / num_directions
        # Last bin includes 2*pi
        if i == num_directions - 1:
            direction_mask = (angle >= angle_start) & (angle <= angle_end)
        else:
            direction_mask = (angle >= angle_start) & (angle < angle_end)
        directional_channels[..., i] = np.where(direction_mask, magnitude, 0)

    return directional_channels

# ========= 3. Compute looming feature =========
def compute_looming_feature(prev_frame, curr_frame):
    """
    Compute looming feature (object approaching/receding).
    Returns normalized area ratio (0~1), then sqrt-compressed.
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Frame difference to get change region
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours of change region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Take largest contour as main object
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = float(w * h)
    else:
        area = 0.0  # no significant motion -> area 0

    frame_h, frame_w = prev_gray.shape 
    frame_area = float(frame_h * frame_w) if frame_h > 0 and frame_w > 0 else 1.0

    # Normalize to [0,1]: fraction of frame area
    area_norm = area / frame_area  # in [0, 1]
    looming_value = np.sqrt(area_norm)

    return looming_value

# ========= 4. Process video and extract features =========
def video_to_retina_features(
    video_path,
    out_path,
    target_size=(32, 32),   # retina spatial resolution
    frame_step=2,           # subsample frames to reduce T
    scales=(1.5, 3.0, 6.0),
    dtype=np.float16
):
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build filters
    retina_filters = build_retina_filters(scales=scales)
    n_filters = len(retina_filters)

    # 2) Load video and preprocess
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    prev_frame = None
    curr_frame = None
    directional_channels_all = []
    looming_features_all = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # BGR -> Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to target retina resolution
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0

        frames.append(gray)
        
        # Temporal features need two frames; compute from second frame onward
        if prev_frame is not None and curr_frame is not None:
            # Directional channels (DSGC) on full-resolution frames
            directional_channels = compute_dsgc_directional_channels(prev_frame, curr_frame)
            # Resize directional channels to target size for downstream
            directional_channels_resized = np.zeros((target_size[1], target_size[0], directional_channels.shape[2]), dtype=np.float32)
            for d in range(directional_channels.shape[2]):
                directional_channels_resized[:, :, d] = cv2.resize(
                    directional_channels[:, :, d], 
                    target_size, 
                    interpolation=cv2.INTER_AREA
                )
            directional_channels_all.append(directional_channels_resized)

            # Looming feature (approach/recede)
            looming_feature = compute_looming_feature(prev_frame, curr_frame)
            looming_features_all.append(looming_feature)
        else:
            # First frame: pad with zeros for alignment
            directional_channels_all.append(np.zeros((target_size[1], target_size[0], 8), dtype=np.float32))
            looming_features_all.append(0.0)

        prev_frame = curr_frame
        curr_frame = frame
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        print(f"[WARN] No valid frames read from {video_path}, skipping.")
        return None

    frames = np.stack(frames, axis=0)  # [T, H, W]
    T, H, W = frames.shape

    # 3) Spatial DoG filtering: ON/OFF multi-scale responses
    spatial_resp = np.zeros((T, H, W, n_filters), dtype=np.float32)

    for k, f in enumerate(retina_filters):
        kernel = f["kernel"]
        for t in range(T):
            spatial_resp[t, :, :, k] = cv2.filter2D(frames[t], -1, kernel)

    # 4) Simple temporal high-pass: current - previous frame (transient/motion-sensitive)
    temporal_resp = np.zeros_like(spatial_resp)
    temporal_resp[0] = spatial_resp[0]           # first frame as sustained
    temporal_resp[1:] = spatial_resp[1:] - spatial_resp[:-1]

    # 5) Nonlinearity: ReLU then tanh (avoid extremes)
    retinal = np.maximum(temporal_resp, 0.0)      # half-wave rectification
    retinal = np.tanh(retinal)                   # mild compression

    # 6) Cast to low precision to save space
    retinal = retinal.astype(dtype)

    # 7) Flatten spatial dims -> (T, N_input)
    T, H, W, C = retinal.shape
    retinal_flat = retinal.reshape(T, H * W * C)  # input to reservoir downstream

    # 8) Append DSGC directional channels
    if directional_channels_all and len(directional_channels_all) == T:
        # Ensure data present and length matches frame count
        directional_flat = np.stack(directional_channels_all, axis=0)  # [T, H, W, num_directions]
        num_directions = directional_flat.shape[3]
        # Flatten spatial dimensions
        directional_flat = directional_flat.reshape(T, H * W * num_directions)
    else:
        # If empty or length mismatch, create zero array
        num_directions = 8  # default 8 directions
        print(f"[WARN] Directional channel data anomaly: expected length {T}, actual {len(directional_channels_all) if directional_channels_all else 0}")
        directional_flat = np.zeros((T, H * W * num_directions), dtype=np.float32)

    # 9) Append looming feature
    if len(looming_features_all) == T:
        looming_flat = np.array(looming_features_all, dtype=np.float32)
    else:
        print(f"[WARN] Looming feature length anomaly: expected {T}, actual {len(looming_features_all)}")
        looming_flat = np.zeros(T, dtype=np.float32)

    # 10) Save as .npz (compressed)
    np.savez_compressed(
        out_path,
        features=retinal_flat,
        directional_features=directional_flat,
        looming_features=looming_flat,
        T=T, H=H, W=W, C=C,
        frame_step=frame_step,
    )

    return retinal_flat, directional_flat, looming_flat

# ========= 5. Batch process one action-class folder =========
def process_dataset(root_dir, out_root, frame_step=2):
    """
    root_dir: path containing subfolders close_to / observation / stay_away
    out_root: feature output root directory
    """
    root_dir = Path(root_dir)
    out_root = Path(out_root)

    classes = ["close_to", "observation", "stay_away"]

    video_count = 1  # for output file naming

    for cls in classes:
        cls_dir = root_dir / cls
        out_dir = out_root / cls  # preserve subfolder structure
        out_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(cls_dir.glob("*.mp4"))

        # Progress bar for current subfolder
        with tqdm(video_files, desc=f"Processing {cls}", unit="video") as video_progress:
            for v in video_progress:
                # Use numeric index for output filenames
                out_path = out_dir / f"{video_count:03d}_retina.npz"
                # if out_path.exists():
                #     continue
                video_to_retina_features(
                    video_path=v,
                    out_path=out_path,
                    frame_step=frame_step
                )
                video_count += 1

base_dir = Path(__file__).resolve().parent
process_dataset(
    root_dir=base_dir / "video_data" / "video classification_raw_data_mp4",
    out_root=base_dir / "video_data" / "feature",
    frame_step=3
)