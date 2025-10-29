
def select_target_bucket(
    original_size: tuple[int, int],
    buckets: list[tuple[int, int]],
    k_nearest: int = 3,
    sharpness: float = 1.0,
) -> tuple[int, int]:
    """
    Selects a target aspect ratio bucket for an entire batch based on a
    representative image's dimensions.
    """
    original_w, original_h = original_size
    original_ar = original_w / original_h

    # --- Calculate distance to each bucket's aspect ratio ---
    bucket_distances = []
    for bucket_w, bucket_h in buckets:
        bucket_ar = bucket_w / bucket_h
        distance = abs(math.log(original_ar) - math.log(bucket_ar))
        bucket_distances.append((distance, (bucket_w, bucket_h)))

    # --- Probabilistically sample a target bucket ---
    bucket_distances.sort(key=lambda x: x[0])
    nearest_buckets = bucket_distances[:k_nearest]
    distances = np.array([-d for d, _ in nearest_buckets])
    probabilities = np.exp((distances - np.max(distances)) * sharpness)
    probabilities /= probabilities.sum()
    chosen_idx = np.random.choice(len(nearest_buckets), p=probabilities)
    _, target_size = nearest_buckets[chosen_idx]
    
    return target_size

def determine_transform_params(
    original_size: tuple[int, int],
    target_size: tuple[int, int], # This is now a fixed input
    prescale_perc: float = 0.8, #
) -> tuple[tuple, tuple, tuple]:
    """
    Determines the full transformation parameters for a single tuple, respecting
    the locked batch target size. It correctly prescales to circumscribe the target.
    
    Returns:
        prescale_size (tuple or None)
        crop_box (tuple)
        reported_original_size (tuple)
    """
    original_w, original_h = original_size
    target_w, target_h = target_size

    
    # --- NEW: Safeguard for small images ---
    is_too_small = original_w < target_w or original_h < target_h
    
    if is_too_small or (random.random() < prescale_perc):
        # --- MODE A: Prescale-then-Crop (Global View) ---
        # 1. Correctly determine the scale factor to make the image *larger* than the target crop.
        #    The goal is to make the smaller dimension of the prescaled image match the target.
        scale_factor = max(target_w / original_w, target_h / original_h)
        prescale_w = int(math.ceil(original_w * scale_factor))
        prescale_h = int(math.ceil(original_h * scale_factor))
        prescale_size = (prescale_w, prescale_h)
        
        # 2. Jitter the crop window within this new, larger prescaled image.
        max_left = prescale_w - target_w
        max_top = prescale_h - target_h
        crop_left = random.randint(0, max_left)
        crop_top = random.randint(0, max_top)
        
        crop_box = (crop_left, crop_top, crop_left + target_w, crop_top + target_h)
        
        # 3. Report the prescaled size to the model.
        reported_original_size = prescale_size
        
    else:
        # --- MODE B: Crop-from-Original (Detail View) ---
        # 1. No prescaling is done.
        prescale_size = None
        
        # 2. Determine the crop window that has the target aspect ratio.
        target_ar = target_w / target_h
        original_ar = original_w / original_h
        
        if original_ar > target_ar: # Original is wider than target AR
            crop_h = original_h
            crop_w = int(target_ar * crop_h)
        else: # Original is taller or same
            crop_w = original_w
            crop_h = int(crop_w / target_ar)

        # 3. Jitter the crop window within the original image.
        max_left = original_w - crop_w
        max_top = original_h - crop_h
        crop_left = random.randint(0, max_left)
        crop_top = random.randint(0, max_top)
        
        crop_box = (crop_left, crop_top, crop_left + crop_w, crop_top + crop_h)
        
        # 4. Report the true original size to the model.
        reported_original_size = original_size
        
    return prescale_size, crop_box, reported_original_size

def generate_ar_buckets(target_area: int, step: int = 64, max_ratio: float = 4.0):
    """
    Generates a list of (width, height) buckets with a target area, where dimensions
    are multiples of a given step.
    """
    buckets = set()
    min_dim = int(math.sqrt(target_area / max_ratio) // step) * step
    max_dim = int(math.sqrt(target_area * max_ratio) // step) * step

    for w in range(min_dim, max_dim + step, step):
        if w == 0: continue
        # Calculate height that gets closest to the target area
        h = int(round(target_area / w / step)) * step
        if h == 0: continue
        buckets.add((w, h))
        
    # Also consider swapped aspect ratios
    for h in range(min_dim, max_dim + step, step):
        if h == 0: continue
        w = int(round(target_area / h / step)) * step
        if w == 0: continue
        buckets.add((w, h))
        
    return sorted(list(buckets))

#hopefully deprecated
def get_transform_params(
    original_size: tuple[int, int],
    buckets: list[tuple[int, int]],
    k_nearest: int = 3,
    sharpness: float = 1.0,
    prescale_perc: float = 0.2, # The new hyperparameter for global (downscaled) views vs superresolution cropped views of data
):
    """
    Determines a full set of transformation parameters for an image, choosing between
    a "global view" (prescale-then-crop) and a "detail view" (crop-from-original).

    Returns:
        prescale_size (tuple or None): The size for the initial resize.
        crop_box (tuple): The box to crop from the (potentially prescaled) image.
        target_size (tuple): The final resolution to resize to.
        reported_original_size (tuple): The "original size" to report to add_time_ids.
    """
    original_w, original_h = original_size
    original_ar = original_w / original_h

    # Calculate distance to each bucket's aspect ratio
    bucket_distances = []
    for bucket_w, bucket_h in buckets:
        bucket_ar = bucket_w / bucket_h
        # Use log-space difference for better perceptual distance
        distance = abs(math.log(original_ar) - math.log(bucket_ar))
        bucket_distances.append((distance, (bucket_w, bucket_h)))

    # Select the k-nearest buckets
    bucket_distances.sort(key=lambda x: x[0])
    nearest_buckets = bucket_distances[:k_nearest]

    # Create a probability distribution using softmax
    distances = np.array([-d for d, _ in nearest_buckets])
    probabilities = np.exp((distances - np.max(distances)) * sharpness)
    probabilities /= probabilities.sum()

    # Sample a target bucket based on the distribution
    chosen_idx = np.random.choice(len(nearest_buckets), p=probabilities)
    _, target_size = nearest_buckets[chosen_idx]
    target_w, target_h = target_size
    target_ar = target_w / target_h

    # Determine the crop box to match the target aspect ratio
    if original_ar > target_ar:
        # Original is wider than target -> crop width
        new_w = int(target_ar * original_h)
        offset = (original_w - new_w) // 2
        crop_box = (offset, 0, offset + new_w, original_h)
    else:
        # Original is taller than target -> crop height
        new_h = int(original_w / target_ar)
        offset = (original_h - new_h) // 2
        crop_box = (0, offset, original_w, offset + new_h)
        
    return crop_box, target_size


def process_image(
    image: Image.Image, 
    prescale_size: Optional[tuple], 
    crop_box: tuple, 
    target_size: tuple
):
    """
    Applies an optional prescale, a crop, and a final resize, handling both
    "global view" (prescale-then-crop) and "detail view" (crop-then-resize) modes.
    """
    # --- MODE A: Prescale-then-Crop (Global View) ---
    if prescale_size is not None:
        # 1. First, resize the entire image to the calculated prescale size.
        image = image.resize(prescale_size, Image.LANCZOS)
        
        # 2. Then, crop the target window from the prescaled image.
        # The result of this crop is already at the target_size, so no final resize is needed.
        return image.crop(crop_box)
        
    # --- MODE B: Crop-from-Original (Detail View) ---
    else:
        # 1. First, crop the detail window from the full-resolution original image.
        cropped_image = image.crop(crop_box)
        
        # 2. Then, resize that (potentially large) crop down to the target bucket size.
        return cropped_image.resize(target_size, Image.LANCZOS)

def create_prefetch_generator(iterable, num_prefetch=2):
    """
    Creates a generator that fetches items in a background thread.
    'iterable' should be a generator or iterator that yields your data.
    """
    q = queue.Queue(maxsize=num_prefetch)
    sentinel = object()  # Marker for the end of the iterator

    def producer():
        for item in iterable:
            q.put(item)
        q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    def consumer():
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
            q.task_done()

    return consumer()

def batch_generator(
    folder_main: str,
    folders: list,
    scales: list,
    scales_unique: list,
    buckets: list,
    k: int,
    sharpness: float,
    batch_size: int,
    n_tuple: int,
    max_denoising_steps: int,
    lock_target_size:bool = True,
):
    """
    Yields packets of structured data for N-tuple contrastive training.
    Respects the arbitrary mapping between scales and folder names.
    """
    # Create numpy arrays for efficient lookup
    scales_array = np.array(scales)
    folders_array = np.array(folders)
    
    while True:
        try:
            # --- Initialize lists for the batch ---
            batch_images = []         # Shape: (B, N) of PILs
            batch_scales = []         # Shape: (B, N) of floats
            batch_orig_sizes = []     # Shape: (B, 2) of ints
            batch_crop_boxes = []     # Shape: (B, 4) of ints
            batch_target_sizes = []   # Shape: (B, 2) of ints
            batch_seeds = []          # Shape: (B,) of ints
            batch_timesteps = []

            if lock_target_size:
                # === BATCH-LEVEL SETUP ===
                # 1. Select a representative image to decide the AR for the whole batch.
                # (This logic to get a random image can be optimized, but is clear for now)
                sample_folder = folders_array[0]
                ims_path = os.path.join(folder_main, sample_folder)
                ims = [f for f in os.listdir(ims_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                if not ims: continue
                rep_image_name = random.choice(ims)
                rep_img_path = os.path.join(ims_path, rep_image_name)
                with Image.open(rep_img_path) as rep_img:
                    rep_original_size = rep_img.size

                # 2. Lock the target size for this entire microbatch.
                target_size = select_target_bucket(
                    rep_original_size, buckets, k_nearest=k, sharpness=sharpness
                )
            else:
                print(f"bro u need to lock target sizes... or implement a batching semantics like paged attention supporting mixed tensor shapes as inputs...") 
                raise NotImplementedError 

            # === TUPLE-LEVEL PROCESSING ===
            for _ in range(batch_size):
                # 1. Select the image name for this specific tuple.
                image_name = random.choice(ims)
                
                # 2. Get its true original size.
                img_path_for_size = os.path.join(ims_path, image_name)
                with Image.open(img_path_for_size) as img_for_size:
                    original_size = img_for_size.size

                # 3. Determine the transform for this tuple using the locked target size.
                # The transform applied to all N images in this tuple will be identical.
                (
                    prescale_size, 
                    crop_box, 
                    reported_original_size
                ) = determine_transform_params(
                    original_size, target_size, prescale_perc=0.2
                )
                # 4. Select N concepts (scales) for this tuple
                sampled_scales = sorted(random.sample(scales_unique, n_tuple))
                
                # 5. Load the N corresponding images using the scale-to-folder map
                images_for_tuple = []
                # 4. Load the N images for the tuple and apply the *exact same transform* to all.
                for scale in sampled_scales:
                    folder_name = folders_array[scales_array == scale][0]
                    img_path = os.path.join(folder_main, folder_name, image_name)
                    with Image.open(img_path) as img:
                        processed_img = process_image(
                            img.convert("RGB"), prescale_size, crop_box, target_size
                        )
                        images_for_tuple.append(processed_img)
                
                # For each tuple, sample a timestep index
                tuple_timestep_index = random.randint(1, max_denoising_steps - 1)
                # 6. Append all data for this tuple to the batch lists
                batch_images.append(images_for_tuple)
                batch_scales.append(sampled_scales)
                batch_orig_sizes.append(reported_original_size)
                batch_crop_boxes.append(crop_box)
                batch_target_sizes.append(target_size)
                batch_seeds.append(random.randint(0, 2**32 - 1))
                batch_timesteps.append(tuple_timestep_index)

            if not batch_images: continue

            # --- Final Tensor Conversion and Broadcasting ---
            b_dim, n_dim = len(batch_scales), len(batch_scales[0])

            timesteps_tensor = torch.tensor(batch_timesteps, dtype=torch.int64)
            broadcast_timesteps = timesteps_tensor.unsqueeze(1).expand(-1, n_tuple)
            
            yield {
                "images":           batch_images,
                "scales":           torch.tensor(batch_scales, dtype=torch.float32),
                "target_sizes":     torch.tensor(batch_target_sizes, dtype=torch.int32).unsqueeze(1).expand(b_dim, n_dim, 2),
                "original_sizes":   torch.tensor(batch_orig_sizes, dtype=torch.int32).unsqueeze(1).expand(b_dim, n_dim, 2),
                "crop_coords":      torch.tensor(batch_crop_boxes, dtype=torch.int32).unsqueeze(1).expand(b_dim, n_dim, 4),
                "tuple_prng_seeds": torch.tensor(batch_seeds, dtype=torch.int64).unsqueeze(1).expand(b_dim, n_dim),
                "timesteps_to":     broadcast_timesteps
            }
            
        except (IOError, FileNotFoundError, IndexError) as e:
            print(f"Warning: Skipping batch due to error in generator: {e}")
            continue

def unflatten_and_split(
        flat_tensor: torch.Tensor, 
        batch_size: int, 
        n_tuple: int
    ) -> tuple[torch.Tensor, ...]:
        """
        Takes a flat tensor of shape (B*N, ...) and splits it into a tuple
        of N tensors, each of shape (B, ...).
        """
        # 1. Get the shape of the trailing dimensions (...)
        trailing_dims = flat_tensor.shape[1:]
        
        # 2. Reshape the flat tensor back to its logical (B, N, ...) structure
        logical_shape = (batch_size, n_tuple) + trailing_dims
        logical_tensor = flat_tensor.view(logical_shape)
        
        # 3. Split along the N_TUPLE dimension (dim=1)
        # This returns a tuple of N tensors, each of shape (B, 1, ...)
        split_tensors = torch.split(logical_tensor, 1, dim=1)
        
        # 4. Squeeze to remove the leftover 'N' dimension from each tensor
        # The final result is a tuple of N tensors, each of shape (B, ...)
        return tuple(tensor.squeeze(1) for tensor in split_tensors)
