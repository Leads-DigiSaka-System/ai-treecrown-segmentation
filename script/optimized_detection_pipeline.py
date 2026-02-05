"""
Optimized Detection Pipeline for Google Colab
==============================================
Key optimizations:
1. Memory management with explicit cleanup
2. Configurable batch processing
3. Progressive memory release
4. GPU memory optimization
5. Smart defaults for Colab constraints
"""

import gc
import time
import warnings
from contextlib import contextmanager

import numpy as np
import cv2
import torch
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from tqdm.auto import tqdm
import json
import os

# Conditional imports
try:
    import dask.bag as db
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available - will use sequential processing")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn("SAM not available")


@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class MemoryOptimizer:
    """Utility class for memory optimization"""
    
    @staticmethod
    def clear_memory():
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_optimal_batch_size(num_items, base_size=50, max_size=200):
        """Calculate optimal batch size based on available memory"""
        if torch.cuda.is_available():
            try:
                mem_info = torch.cuda.mem_get_info()
                available_gb = mem_info[0] / (1024**3)
                
                if available_gb > 10:
                    return min(max_size, num_items)
                elif available_gb > 5:
                    return min(base_size * 2, num_items)
                else:
                    return min(base_size, num_items)
            except:
                pass
        
        return min(base_size, num_items)
    
    @staticmethod
    def estimate_memory_usage(image_shape, num_detections):
        """Estimate memory usage in MB"""
        h, w, c = image_shape
        image_mb = (h * w * c * 4) / (1024**2)  # float32
        detections_mb = num_detections * 0.1  # rough estimate
        return image_mb + detections_mb


class DetectionState:
    """State container for detection pipeline with memory tracking"""

    def __init__(self, image_path, model, config):
        self.image_path = image_path
        self.model = model
        self.config = config

        # State variables
        self.full_image = None
        self.original_dims = None
        self.resized_image = None
        self.scaled_dims = None
        self.transform = None
        self.crs = None

        # Detection state
        self.boxes_list = []
        self.segmentation_results = []
        self.gdf = None
        self.detection_info = []

        # SAM state
        self.sam_predictor = None

        # Timing and memory
        self.start_time = time.process_time()
        self.memory_peak = 0

    def __repr__(self):
        return (f"DetectionState(boxes={len(self.boxes_list)}, "
                f"segments={len(self.segmentation_results)}, "
                f"gdf_size={len(self.gdf) if self.gdf is not None else 0})")
    
    def cleanup_intermediate(self):
        """Clean up intermediate data to save memory"""
        if self.resized_image is not None and self.full_image is not None:
            # Keep only one version if possible
            del self.resized_image
            self.resized_image = None
        MemoryOptimizer.clear_memory()


class StateTransition:
    """Base class for state transitions"""

    def __call__(self, state):
        raise NotImplementedError
    
    def _log_memory(self, message=""):
        """Log current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved {message}")


class LoadImageTransition(StateTransition):
    """Transition: Load and prepare image with memory optimization"""

    def __call__(self, state):
        max_load_size = state.config.get('max_load_size', 8192)
        
        try:
            with rasterio.open(state.image_path) as src:
                state.original_dims = (src.width, src.height)
                state.transform = src.transform
                state.crs = src.crs

                print(f"Original image size: {state.original_dims[0]}x{state.original_dims[1]}")
                
                # Check if image is too large
                if max(state.original_dims) > max_load_size:
                    scale = max_load_size / max(state.original_dims)
                    new_w = int(state.original_dims[0] * scale)
                    new_h = int(state.original_dims[1] * scale)
                    print(f"⚠ Large image detected, loading at {new_w}x{new_h}")
                    
                    # Use windowed reading for memory efficiency
                    window = rasterio.windows.Window(0, 0, src.width, src.height)
                    out_shape = (src.count, new_h, new_w)
                    
                    data = src.read(out_shape=out_shape, window=window)
                    
                    if src.count >= 3:
                        state.full_image = np.transpose(data[[0, 1, 2]], (1, 2, 0))
                    else:
                        state.full_image = data[0]
                        if len(state.full_image.shape) == 2:
                            state.full_image = cv2.cvtColor(state.full_image, cv2.COLOR_GRAY2RGB)
                else:
                    if src.count >= 3:
                        state.full_image = src.read([1, 2, 3])
                        state.full_image = np.transpose(state.full_image, (1, 2, 0))
                    else:
                        state.full_image = src.read(1)
                        if len(state.full_image.shape) == 2:
                            state.full_image = cv2.cvtColor(state.full_image, cv2.COLOR_GRAY2RGB)

                state.full_image = np.clip(state.full_image, 0, 255).astype(np.uint8)
                
        except Exception as e:
            print(f"Loading as regular image (not GeoTIFF): {e}")
            state.full_image = cv2.imread(state.image_path)
            if state.full_image is None:
                raise ValueError(f"Could not load image from {state.image_path}")

            state.full_image = cv2.cvtColor(state.full_image, cv2.COLOR_BGR2RGB)
            h, w = state.full_image.shape[:2]
            
            # Check size limit
            if max(w, h) > max_load_size:
                scale = max_load_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                print(f"⚠ Large image detected, resizing to {new_w}x{new_h}")
                state.full_image = cv2.resize(state.full_image, (new_w, new_h), 
                                             interpolation=cv2.INTER_AREA)
                w, h = new_w, new_h
            
            state.original_dims = (w, h)
            print(f"Loaded image size: {w}x{h}")

            state.transform = from_bounds(0, 0, w, h, w, h)
            state.crs = None

        # Log estimated memory
        mem_estimate = MemoryOptimizer.estimate_memory_usage(
            state.full_image.shape, 0
        )
        print(f"  Estimated memory usage: {mem_estimate:.1f} MB")
        
        return state


class ResizeImageTransition(StateTransition):
    """Transition: Resize image based on resolution config"""

    def __call__(self, state):
        resolution = state.config['resolution']
        w, h = state.original_dims

        new_w = int(w * resolution)
        new_h = int(h * resolution)
        state.scaled_dims = (new_w, new_h)

        print(f"Processing at: {new_w}x{new_h} (resolution={resolution})")

        # Use INTER_AREA for downsampling (more memory efficient)
        interp = cv2.INTER_AREA if resolution < 1.0 else cv2.INTER_LINEAR
        
        state.resized_image = cv2.resize(
            state.full_image,
            (new_w, new_h),
            interpolation=interp
        )
        
        # If we're downsampling significantly, we can free the original
        if resolution < 0.5:
            del state.full_image
            state.full_image = state.resized_image.copy()
            MemoryOptimizer.clear_memory()

        return state


class YOLODetectionTransition(StateTransition):
    """Transition: Perform YOLO detection with optimized batching"""

    def __call__(self, state):
        tile_size = state.config['tile_size']
        overlap = state.config['overlap']
        conf_threshold = state.config['conf_threshold']
        iou_threshold = state.config['iou_threshold']
        max_det = state.config['max_det']
        resolution = state.config['resolution']
        use_dask = state.config.get('use_dask', False) and DASK_AVAILABLE
        
        # Colab optimization: limit dask usage
        max_tiles_for_dask = state.config.get('max_tiles_for_dask', 500)

        new_w, new_h = state.scaled_dims
        step_size = tile_size - overlap

        # Generate tile positions
        tile_positions = []
        for y in range(0, new_h, step_size):
            for x in range(0, new_w, step_size):
                x_end = min(x + tile_size, new_w)
                y_end = min(y + tile_size, new_h)
                tile_positions.append((x, y, x_end, y_end))

        print(f"Processing {len(tile_positions)} tiles with YOLO...")
        
        # Smart decision on parallel processing
        if use_dask and 10 < len(tile_positions) <= max_tiles_for_dask:
            print(f"  Using Dask parallel processing...")
            state.boxes_list = self._process_tiles_dask(
                state, tile_positions, tile_size, 
                conf_threshold, iou_threshold, max_det, resolution
            )
        else:
            if len(tile_positions) > max_tiles_for_dask:
                print(f"  Too many tiles ({len(tile_positions)}), using sequential with batching...")
            state.boxes_list = self._process_tiles_sequential(
                state, tile_positions, tile_size,
                conf_threshold, iou_threshold, max_det, resolution
            )

        print(f"YOLO detected {len(state.boxes_list)} boxes before NMS")
        
        # Cleanup
        MemoryOptimizer.clear_memory()
        
        return state

    def _process_tiles_sequential(self, state, tile_positions, tile_size, 
                                  conf, iou, max_det, resolution):
        """Process tiles sequentially with memory management"""
        all_boxes = []
        batch_size = state.config.get('yolo_batch_size', 10)
        
        for i in tqdm(range(0, len(tile_positions), batch_size), desc="YOLO Detection"):
            batch_tiles = tile_positions[i:i+batch_size]
            
            for x, y, x_end, y_end in batch_tiles:
                win_w = x_end - x
                win_h = y_end - y
                tile = state.resized_image[y:y_end, x:x_end]

                if tile.shape[:2] != (tile_size, tile_size):
                    tile = cv2.resize(tile, (tile_size, tile_size))
                    scale_x = win_w / tile_size
                    scale_y = win_h / tile_size
                else:
                    scale_x = scale_y = 1.0

                tile = np.ascontiguousarray(tile)

                with gpu_memory_manager():
                    results = state.model.predict(
                        source=tile,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        save_txt=False,
                        save_conf=True,
                        verbose=False
                    )

                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()

                        for j in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[j]
                            x1_orig = int((x1 * scale_x + x) / resolution)
                            y1_orig = int((y1 * scale_y + y) / resolution)
                            x2_orig = int((x2 * scale_x + x) / resolution)
                            y2_orig = int((y2 * scale_y + y) / resolution)

                            all_boxes.append({
                                'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                                'confidence': float(confidences[j])
                            })
            
            # Periodic cleanup
            if i % (batch_size * 5) == 0:
                MemoryOptimizer.clear_memory()
        
        return all_boxes

    def _process_single_tile(self, args):
        """Process a single tile - designed for Dask"""
        resized_image, x, y, x_end, y_end, tile_size, model_path, conf, iou, max_det, resolution = args
        
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        win_w = x_end - x
        win_h = y_end - y
        tile = resized_image[y:y_end, x:x_end]

        if tile.shape[:2] != (tile_size, tile_size):
            tile = cv2.resize(tile, (tile_size, tile_size))
            scale_x = win_w / tile_size
            scale_y = win_h / tile_size
        else:
            scale_x = scale_y = 1.0

        tile = np.ascontiguousarray(tile)

        results = model.predict(
            source=tile,
            conf=conf,
            iou=iou,
            max_det=max_det,
            save_txt=False,
            save_conf=True,
            verbose=False
        )

        boxes_from_tile = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1_orig = int((x1 * scale_x + x) / resolution)
                y1_orig = int((y1 * scale_y + y) / resolution)
                x2_orig = int((x2 * scale_x + x) / resolution)
                y2_orig = int((y2 * scale_y + y) / resolution)

                boxes_from_tile.append({
                    'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                    'confidence': float(confidences[i])
                })

        return boxes_from_tile

    def _process_tiles_dask(self, state, tile_positions, tile_size, 
                           conf, iou, max_det, resolution):
        """Process tiles in parallel using Dask with memory limits"""
        model_path = state.config.get('model_path')
        
        # Limit partitions for Colab
        max_partitions = min(len(tile_positions), state.config.get('max_dask_partitions', 20))
        
        tile_args = [
            (state.resized_image, x, y, x_end, y_end, tile_size, 
             model_path, conf, iou, max_det, resolution)
            for x, y, x_end, y_end in tile_positions
        ]
        
        bag = db.from_sequence(tile_args, npartitions=max_partitions)
        
        with ProgressBar():
            results = bag.map(self._process_single_tile).compute()
        
        all_boxes = []
        for tile_boxes in results:
            all_boxes.extend(tile_boxes)
        
        return all_boxes


class NMSTransition(StateTransition):
    """Transition: Apply Non-Maximum Suppression"""

    def __call__(self, state):
        if not state.config.get('apply_nms', True) or len(state.boxes_list) == 0:
            return state

        nms_iou = state.config.get('nms_iou', 0.3)
        print(f"Applying NMS with IOU threshold={nms_iou}...")
        state.boxes_list = self._apply_nms(state.boxes_list, nms_iou)
        print(f"After NMS: {len(state.boxes_list)} boxes")

        return state

    @staticmethod
    def _apply_nms(boxes_list, iou_threshold):
        if len(boxes_list) == 0:
            return boxes_list

        boxes_list = sorted(boxes_list, key=lambda x: x['confidence'], reverse=True)
        keep = []
        removed = set()

        for i in range(len(boxes_list)):
            if i in removed:
                continue

            keep.append(boxes_list[i])
            box_i = boxes_list[i]['bbox']

            for j in range(i + 1, len(boxes_list)):
                if j in removed:
                    continue

                box_j = boxes_list[j]['bbox']
                iou = NMSTransition._calculate_iou(box_i, box_j)

                if iou > iou_threshold:
                    removed.add(j)

        return keep

    @staticmethod
    def _calculate_iou(box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


class SAMInitTransition(StateTransition):
    """Transition: Initialize SAM model with memory optimization"""

    def __call__(self, state):
        if not SAM_AVAILABLE:
            print("⚠ SAM not available, skipping segmentation")
            return state
            
        sam_checkpoint = state.config['sam_checkpoint']
        model_type = state.config.get('sam_model_type', 'vit_h')
        device = state.config.get('device', 'cuda')

        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint}")

        print(f"\nLoading SAM model ({model_type})...")

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"

        # Clear memory before loading SAM
        MemoryOptimizer.clear_memory()

        with gpu_memory_manager():
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            state.sam_predictor = SamPredictor(sam)

        print(f"SAM model loaded on {device}")
        self._log_memory()

        return state


class SAMSegmentationTransition(StateTransition):
    """Transition: Perform SAM segmentation with chunked processing and batching"""

    def __call__(self, state):
        if not SAM_AVAILABLE or state.sam_predictor is None:
            print("⚠ Skipping SAM segmentation")
            return state
            
        print(f"Segmenting {len(state.boxes_list)} objects with SAM...")
        
        h, w = state.full_image.shape[:2]
        max_image_size = state.config.get('sam_max_image_size', 2048)  # Lower for Colab
        use_chunked = max(h, w) > max_image_size
        
        if use_chunked:
            print(f"⚠ Large image ({w}x{h}), using chunked processing (max: {max_image_size}px)")
            self._segment_chunked(state, max_image_size)
        else:
            print(f"Setting full image for SAM ({w}x{h})")
            with gpu_memory_manager():
                state.sam_predictor.set_image(state.full_image)
            self._segment_full(state)

        print(f"SAM segmentation complete: {len(state.segmentation_results)} masks generated")
        
        # Cleanup SAM predictor to free GPU memory
        if state.config.get('free_sam_after_use', True):
            del state.sam_predictor
            state.sam_predictor = None
            MemoryOptimizer.clear_memory()
            print("  Freed SAM model from memory")
        
        return state

    def _segment_full(self, state):
        """Segment using full image context with batching"""
        batch_size = state.config.get('sam_batch_size', 20)
        
        for i in tqdm(range(0, len(state.boxes_list), batch_size), desc="SAM Segmentation"):
            batch = state.boxes_list[i:i+batch_size]
            
            for box_info in batch:
                bbox = box_info['bbox']
                confidence = box_info['confidence']
                input_box = np.array(bbox)

                with gpu_memory_manager():
                    try:
                        masks, scores, _ = state.sam_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )

                        mask = masks[0]
                        mask_coords = self._mask_to_polygon(mask)

                        if mask_coords is not None and len(mask_coords) >= 3:
                            state.segmentation_results.append({
                                'mask': None,  # Don't store full mask to save memory
                                'coords': mask_coords,
                                'bbox': bbox,
                                'confidence': confidence,
                                'sam_score': float(scores[0])
                            })
                    except Exception as e:
                        print(f"  Warning: Failed to segment box {bbox}: {e}")
                        continue
            
            # Cleanup after each batch
            if i % (batch_size * 3) == 0:
                MemoryOptimizer.clear_memory()

    def _segment_chunked(self, state, max_size):
        """Segment using local image crops with aggressive memory management"""
        h, w = state.full_image.shape[:2]
        batch_size = state.config.get('sam_batch_size', 10)  # Smaller batches for chunked
        
        for i in tqdm(range(0, len(state.boxes_list), batch_size), 
                     desc="SAM Segmentation (Chunked)"):
            batch = state.boxes_list[i:i+batch_size]
            
            for box_info in batch:
                bbox = box_info['bbox']
                confidence = box_info['confidence']
                
                x1, y1, x2, y2 = bbox
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Adaptive padding
                pad_factor = 1.5
                pad_w = int(box_w * (pad_factor - 1) / 2)
                pad_h = int(box_h * (pad_factor - 1) / 2)
                
                crop_x1 = max(0, x1 - pad_w)
                crop_y1 = max(0, y1 - pad_h)
                crop_x2 = min(w, x2 + pad_w)
                crop_y2 = min(h, y2 + pad_h)
                
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                
                # Ensure crop fits in max_size
                if crop_w > max_size or crop_h > max_size:
                    pad_w = min(pad_w, (max_size - box_w) // 2)
                    pad_h = min(pad_h, (max_size - box_h) // 2)
                    crop_x1 = max(0, x1 - pad_w)
                    crop_y1 = max(0, y1 - pad_h)
                    crop_x2 = min(w, x2 + pad_w)
                    crop_y2 = min(h, y2 + pad_h)
                
                crop = state.full_image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                with gpu_memory_manager():
                    try:
                        state.sam_predictor.set_image(crop)
                        
                        bbox_in_crop = [
                            x1 - crop_x1,
                            y1 - crop_y1,
                            x2 - crop_x1,
                            y2 - crop_y1
                        ]
                        
                        input_box = np.array(bbox_in_crop)
                        
                        masks, scores, _ = state.sam_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )
                        
                        mask = masks[0]
                        mask_coords_crop = self._mask_to_polygon(mask)
                        
                        if mask_coords_crop is not None and len(mask_coords_crop) >= 3:
                            mask_coords = mask_coords_crop + np.array([crop_x1, crop_y1])
                            
                            state.segmentation_results.append({
                                'mask': None,
                                'coords': mask_coords,
                                'bbox': bbox,
                                'confidence': confidence,
                                'sam_score': float(scores[0])
                            })
                    except Exception as e:
                        continue
            
            # Aggressive cleanup between batches
            MemoryOptimizer.clear_memory()

    @staticmethod
    def _mask_to_polygon(mask):
        """Convert mask to polygon coordinates"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return approx.reshape(-1, 2)


class CreateGeoDataFrameTransition(StateTransition):
    """Transition: Create GeoDataFrame with optimized processing"""

    def __call__(self, state):
        use_dask = state.config.get('use_dask', False) and DASK_AVAILABLE
        
        if use_dask and len(state.segmentation_results) > 200:
            print(f"  Using Dask for GeoDataFrame creation...")
            self._create_gdf_dask(state)
        else:
            self._create_gdf_sequential(state)
        
        return state

    def _create_gdf_sequential(self, state):
        """Create GeoDataFrame sequentially with batching"""
        all_polygons = []
        detection_info = []
        
        batch_size = 100
        
        for i in tqdm(range(0, len(state.segmentation_results), batch_size),
                     desc="Creating GeoDataFrame"):
            batch = state.segmentation_results[i:i+batch_size]
            
            for result in batch:
                coords = result['coords']

                try:
                    polygon_shapely = Polygon(coords)
                    if not polygon_shapely.is_valid:
                        polygon_shapely = polygon_shapely.buffer(0)
                    area_pixels = polygon_shapely.area
                except:
                    continue

                geo_coords = []
                for px, py in coords:
                    geo_x, geo_y = rasterio.transform.xy(state.transform, py, px)
                    geo_coords.append((geo_x, geo_y))

                try:
                    geo_polygon = Polygon(geo_coords)
                    if not geo_polygon.is_valid:
                        geo_polygon = geo_polygon.buffer(0)
                except:
                    continue

                all_polygons.append(geo_polygon)
                detection_info.append({
                    'centroid_x': coords[:, 0].mean(),
                    'centroid_y': coords[:, 1].mean(),
                    'area_pixels': area_pixels,
                    'confidence': result['confidence'],
                    'sam_score': result['sam_score'],
                    'num_points': len(coords)
                })

        state.gdf = gpd.GeoDataFrame({
            'geometry': all_polygons,
            'area_pixels': [d['area_pixels'] for d in detection_info],
            'confidence': [d['confidence'] for d in detection_info],
            'sam_score': [d['sam_score'] for d in detection_info],
            'centroid_x': [d['centroid_x'] for d in detection_info],
            'centroid_y': [d['centroid_y'] for d in detection_info],
            'num_points': [d['num_points'] for d in detection_info]
        }, crs=state.crs)

        state.detection_info = detection_info

    def _process_single_result(self, args):
        """Process single segmentation result for Dask"""
        result, transform = args
        coords = result['coords']

        try:
            polygon_shapely = Polygon(coords)
            if not polygon_shapely.is_valid:
                polygon_shapely = polygon_shapely.buffer(0)
            area_pixels = polygon_shapely.area
        except:
            return None

        geo_coords = []
        for px, py in coords:
            geo_x, geo_y = rasterio.transform.xy(transform, py, px)
            geo_coords.append((geo_x, geo_y))

        try:
            geo_polygon = Polygon(geo_coords)
            if not geo_polygon.is_valid:
                geo_polygon = geo_polygon.buffer(0)
        except:
            return None

        return {
            'geometry': geo_polygon,
            'area_pixels': area_pixels,
            'confidence': result['confidence'],
            'sam_score': result['sam_score'],
            'centroid_x': coords[:, 0].mean(),
            'centroid_y': coords[:, 1].mean(),
            'num_points': len(coords)
        }

    def _create_gdf_dask(self, state):
        """Create GeoDataFrame using Dask"""
        args_list = [(result, state.transform) for result in state.segmentation_results]
        
        n_partitions = min(len(args_list), state.config.get('max_dask_partitions', 20))
        bag = db.from_sequence(args_list, npartitions=n_partitions)
        
        with ProgressBar():
            results = bag.map(self._process_single_result).compute()
        
        results = [r for r in results if r is not None]
        
        if len(results) == 0:
            state.gdf = gpd.GeoDataFrame()
            state.detection_info = []
            return
        
        state.gdf = gpd.GeoDataFrame(results, crs=state.crs)
        state.detection_info = results


class VisualizationTransition(StateTransition):
    """Transition: Visualize results with memory-efficient rendering"""

    def __call__(self, state):
        output_dir = state.config['output_dir']
        height, width = state.full_image.shape[:2]
        
        max_viz_size = state.config.get('max_viz_size', 2048)
        
        if max(width, height) > max_viz_size:
            print(f"⚠ Large image ({width}x{height}), downsampling for visualization...")
            scale = max_viz_size / max(width, height)
            viz_width = int(width * scale)
            viz_height = int(height * scale)
            viz_image = cv2.resize(state.full_image, (viz_width, viz_height), 
                                  interpolation=cv2.INTER_AREA)
            print(f"  Visualization size: {viz_width}x{viz_height}")
        else:
            viz_image = state.full_image
            viz_width, viz_height = width, height
            scale = 1.0
        
        # Close any existing figures
        plt.close('all')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)  # Lower DPI for Colab
        ax.imshow(viz_image)
        ax.set_title(f'YOLO + SAM Segmentation - {len(state.segmentation_results)} Objects Detected',
                     fontsize=14, fontweight='bold')
        ax.set_axis_off()

        if state.segmentation_results:
            print(f"  Drawing {len(state.segmentation_results)} detections...")
            
            # Limit number of drawn polygons for very large datasets
            max_draw = state.config.get('max_viz_polygons', 1000)
            results_to_draw = state.segmentation_results[:max_draw]
            
            if len(state.segmentation_results) > max_draw:
                print(f"  ⚠ Limiting visualization to {max_draw} polygons")
            
            for result in results_to_draw:
                coords = result['coords'] * scale
                coords_closed = np.vstack([coords, coords[0]])

                ax.plot(coords_closed[:, 0], coords_closed[:, 1],
                       'lime', linewidth=1, alpha=0.7)

                polygon = MPLPolygon(coords, closed=True,
                                   facecolor='lime', alpha=0.2,
                                   edgecolor='lime', linewidth=1)
                ax.add_patch(polygon)

            avg_conf = np.mean([r['confidence'] for r in state.segmentation_results])
            avg_sam = np.mean([r['sam_score'] for r in state.segmentation_results])

            stats_text = (f"Total Objects: {len(state.segmentation_results)}\n"
                         f"Avg YOLO Conf: {avg_conf:.3f}\n"
                         f"Avg SAM Score: {avg_sam:.3f}")

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round", fc="white", alpha=0.9, pad=0.5))

        ax.set_xlim(0, viz_width)
        ax.set_ylim(viz_height, 0)

        plt.tight_layout()
        
        save_dpi = state.config.get('save_dpi', 100)  # Lower DPI for Colab
        plt.savefig(f'{output_dir}/yolo_sam_segmentation.png', 
                   dpi=save_dpi, bbox_inches='tight')
        
        plt.close('all')
        
        print(f"✓ Visualization saved to {output_dir}/yolo_sam_segmentation.png")
        
        # Clear matplotlib memory
        MemoryOptimizer.clear_memory()
        
        return state


class SaveResultsTransition(StateTransition):
    """Transition: Save results to files"""

    def __call__(self, state):
        if len(state.gdf) == 0:
            print("No detections to save")
            return state

        output_dir = state.config['output_dir']
        image_name = state.config.get('image_name', 'output')

        if state.gdf.crs is not None:
            geojson_path = f'{output_dir}/{image_name}_yolo_sam.geojson'
            state.gdf.to_file(geojson_path, driver='GeoJSON')
            print(f"✓ Saved GeoJSON to {geojson_path}")
        else:
            print("⚠ Skipping GeoJSON (no CRS)")

        csv_path = f'{output_dir}/{image_name}_yolo_sam.csv'
        state.gdf.drop(columns='geometry').to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to {csv_path}")

        stats = {
            'total_detections': len(state.gdf),
            'average_area': float(np.mean([d['area_pixels'] for d in state.detection_info])),
            'median_area': float(np.median([d['area_pixels'] for d in state.detection_info])),
            'average_yolo_confidence': float(np.mean([d['confidence'] for d in state.detection_info])),
            'average_sam_score': float(np.mean([d['sam_score'] for d in state.detection_info]))
        }

        stats_path = f'{output_dir}/{image_name}_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")

        return state


class DetectionPipeline:
    """State-space detection pipeline orchestrator with error recovery"""

    def __init__(self, transitions):
        self.transitions = transitions

    def run(self, initial_state):
        state = initial_state

        for i, transition in enumerate(self.transitions):
            print(f"\n{'='*60}")
            print(f"Transition {i+1}/{len(self.transitions)}: {transition.__class__.__name__}")
            print(f"{'='*60}")
            
            try:
                state = transition(state)
            except Exception as e:
                print(f"❌ Error in {transition.__class__.__name__}: {e}")
                
                # Attempt recovery
                MemoryOptimizer.clear_memory()
                
                # For non-critical transitions, continue
                if isinstance(transition, (VisualizationTransition, SaveResultsTransition)):
                    print(f"  ⚠ Continuing despite error in non-critical transition")
                    continue
                else:
                    raise

        return state


def create_detection_pipeline(config):
    """Factory function to create the detection pipeline"""
    transitions = [
        LoadImageTransition(),
        ResizeImageTransition(),
        YOLODetectionTransition(),
        NMSTransition(),
        SAMInitTransition(),
        SAMSegmentationTransition(),
        CreateGeoDataFrameTransition(),
        VisualizationTransition(),
        SaveResultsTransition()
    ]

    return DetectionPipeline(transitions)


def get_colab_optimized_config():
    """
    Returns a configuration optimized for Google Colab constraints
    
    Usage:
        config = get_colab_optimized_config()
        config.update({'image_path': 'your_image.tif', ...})
    """
    return {
        # Image loading
        'max_load_size': 8192,  # Maximum image dimension to load
        
        # Processing resolution
        'resolution': 0.30,  # Process at 50% of original size
        
        # YOLO settings
        'tile_size': 640,
        'overlap': 64,
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_det': 1000,
        'yolo_batch_size': 10,  # Process tiles in batches
        
        # NMS
        'apply_nms': True,
        'nms_iou': 0.3,
        
        # SAM settings
        'sam_model_type': 'vit_h',
        'sam_max_image_size': 2048,  # Use chunked processing above this
        'sam_batch_size': 20,  # Segments per batch
        'free_sam_after_use': True,  # Free GPU memory after SAM
        
        # Dask settings (disabled by default for Colab)
        'use_dask': False,  # Set to True if you have enough memory
        'max_tiles_for_dask': 500,
        'max_dask_partitions': 20,
        
        # Visualization
        'max_viz_size': 2048,
        'max_viz_polygons': 1000,
        'save_dpi': 100,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }