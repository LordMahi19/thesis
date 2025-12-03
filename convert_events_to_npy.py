import h5py
import numpy as np
from pathlib import Path

def convert_h5_to_npy(h5_path, output_path):
    """
    Convert v2e generated .h5 file to DVS128 Gesture dataset .npy format.
    
    Args:
        h5_path: Path to input .h5 file
        output_path: Path to output .npy file
    """
    print(f"Converting {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"Available keys in h5 file: {list(f.keys())}")
        
        if 'events' not in f:
            raise KeyError("No 'events' dataset or group found in H5 file.")

        events_obj = f['events']
        print(f"Type of f['events']: {type(events_obj)}")

        if isinstance(events_obj, h5py.Dataset):
            print(f"Found 'events' as a single dataset with shape: {events_obj.shape}")
            if events_obj.dtype.names:
                # It is a structured array
                print("Dataset is structured.")
                d = events_obj[:]
                x = d['x']
                y = d['y']
                t = d['t']
                if 'p' in d.dtype.names:
                    p = d['p']
                elif 'pol' in d.dtype.names:
                    p = d['pol']
                else:
                    raise KeyError("Could not find polarity field.")
            else:
                # It's a plain N x 4 array, assume order t, x, y, p
                print("Dataset is not structured, assuming N x 4 layout (t, x, y, p).")
                data = events_obj[:]
                t = data[:, 0]
                x = data[:, 1]
                y = data[:, 2]
                p = data[:, 3]

        elif isinstance(events_obj, h5py.Group):
            print(f"Found 'events' as a group. Keys: {list(events_obj.keys())}")
            x = events_obj['x'][:]
            y = events_obj['y'][:]
            t = events_obj['t'][:]
            if 'p' in events_obj.keys():
                p = events_obj['p'][:]
            elif 'pol' in events_obj.keys():
                p = events_obj['pol'][:]
            else:
                raise KeyError("Could not find polarity field.")
        else:
            raise TypeError(f"Unsupported type for 'events': {type(events_obj)}")

    # Data type conversion and normalization
    x = np.clip(x, 0, 127).astype(np.uint16)
    y = np.clip(y, 0, 127).astype(np.uint16)
    p = (p > 0).astype(np.uint8)
    
    t = t.astype(np.uint64)
    if t.size > 0:
        t = t - t.min()
    
    # Create and save structured array
    num_events = len(t)
    events = np.zeros(num_events, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])
    
    events['t'] = t
    events['x'] = x
    events['y'] = y
    events['p'] = p
    
    np.save(output_path, events)
    
    print(f"Saved {num_events} events to {output_path}")
    if num_events > 0:
        print(f"Time range: {t.min()}us to {t.max()}us")
        print(f"Spatial range: x=[{x.min()}, {x.max()}], y=[{y.min()}, {y.max()}]")
        print(f"Polarities: {np.unique(p)}\n")
    
    return events

def batch_convert_h5_to_npy():
    """
    Convert all .h5 files in generated_events directory to .npy format
    and organize them for the DVS128 dataset structure.
    """
    # Paths
    generated_events_dir = Path("./dataset/generated_events")
    output_base_dir = Path("./dataset/custom_data")
    
    # Create output directory structure
    # Using a custom user name to distinguish from original data
    output_user_dir = output_base_dir / "user_generated"
    output_user_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .h5 files
    h5_files = list(generated_events_dir.rglob("*.h5"))
    
    if not h5_files:
        print(f"No .h5 files found in {generated_events_dir}")
        return
    
    print(f"Found {len(h5_files)} .h5 files to convert.\n")
    
    class_counts = {}
    
    # Convert each file
    for h5_file in sorted(h5_files):
        try:
            # The class label is the name of the parent directory
            class_label = int(h5_file.parent.name)
            
            # Get a sample index for this class
            sample_idx = class_counts.get(class_label, 0)
            
            # Output filename: class_label_sample_idx.npy
            output_filename = f"{class_label}_{sample_idx}.npy"
            output_path = output_user_dir / output_filename
            
            convert_h5_to_npy(h5_file, output_path)
            
            # Verify the conversion
            loaded = np.load(output_path)
            print(f"Verification - Shape: {loaded.shape}, Dtype: {loaded.dtype}")
            print(f"First 3 events:\n{loaded[:3]}\n")
            print("-" * 60)
            
            # Update counts
            class_counts[class_label] = sample_idx + 1
            
        except Exception as e:
            print(f"Error converting {h5_file}: {e}\n")

    total_converted = sum(class_counts.values())
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print(f"Output directory: {output_user_dir}")
    print(f"Converted {total_converted} files.")
    print("Files per class:")
    for class_label, count in sorted(class_counts.items()):
        print(f"  Class {class_label}: {count} files")
    print("="*60)

if __name__ == "__main__":
    batch_convert_h5_to_npy()