"""
PVD Steganography
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ============= HELPER FUNCTIONS =============

def file_to_bin(filename):
    """Convert file to binary string"""
    with open(filename, 'rb') as f:
        content = f.read()
    return list(''.join(format(byte, '08b') for byte in content))

def bin_to_file(bitstream, output_filename):
    """Convert binary string to file"""
    if isinstance(bitstream, list):
        bitstream = ''.join(bitstream)
    
    padding = 8 - (len(bitstream) % 8)
    if padding != 8:
        bitstream += '0' * padding
    
    byte_array = bytearray()
    for i in range(0, len(bitstream), 8):
        byte = bitstream[i:i+8]
        byte_array.append(int(byte, 2))
    
    with open(output_filename, 'wb') as f:
        f.write(byte_array)

def embed_number(diff):
    """PVD range table"""
    range_table = [
        (0, 7, 3), (8, 15, 3), (16, 31, 4),
        (32, 63, 5), (64, 127, 6), (128, 255, 7),
    ]
    for lower, upper, bits in range_table:
        if lower <= diff <= upper:
            return (bits, lower, upper)
    return (0, 0, 0)

def change_difference(pixel1, pixel2, old_diff, new_diff):
    """Adjust pixel values"""
    p1, p2 = int(pixel1), int(pixel2)
    diff_change = new_diff - old_diff
    
    if p1 >= p2:
        p_large, p_small, swap = p1, p2, False
    else:
        p_large, p_small, swap = p2, p1, True
    
    m = abs(diff_change) // 2
    
    if diff_change >= 0:
        new_large = min(255, p_large + diff_change - m)
        new_small = max(0, p_small - m)
    else:
        new_large = max(0, p_large + diff_change + m)
        new_small = min(255, p_small + m)
    
    if swap:
        return (np.uint8(new_small), np.uint8(new_large))
    else:
        return (np.uint8(new_large), np.uint8(new_small))

def embed_pvd(input_img, secret_data):
    """Embed secret data into image using PVD"""
    height, width = input_img.shape[0], input_img.shape[1]
    channels = 3 if len(input_img.shape) == 3 else 1
    
    bit_index = 0
    width -= width % 2
    row = 0
    total_capacity = 0
    msg_not_finished = True
    
    while row < height and msg_not_finished:
        for col in range(0, width, 2):
            for ch in range(channels):
                if channels == 1:
                    pixel_diff = abs(int(input_img[row, col + 1]) - int(input_img[row, col]))
                else:
                    pixel_diff = abs(int(input_img[row, col + 1, ch]) - int(input_img[row, col, ch]))
                
                embed_info = embed_number(pixel_diff)
                
                if bit_index + embed_info[0] >= len(secret_data):
                    msg_not_finished = False
                    break
                
                embedded_bits = int(''.join(secret_data[bit_index:bit_index + embed_info[0]]), base=2)
                new_diff = embed_info[1] + embedded_bits
                
                if channels == 1:
                    input_img[row, col], input_img[row, col + 1] = change_difference(
                        input_img[row, col], input_img[row, col + 1], pixel_diff, new_diff)
                else:
                    input_img[row, col, ch], input_img[row, col + 1, ch] = change_difference(
                        input_img[row, col, ch], input_img[row, col + 1, ch], pixel_diff, new_diff)
                
                bit_index += embed_info[0]
                total_capacity += embed_info[0]
            
            if not msg_not_finished:
                break
        row += 1
    
    return input_img, total_capacity, bit_index

def extract_pvd(input_img, num_bits):
    """Extract specific number of bits from stego image"""
    height, width = input_img.shape[0], input_img.shape[1]
    channels = 3 if len(input_img.shape) == 3 else 1
    width -= width % 2
    
    bitstream = []
    bits_extracted = 0
    
    for row in range(height):
        if bits_extracted >= num_bits:
            break
        for col in range(0, width, 2):
            if bits_extracted >= num_bits:
                break
            for ch in range(channels):
                if bits_extracted >= num_bits:
                    break
                    
                if channels == 1:
                    pixel_diff = abs(int(input_img[row, col + 1]) - int(input_img[row, col]))
                else:
                    pixel_diff = abs(int(input_img[row, col + 1, ch]) - int(input_img[row, col, ch]))
                
                embed_info = embed_number(pixel_diff)
                secret_num = pixel_diff - embed_info[1]
                bits = bin(secret_num)[2:].rjust(embed_info[0], "0")
                
                remaining = num_bits - bits_extracted
                if remaining >= len(bits):
                    bitstream.extend(bits)
                    bits_extracted += len(bits)
                else:
                    bitstream.extend(bits[:remaining])
                    bits_extracted = num_bits
                    break
    
    return bitstream

def calculate_metrics(original_img, stego_img):
    """Calculate quality metrics"""
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    if isinstance(stego_img, Image.Image):
        stego_img = np.array(stego_img)
    
    mse = np.mean((original_img.astype(float) - stego_img.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    diff = np.abs(original_img.astype(int) - stego_img.astype(int))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    changed_pixels = np.count_nonzero(diff)
    total_pixels = original_img.size
    change_percentage = (changed_pixels / total_pixels) * 100
    
    return {
        'mse': mse,
        'psnr': psnr,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'changed_pixels': changed_pixels,
        'total_pixels': total_pixels,
        'change_percentage': change_percentage
    }

def create_difference_image(original_img, stego_img, output_path):
    """Create visual difference map"""
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    if isinstance(stego_img, Image.Image):
        stego_img = np.array(stego_img)
    
    diff = np.abs(original_img.astype(int) - stego_img.astype(int))
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)
    
    Image.fromarray(diff_amplified).save(output_path)
    return diff_amplified

def create_comparison_figure(original_img, stego_img, metrics, output_path, image_name):
    """
    Create side-by-side comparison figure with labels and comprehensive metrics
    Including difference rate and quality assessment
    """
    # Convert to numpy arrays if needed
    if isinstance(original_img, Image.Image):
        original_array = np.array(original_img)
    else:
        original_array = original_img
    
    if isinstance(stego_img, Image.Image):
        stego_array = np.array(stego_img)
    else:
        stego_array = stego_img
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot original image
    ax1.imshow(original_array)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold', pad=10)
    ax1.axis('on')
    ax1.set_xlabel('Pixels', fontsize=11)
    ax1.set_ylabel('Pixels', fontsize=11)
    ax1.grid(False)
    
    # Plot stego image
    ax2.imshow(stego_array)
    ax2.set_title('Stego Image', fontsize=16, fontweight='bold', pad=10)
    ax2.axis('on')
    ax2.set_xlabel('Pixels', fontsize=11)
    ax2.set_ylabel('Pixels', fontsize=11)
    ax2.grid(False)
    
    # Calculate difference percentage for unchanged pixels
    unchanged_percentage = 100 - metrics['change_percentage']
    
    # Determine quality assessment
    if metrics['psnr'] > 50:
        quality = "Excellent"
        color = 'green'
    elif metrics['psnr'] > 40:
        quality = "Good"
        color = 'blue'
    else:
        quality = "Acceptable"
        color = 'orange'
    
    # Create comprehensive metrics text with multiple lines
    metrics_line1 = f"PSNR: {metrics['psnr']:.2f} dB  |  MSE: {metrics['mse']:.6f}  |  Max Diff: {metrics['max_diff']}/255"
    metrics_line2 = f"Pixels Changed: {metrics['changed_pixels']:,} / {metrics['total_pixels']:,} ({metrics['change_percentage']:.4f}%)"
    metrics_line3 = f"Pixels Unchanged: {metrics['total_pixels'] - metrics['changed_pixels']:,} ({unchanged_percentage:.4f}%)  |  Mean Diff: {metrics['mean_diff']:.4f}"
    
    # Add metrics text in a box below the images
    textstr = f"{metrics_line1}\n{metrics_line2}\n{metrics_line3}\nQuality: {quality} (Imperceptible)"
    
    # Position the text box
    fig.text(0.5, 0.02, textstr, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor=color, linewidth=2),
            family='monospace')
    
    # Add a main title
    fig.suptitle(f'PVD Steganography - Visual Comparison\n{image_name}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison figure saved: {output_path}")

def process_image(image_path, secret_file, output_dir):
    """Process a single image with all outputs"""
    
    image_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(image_name)[0]
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}")
    
    # Load original image
    try:
        original_pil = Image.open(image_path).convert('RGB')
        original_array = np.array(original_pil)
        print(f"✓ Image loaded: {original_array.shape}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None
    
    # Load secret data
    secret_data = file_to_bin(secret_file)
    secret_size_bits = len(secret_data)
    secret_size_bytes = secret_size_bits / 8
    print(f"✓ Secret data: {secret_size_bits} bits ({secret_size_bytes:.1f} bytes)")
    
    # Calculate image capacity
    height, width, channels = original_array.shape
    max_capacity = (height * (width // 2) * 2 * channels * 7)
    print(f"✓ Image capacity: ~{max_capacity} bits (max theoretical)")
    
    if secret_size_bits > max_capacity * 0.8:
        print(f"⚠ Warning: Secret message may be too large for this image")
    
    # Embed secret message
    print(f"\nEmbedding secret message...")
    stego_array = original_array.copy()
    stego_array, capacity_used, bits_embedded = embed_pvd(stego_array, secret_data)
    print(f"✓ Embedded: {bits_embedded}/{secret_size_bits} bits ({bits_embedded/8:.1f} bytes)")
    print(f"✓ Capacity used: {capacity_used} bits")
    
    # Calculate metrics
    print(f"\nCalculating quality metrics...")
    metrics = calculate_metrics(original_array, stego_array)
    
    print(f"\n--- Quality Metrics ---")
    print(f"MSE (Mean Squared Error):     {metrics['mse']:.6f}")
    print(f"PSNR (Peak SNR):              {metrics['psnr']:.2f} dB")
    print(f"Max pixel difference:         {metrics['max_diff']}")
    print(f"Mean pixel difference:        {metrics['mean_diff']:.4f}")
    print(f"Changed pixels:               {metrics['changed_pixels']:,} / {metrics['total_pixels']:,}")
    print(f"Change percentage:            {metrics['change_percentage']:.4f}%")
    
    # Quality assessment
    if metrics['psnr'] > 50:
        quality = "EXCELLENT (imperceptible)"
    elif metrics['psnr'] > 40:
        quality = "GOOD (minor artifacts)"
    elif metrics['psnr'] > 30:
        quality = "ACCEPTABLE (visible differences)"
    else:
        quality = "POOR (significant distortion)"
    print(f"Quality assessment:           {quality}")
    
    # Save stego image
    stego_path = os.path.join(output_dir, f"{name_without_ext}_stego.png")
    Image.fromarray(stego_array).save(stego_path)
    print(f"\n✓ Stego image saved: {stego_path}")
    
    # Create and save difference image
    diff_path = os.path.join(output_dir, f"{name_without_ext}_diff.png")
    create_difference_image(original_array, stego_array, diff_path)
    print(f"✓ Difference map saved: {diff_path}")
    
    # Create side-by-side comparison figure
    comparison_path = os.path.join(output_dir, f"{name_without_ext}_comparison.png")
    create_comparison_figure(original_array, stego_array, metrics, comparison_path, image_name)
    
    # Extract and verify
    print(f"\nVerifying extraction...")
    extracted_bits = extract_pvd(stego_array, secret_size_bits)
    recovered_path = os.path.join(output_dir, f"{name_without_ext}_recovered.txt")
    bin_to_file(extracted_bits, recovered_path)
    
    # Verify correctness
    with open(secret_file, 'rb') as f:
        original_secret = f.read()
    with open(recovered_path, 'rb') as f:
        recovered_secret = f.read()[:len(original_secret)]
    
    if original_secret == recovered_secret:
        print(f"✓ Extraction: 100% accurate!")
        accuracy = 100.0
    else:
        matches = sum(1 for i in range(min(len(original_secret), len(recovered_secret))) 
                    if original_secret[i] == recovered_secret[i])
        accuracy = (matches / len(original_secret)) * 100
        print(f"⚠ Extraction: {accuracy:.2f}% accurate ({matches}/{len(original_secret)} bytes)")
    
    return {
        'image_name': image_name,
        'image_size': f"{width}x{height}",
        'secret_size': secret_size_bytes,
        'bits_embedded': bits_embedded,
        'capacity_used': capacity_used,
        'mse': metrics['mse'],
        'psnr': metrics['psnr'],
        'max_diff': metrics['max_diff'],
        'mean_diff': metrics['mean_diff'],
        'change_percentage': metrics['change_percentage'],
        'accuracy': accuracy,
        'stego_path': stego_path,
        'diff_path': diff_path,
        'comparison_path': comparison_path,
        'recovered_path': recovered_path
    }

# ============= MAIN PROGRAM =============

def main():
    print("=" * 70)
    print("PVD STEGANOGRAPHY - ENHANCED BATCH PROCESSOR")
    print("With Side-by-Side Comparison Figures")
    print("=" * 70)
    
    # Get input directory
    print("\nEnter the directory containing images:")
    print("(Or press Enter to use current directory)")
    input_dir = input("Directory: ").strip().strip('"')
    
    if not input_dir:
        input_dir = os.getcwd()
    
    if not os.path.exists(input_dir):
        print(f"\n✗ Error: Directory not found: {input_dir}")
        input("\nPress Enter to exit...")
        return
    
    print(f"\n✓ Input directory: {input_dir}")
    
    # Get secret file
    print("\nEnter the path to your secret message file:")
    secret_file = input("Secret file: ").strip().strip('"')
    
    if not os.path.exists(secret_file):
        print(f"\n✗ Error: Secret file not found: {secret_file}")
        input("\nPress Enter to exit...")
        return
    
    print(f"✓ Secret file: {secret_file}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(input_dir, f"PVD_Output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Find all image files
    image_files = [f for f in os.listdir(input_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"\n✗ No image files found in {input_dir}")
        input("\nPress Enter to exit...")
        return
    
    print(f"\n✓ Found {len(image_files)} images")
    
    # Process each image
    results = []
    start_time = time.time()
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n{'#'*70}")
        print(f"IMAGE {i}/{len(image_files)}")
        print(f"{'#'*70}")
        
        img_path = os.path.join(input_dir, img_file)
        result = process_image(img_path, secret_file, output_dir)
        
        if result:
            results.append(result)
    
    # Generate summary report
    print(f"\n{'='*70}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*70}")
    
    report_path = os.path.join(output_dir, "REPORT.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PVD STEGANOGRAPHY - BATCH PROCESSING REPORT\n")
        f.write("With Side-by-Side Comparison Figures\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Secret file: {secret_file}\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
        f.write("="*70 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'='*70}\n")
            f.write(f"IMAGE {i}: {result['image_name']}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Image size:           {result['image_size']}\n")
            f.write(f"Secret size:          {result['secret_size']:.1f} bytes\n")
            f.write(f"Bits embedded:        {result['bits_embedded']}\n")
            f.write(f"Capacity used:        {result['capacity_used']} bits\n")
            f.write(f"MSE:                  {result['mse']:.6f}\n")
            f.write(f"PSNR:                 {result['psnr']:.2f} dB\n")
            f.write(f"Max difference:       {result['max_diff']}\n")
            f.write(f"Mean difference:      {result['mean_diff']:.4f}\n")
            f.write(f"Pixels changed:       {result['change_percentage']:.4f}%\n")
            f.write(f"Extraction accuracy:  {result['accuracy']:.2f}%\n")
            f.write(f"\nOutput files:\n")
            f.write(f"  - Stego image:      {os.path.basename(result['stego_path'])}\n")
            f.write(f"  - Difference map:   {os.path.basename(result['diff_path'])}\n")
            f.write(f"  - Comparison figure:{os.path.basename(result['comparison_path'])}\n")
            f.write(f"  - Recovered secret: {os.path.basename(result['recovered_path'])}\n")
        
        # Summary statistics
        if results:
            avg_psnr = sum(r['psnr'] for r in results) / len(results)
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            f.write(f"\n{'='*70}\n")
            f.write("SUMMARY STATISTICS\n")
            f.write(f"{'='*70}\n")
            f.write(f"Average PSNR:         {avg_psnr:.2f} dB\n")
            f.write(f"Average accuracy:     {avg_accuracy:.2f}%\n")
            f.write(f"Successful extractions: {sum(1 for r in results if r['accuracy'] == 100)}/{len(results)}\n")
    
    print(f"\n✓ Summary report saved: {report_path}")
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed: {len(results)} images")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Output folder: {output_dir}")
    print(f"\nFiles created for each image:")
    print(f"  - *_stego.png        (image with hidden message)")
    print(f"  - *_diff.png         (visual difference map)")
    print(f"  - *_comparison.png   (side-by-side comparison) ← NEW!")
    print(f"  - *_recovered.txt    (extracted secret message)")
    print(f"  - REPORT.txt         (detailed comparison report)")
    print(f"{'='*70}")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()