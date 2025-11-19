import numpy as np
from PIL import Image
import os

RANGES = [(0,7,3), (8,15,3), (16,31,4), (32,63,5), (64,127,6), (128,255,8)]

def get_range(d):
    d = abs(d)
    for l, u, b in RANGES:
        if l <= d <= u:
            return l, u, b
    raise ValueError("Invalid difference")

def boundary_safe_embed(p1, p2, target_d):
    p1, p2 = int(p1), int(p2)
    orig_d = abs(int(p1) - int(p2))
    L, U, bits = get_range(orig_d)

    delta = target_d - orig_d
    m = delta // 2
    r = delta - 2 * m        # -1, 0, +1

    larger  = p1 if p1 >= p2 else p2
    smaller = p2 if p1 >= p2 else p1

    new_larger  = larger  + m
    new_smaller = smaller - m
    if r > 0:
        new_larger  += 1
    elif r < 0:
        new_smaller += 1

    # Modulus first
    new_larger  = new_larger % 256
    new_smaller = new_smaller % 256

    np1 = new_larger  if p1 >= p2 else new_smaller
    np2 = new_smaller if p1 >= p2 else new_larger

    if abs(np1 - np2) == target_d:
        return np1, np2, True, bits

    # Fallback – search original range for any embeddable value that stays in bounds
    for payload in range((1 << bits) - 1, -1, -1):
        test_d = L + payload
        delta2 = test_d - orig_d
        m2 = delta2 // 2
        r2 = delta2 - 2 * m2

        tl = larger + m2
        ts = smaller - m2
        if r2 > 0:
            tl += 1
        elif r2 < 0:
            ts += 1

        if 0 <= tl <= 255 and 0 <= ts <= 255:
            np1 = tl if p1 >= p2 else ts
            np2 = ts if p1 >= p2 else tl
            return np1, np2, False, bits

    return p1, p2, False, 0

def embed(img_path, secret_bits, out_path=None):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    h, w, _ = arr.shape

    bits = list(secret_bits)
    bit_idx = perfect = total = embedded = 0
    stego = arr.copy()
    max_x = w - (w % 2)

    for y in range(h):
        for x in range(0, max_x, 2):
            for ch in range(3):
                total += 1
                p1 = arr[y, x, ch]
                p2 = arr[y, x + 1, ch]

                d = abs(int(p1) - int(p2))
                L, U, b = get_range(d)

                if bit_idx + b > len(bits):
                    mse = np.mean((arr.astype(np.float64) - stego.astype(np.float64))**2)
                    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else 99.99
                    print(f"Embedded {embedded//8:,} bytes | PSNR {psnr:.2f} dB | Perfect blocks {100*perfect/total:.2f}%")
                    if out_path:
                        Image.fromarray(stego).save(out_path)
                    return stego, embedded, psnr

                payload = int(''.join(bits[bit_idx:bit_idx + b]), 2)
                np1, np2, is_perfect, used = boundary_safe_embed(p1, p2, L + payload)
                stego[y, x, ch] = np1
                stego[y, x + 1, ch] = np2

                embedded += used
                bit_idx += used
                if is_perfect:
                    perfect += 1

    mse = np.mean((arr.astype(np.float64) - stego.astype(np.float64))**2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else 99.99
    print(f"Capacity full – PSNR {psnr:.2f} dB")
    if out_path:
        Image.fromarray(stego).save(out_path)
    return stego, embedded, psnr

def extract(stego_arr, n_bits):
    h, w, _ = stego_arr.shape
    out = []
    cnt = 0
    max_x = w - (w % 2)
    for y in range(h):
        for x in range(0, max_x, 2):
            for ch in range(3):
                d = abs(int(stego_arr[y, x, ch]) - int(stego_arr[y, x + 1, ch]))
                L, U, b = get_range(d)
                payload = d - L
                out.extend(format(payload, f'0{b}b'))
                cnt += b
                if cnt >= n_bits:
                    return ''.join(out[:n_bits])
    return ''.join(out[:n_bits])

if __name__ == '__main__':
    import random, glob, time

    folder = input("Enter folder path (or press Enter for current folder): ").strip().strip('"')
    if not folder:
        folder = os.getcwd()

    imgs = glob.glob(os.path.join(folder, "*.png")) + \
        glob.glob(os.path.join(folder, "*.jpg")) + \
        glob.glob(os.path.join(folder, "*.jpeg")) + \
        glob.glob(os.path.join(folder, "*.bmp"))

    if not imgs:
        print("No images found in that folder")
        exit()

    print(f"Found {len(imgs)} images — starting batch processing...\n")

    for i, path in enumerate(imgs, 1):
        start = time.time()
        arr = np.array(Image.open(path).convert('RGB'))
        payload_bits = int(arr.size * 0.95)
        secret = ''.join(random.choice('01') for _ in range(payload_bits))

        print(f"[{i}/{len(imgs)}] {os.path.basename(path)} {arr.shape} — {payload_bits//8:,} bytes payload")
        stego, _, _ = embed(path, secret, os.path.join(folder, "STEGO_" + os.path.basename(path)))

        rec = extract(stego, len(secret))
        acc = 100 * sum(a == b for a, b in zip(secret, rec)) / len(secret)
        print(f"    → PSNR {20*np.log10(255/np.sqrt(np.mean((arr.astype(float)-stego.astype(float))**2))):.2f} dB | Extraction {acc:.2f}%\n")
        print(f"    Time: {time.time()-start:.1f}s\n")
