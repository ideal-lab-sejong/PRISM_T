import os
import urllib.request

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "Classical")

DATASETS = [
    "1_ALOI", "2_annthyroid", "3_backdoor", "4_breastw", "5_campaign", "6_cardio",
    "7_Cardiotocography", "8_celeba", "9_census", "10_cover", "11_donors", "12_fault",
    "13_fraud", "14_glass", "15_Hepatitis", "16_http", "17_InternetAds", "18_Ionosphere",
    "19_landsat", "20_letter", "21_Lymphography", "22_magic.gamma", "23_mammography",
    "24_mnist", "25_musk", "26_optdigits", "27_PageBlocks", "28_pendigits", "29_Pima",
    "30_satellite", "31_satimage-2", "32_shuttle", "33_skin", "34_smtp", "35_SpamBase",
    "36_speech", "37_Stamps", "38_thyroid", "39_vertebral", "40_vowels", "41_Waveform",
    "42_WBC", "43_WDBC", "44_Wilt", "45_wine", "46_WPBC", "47_yeast",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    ok, fail = 0, []
    for i, name in enumerate(DATASETS, 1):
        dst = os.path.join(OUT, name + ".npz")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[{i:2d}/47] {name:20s} already present"); ok += 1; continue
        url = f"{BASE}/{name}.npz"
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"[{i:2d}/47] {name:20s} downloaded ({os.path.getsize(dst)/1e3:.0f} KB)"); ok += 1
        except Exception as e:
            print(f"[{i:2d}/47] {name:20s} FAILED: {e}"); fail.append(name)
    print(f"\nDone: {ok}/47 available." + (f"  Failed: {fail}" if fail else ""))
    if fail:
        print("If downloads fail (network/rate-limit), fetch the .npz files manually from")
        print("https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical")
        print(f"and place them in {OUT}")


if __name__ == "__main__":
    main()
