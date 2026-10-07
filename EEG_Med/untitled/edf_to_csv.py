import csv
import sys
from pathlib import Path

import mne


def save_csv(output_path, data, channel_names, sfreq):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + channel_names)

        for i in range(data.shape[1]):
            row = [f"{i / sfreq:.6f}"]
            row.extend(f"{value:.8f}" for value in data[:, i])
            writer.writerow(row)


def convert_edf(input_path, output_path):
    raw = mne.io.read_raw_edf(input_path, preload=True, verbose="ERROR")
    eeg = raw.copy().pick("eeg")

    if not eeg.ch_names:
        raise RuntimeError("У EDF-файлі не знайдено EEG-каналів.")

    # MNE зберігає EEG у вольтах; експортуємо в мікровольтах.
    data = eeg.get_data() * 1_000_000
    save_csv(output_path, data, eeg.ch_names, float(eeg.info["sfreq"]))

    print("Тип: EEG")
    print(f"Частота дискретизації: {eeg.info['sfreq']} Hz")
    print("Одиниці: µV")


def convert_snirf(input_path, output_path):
    from mne.preprocessing.nirs import optical_density, beer_lambert_law

    raw = mne.io.read_raw_snirf(input_path, preload=True, verbose="ERROR")
    optical = optical_density(raw)
    hb = beer_lambert_law(optical, ppf=6.0)

    types = hb.get_channel_types()
    picks = [i for i, kind in enumerate(types) if kind in ("hbo", "hbr")]

    if not picks:
        raise RuntimeError("У файлі не знайдено каналів HbO/HbR.")

    data = hb.get_data(picks=picks) * 1_000_000
    names = [hb.ch_names[i] for i in picks]
    sfreq = float(hb.info["sfreq"])

    save_csv(output_path, data, names, sfreq)

    print("Тип: fNIRS")
    print(f"Частота дискретизації: {sfreq} Hz")
    print(f"Каналів HbO/HbR: {len(names)}")
    print("Одиниці: µM")


def main():
    if len(sys.argv) != 3:
        print("Usage: edf_to_csv.py input.edf-or-snirf output.csv")
        sys.exit(2)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    extension = Path(input_path).suffix.lower()

    try:
        if extension == ".edf":
            convert_edf(input_path, output_path)
        elif extension == ".snirf":
            convert_snirf(input_path, output_path)
        else:
            raise ValueError(
                f"Непідтримуваний формат {extension}; очікується .edf або .snirf."
            )

        print(f"CSV збережено: {output_path}")

    except Exception as error:
        print(f"Conversion error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
