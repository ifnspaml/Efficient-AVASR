import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

_ITUT_LEVEL_METER = None
NOISE_METHOD_RMS = "rms"
NOISE_METHOD_ITUT = "itut"
NOISE_METHOD_P56 = "p56"
NOISE_METHODS = {NOISE_METHOD_RMS, NOISE_METHOD_ITUT, NOISE_METHOD_P56}


def _find_itut_package_root():
    path = Path(__file__).resolve().parent
    for _ in range(6):
        if (path / "itut_p56_noise_addition").is_dir():
            return path
        path = path.parent
    raise ImportError(
        "noise_method='itut' requires the sibling package itut_p56_noise_addition "
        "(expected under a parent of avhubert/)"
    )


def select_noise(noise_wavs):
    rand_indexes = np.random.randint(0, len(noise_wavs), size=1)
    noise_wav = []
    for x in rand_indexes:
        noise_wav.append(wavfile.read(noise_wavs[x])[1].astype(np.float32))
    return noise_wav[0]


def _resolve_noise_snr(noise_snr):
    if type(noise_snr) == int or type(noise_snr) == float:
        return noise_snr
    if type(noise_snr) == tuple:
        return np.random.randint(noise_snr[0], noise_snr[1] + 1)
    raise TypeError(f"Unsupported noise_snr type: {type(noise_snr)}")


def _match_noise_length_like_current_code(clean_wav, noise_wav):
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav) / len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start : start + len(clean_wav)]
    return noise_wav


def _avoid_int16_clipping(mixed):
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * reduction_rate
    return mixed


def _get_itut_noise_tools():
    itut_root = _find_itut_package_root()
    if str(itut_root) not in sys.path:
        sys.path.append(str(itut_root))
    try:
        from itut_p56_noise_addition import LevelMeter, add_noise_at_snr
    except ImportError as exc:
        raise ImportError(
            "noise_method='itut' requires the sibling package "
            f"{itut_root / 'itut_p56_noise_addition'}"
        ) from exc
    return LevelMeter, add_noise_at_snr


def _get_itut_level_meter():
    global _ITUT_LEVEL_METER
    if _ITUT_LEVEL_METER is None:
        level_meter_cls, _ = _get_itut_noise_tools()
        _ITUT_LEVEL_METER = level_meter_cls()
    return _ITUT_LEVEL_METER


def _float_pcm_to_int16(signal):
    signal = np.asarray(signal, dtype=np.float64)
    peak = np.max(np.abs(signal))
    if peak > 0:
        max_float = np.nextafter(1.0, 0.0)
        if peak > max_float:
            signal = signal * (max_float / peak)
    signal = np.clip(signal, -1.0, np.nextafter(1.0, 0.0))
    return (signal * 32768.0).astype(np.int16)


def add_noise_rms(clean_wav, noise_wavs, noise_snr=0):
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = select_noise(noise_wavs)
    snr = _resolve_noise_snr(noise_snr)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    noise_wav = _match_noise_length_like_current_code(clean_wav, noise_wav)
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr / 20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav
    mixed = _avoid_int16_clipping(mixed)
    mixed = mixed.astype(np.int16)
    return mixed


def add_noise_itut(clean_wav, noise_wavs, noise_snr=0, speech_level_dbov=-26):
    _, add_noise_at_snr = _get_itut_noise_tools()
    clean_wav = clean_wav.astype(np.float32).reshape(-1)
    noise_wav = select_noise(noise_wavs).astype(np.float32).reshape(-1)
    snr = _resolve_noise_snr(noise_snr)
    noise_wav = _match_noise_length_like_current_code(clean_wav, noise_wav)

    clean_float = clean_wav.astype(np.float64) / 32768.0
    noise_float = noise_wav.astype(np.float64) / 32768.0
    result = add_noise_at_snr(
        clean_float,
        noise_float,
        snr_db=snr,
        fs=16000,
        speech_level_dbov=speech_level_dbov,
        normalize_speech=True,
        meter=_get_itut_level_meter(),
    )
    return _float_pcm_to_int16(result.mixture)


def require_noise_method_available(noise_method: str) -> None:
    """Fail fast when a requested noise mixer cannot be imported.

    RMS needs no extra package. ITU-T / P.56 require the sibling
    ``itut_p56_noise_addition`` package and its ``LevelMeter`` /
    ``add_noise_at_snr`` entry points. This helper does not change mixing
    mathematics; it only verifies importability before decoding starts.
    """

    method = str(noise_method).lower()
    if method == NOISE_METHOD_RMS:
        return
    if method in {NOISE_METHOD_ITUT, NOISE_METHOD_P56}:
        level_meter_cls, add_noise_at_snr = _get_itut_noise_tools()
        if level_meter_cls is None or add_noise_at_snr is None:
            raise ImportError(
                f"noise_method={method!r} resolved incomplete ITU-T imports"
            )
        return
    raise ValueError(
        f"noise_method must be one of {sorted(NOISE_METHODS)}, got {noise_method!r}"
    )


def add_noise(clean_wav, noise_wavs, noise_snr=0, noise_method=NOISE_METHOD_RMS):
    noise_method = noise_method.lower()
    if noise_method == NOISE_METHOD_RMS:
        return add_noise_rms(clean_wav, noise_wavs, noise_snr=noise_snr)
    if noise_method in {NOISE_METHOD_ITUT, NOISE_METHOD_P56}:
        return add_noise_itut(clean_wav, noise_wavs, noise_snr=noise_snr)
    raise ValueError(f"noise_method must be one of {sorted(NOISE_METHODS)}, got {noise_method!r}")
