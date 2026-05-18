import argparse
import logging
import sys

from upmixer.config import UpmixConfig
from upmixer.formats import INPUT_FORMAT_MAP
from upmixer.pipeline import UpmixPipeline
from upmixer.profiles import PROFILES, DeliveryProfile
from upmixer.separation.separator import DEFAULT_MODEL

_INPUT_FORMAT_CHOICES = sorted(INPUT_FORMAT_MAP.keys())
_OUTPUT_FORMAT_CHOICES = ["5.1", "7.1", "5.1.2", "5.1.4", "7.1.2", "7.1.4"]
_PROFILE_CHOICES = sorted(PROFILES.keys())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Universal multichannel audio upmixer. "
            "Upmix mono, stereo, or any surround format to a higher channel layout. "
            "Supported inputs: mono, stereo, 5.0, 5.1, 7.1, 5.1.2, 5.1.4, 7.1.2."
        )
    )
    parser.add_argument(
        "input",
        help="Input audio file (WAV/FLAC)",
    )
    parser.add_argument("output", help="Output multichannel audio file")
    parser.add_argument(
        "--format",
        choices=_OUTPUT_FORMAT_CHOICES,
        default="5.1",
        help="Output channel format (default: 5.1)",
    )
    parser.add_argument(
        "--profile",
        choices=_PROFILE_CHOICES,
        default=None,
        metavar="PROFILE",
        help=(
            "Delivery target profile. Sets loudness target, True Peak ceiling, "
            "sample rate, bit depth, and output container to match the platform spec. "
            "Individual flags (--loudness-target, --output-sample-rate, etc.) override "
            "the profile. "
            f"Choices: {', '.join(_PROFILE_CHOICES)}. "
            "Use --profile-info to print full spec for each profile."
        ),
    )
    parser.add_argument(
        "--profile-info",
        action="store_true",
        help="Print delivery spec for all built-in profiles and exit.",
    )
    parser.add_argument(
        "--input-format",
        choices=_INPUT_FORMAT_CHOICES,
        default=None,
        metavar="FMT",
        help=(
            "Override auto-detected input format. "
            f"Choices: {', '.join(_INPUT_FORMAT_CHOICES)}. "
            "Required when channel count is ambiguous (8ch = 7.1 or 5.1.2; 10ch = 7.1.2 or 5.1.4)."
        ),
    )

    # --- Processing mode ---
    parser.add_argument(
        "--mode",
        choices=["realtime", "stem"],
        default="realtime",
        help=(
            "Processing mode. "
            "'realtime' (default): coherence-based STFT pipeline, works on any input, low latency. "
            "'stem': source-separation pipeline — separates instruments then places each in 3D space. "
            "Requires: pip install 'audio-separator[cpu]'. Only supports mono/stereo input."
        ),
    )
    parser.add_argument(
        "--stem-model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"audio-separator model for stem mode (default: {DEFAULT_MODEL}). "
            "4-stem models (drums/bass/vocals/other) give best spatial placement. "
            "Models are auto-downloaded on first use."
        ),
    )
    parser.add_argument(
        "--stem-model-dir",
        default=None,
        metavar="DIR",
        help="Directory to cache downloaded separation models (default: ~/.cache/upmixer-models).",
    )

    # --- Gain controls ---
    parser.add_argument(
        "--center-gain", type=float, default=None,
        help="Center channel output gain (default: 0.85)",
    )
    parser.add_argument(
        "--surround-gain", type=float, default=None,
        help="Side surround channel gain (default: 0.6)",
    )
    parser.add_argument(
        "--back-gain", type=float, default=None,
        help="Rear back channel gain for 7.1 formats (default: 0.55)",
    )
    parser.add_argument(
        "--height-gain", type=float, default=None,
        help="Height channel gain for Atmos formats (default: 0.55)",
    )
    parser.add_argument(
        "--lfe-gain", type=float, default=None,
        help="LFE channel gain (default: 0.5)",
    )

    # --- Center extraction ---
    parser.add_argument(
        "--center-extraction-gain", type=float, default=None,
        help="How much mid signal goes to center channel (default: 0.85)",
    )
    parser.add_argument(
        "--center-attenuation", type=float, default=None,
        help="How much center-panned content is attenuated from FL/FR (default: 0.5)",
    )

    # --- LFE ---
    parser.add_argument(
        "--lfe-cutoff", type=float, default=None, metavar="HZ",
        help="LFE low-pass cutoff frequency in Hz (default: 120)",
    )

    # --- Height EQ ---
    parser.add_argument(
        "--height-low-rolloff-gain", type=float, default=None,
        help="Sub-bass gain for height channels, 0=full rolloff 1=flat (default: 0.15)",
    )
    parser.add_argument(
        "--height-high-shelf-gain", type=float, default=None,
        help="High-frequency presence boost for height channels, >1.0=lift (default: 1.5)",
    )

    # --- STFT / processing ---
    parser.add_argument("--fft-size", type=int, default=None, help="STFT window size")
    parser.add_argument(
        "--no-auto-fft", action="store_true",
        help="Disable automatic FFT size scaling for high sample rates",
    )
    parser.add_argument(
        "--block-size", type=int, default=None,
        help="Streaming block size in samples (default: 4096)",
    )

    # --- Output ---
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable output energy normalization",
    )
    parser.add_argument(
        "--content-mix-strength", type=float, default=None, metavar="S",
        help="Content-aware mixing strength 0.0–1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--no-loudness-normalize", action="store_true",
        help=(
            "Disable ITU-R BS.1770-4 loudness normalization "
            "(Dolby DEE compliance, default: enabled)"
        ),
    )
    parser.add_argument(
        "--loudness-target", type=float, default=None, metavar="LKFS",
        help="Target integrated loudness in LKFS (default: -18.0, Dolby Atmos Music Delivery Playbook)",
    )
    parser.add_argument(
        "--output-type",
        choices=["wav", "adm-bwf"],
        default=None,
        help=(
            "Output file format. 'wav' = standard multichannel WAV. "
            "'adm-bwf' = Broadcast Wave with ITU-R BS.2076-2 ADM metadata "
            "for Logic Pro, DaVinci Resolve, Pro Tools, etc. "
            "Default: determined by --profile, or 'wav' if no profile."
        ),
    )
    parser.add_argument(
        "--output-subtype",
        choices=["PCM_16", "PCM_24", "PCM_32"],
        default=None,
        help="Output bit depth (default: PCM_24)",
    )
    parser.add_argument(
        "--output-sample-rate", type=int, default=None, metavar="HZ",
        help="Resample output to this sample rate (e.g. 48000, 96000). Default: same as input.",
    )

    # --- Preview mode ---
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Process a short excerpt instead of the full file. "
            "Useful for quickly auditing upmix settings before a full run. "
            "Default window: 30 s from the middle of the track."
        ),
    )
    parser.add_argument(
        "--preview-duration",
        type=float,
        default=None,
        metavar="S",
        help="Preview window length in seconds (default: 30).",
    )
    parser.add_argument(
        "--preview-start",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Preview start time in seconds from the beginning of the file. "
            "Default: auto-center (middle of track minus half the window)."
        ),
    )

    # --- Verbosity / output format ---
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except warnings and errors.",
    )
    verbosity.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging (includes audio-separator internals).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print a JSON summary of the result to stdout when done. "
            "Implies --quiet (log output goes to stderr only at WARNING level)."
        ),
    )

    # --profile-info handled before full parse (input/output are required positionals)
    if "--profile-info" in sys.argv:
        for key, p in sorted(PROFILES.items()):
            print(f"\n{'─' * 60}")
            print(f"  {p.display_name}  (--profile {p.name})")
            print(f"{'─' * 60}")
            print(f"  Loudness target : {p.loudness_target_lkfs:+.1f} LKFS")
            print(f"  True Peak ceil  : {p.loudness_max_tp:+.1f} dBTP")
            print(f"  Sample rate     : {p.sample_rate // 1000} kHz")
            print(f"  Bit depth       : {p.bit_depth}-bit")
            print(f"  Output type     : {p.output_type}")
            print(f"  Notes           : {p.codec_note}")
        print()
        sys.exit(0)

    args = parser.parse_args()

    # Configure logging — all upmixer output goes to stderr so stdout is clean
    # for --json output.
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet or args.json:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stderr,
    )

    config = UpmixConfig(output_format=args.format)

    # ── Apply delivery profile (sets platform-specific defaults) ──────────────
    # Individual CLI flags below override the profile where specified.
    if args.profile is not None:
        profile: DeliveryProfile = PROFILES[args.profile]
        config.loudness_target_lkfs = profile.loudness_target_lkfs
        config.loudness_max_tp = profile.loudness_max_tp
        config.output_subtype = profile.output_subtype
        # Only set sample rate / output type from profile if user didn't explicitly
        # pass the corresponding flag (checked later; store profile values as
        # provisional defaults by pre-applying them now — explicit flags win below).
        config.output_type = profile.output_type
        if args.output_sample_rate is None:
            config.output_sample_rate = profile.sample_rate
        import logging as _l
        _l.getLogger("upmixer").info(
            "  Profile: %s — %+.1f LKFS / %+.1f dBTP / %d kHz / %s / %s",
            profile.display_name,
            profile.loudness_target_lkfs,
            profile.loudness_max_tp,
            profile.sample_rate // 1000,
            profile.output_subtype,
            profile.output_type,
        )

    if args.center_gain is not None:
        config.center_gain = args.center_gain
    if args.surround_gain is not None:
        config.surround_gain = args.surround_gain
    if args.back_gain is not None:
        config.back_gain = args.back_gain
    if args.height_gain is not None:
        config.height_gain = args.height_gain
    if args.lfe_gain is not None:
        config.lfe_gain = args.lfe_gain
    if args.center_extraction_gain is not None:
        config.center_extraction_gain = args.center_extraction_gain
    if args.center_attenuation is not None:
        config.center_attenuation = args.center_attenuation
    if args.lfe_cutoff is not None:
        config.lfe_cutoff_hz = args.lfe_cutoff
    if args.height_low_rolloff_gain is not None:
        config.height_low_rolloff_gain = args.height_low_rolloff_gain
    if args.height_high_shelf_gain is not None:
        config.height_high_shelf_gain = args.height_high_shelf_gain
    if args.fft_size is not None:
        config.fft_size = args.fft_size
        config.hop_size = args.fft_size // 4
    if args.no_auto_fft:
        config.auto_fft_size = False
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.no_normalize:
        config.normalize_output = False
    if args.content_mix_strength is not None:
        config.content_mix_strength = max(0.0, min(1.0, args.content_mix_strength))
    if args.no_loudness_normalize:
        config.loudness_normalize = False
    if args.loudness_target is not None:
        config.loudness_target_lkfs = args.loudness_target
    # Explicit --output-type overrides profile; fall back to "wav" if neither set
    if args.output_type is not None:
        config.output_type = args.output_type
    elif args.profile is None:
        config.output_type = "wav"
    if args.output_subtype is not None:
        config.output_subtype = args.output_subtype
    if args.output_sample_rate is not None:
        config.output_sample_rate = args.output_sample_rate
    if args.preview:
        config.preview = True
    if args.preview_duration is not None:
        config.preview_duration_s = args.preview_duration
    if args.preview_start is not None:
        config.preview_start_s = args.preview_start

    if args.mode == "stem":
        from upmixer.separation.stem_pipeline import StemUpmixPipeline
        stem_pipeline = StemUpmixPipeline(
            config=config,
            model=args.stem_model,
            model_dir=args.stem_model_dir,
        )
        result = stem_pipeline.process_file(
            args.input, args.output, input_format_override=args.input_format
        )
    else:
        pipeline = UpmixPipeline(config)
        result = pipeline.process_file(
            args.input, args.output, input_format_override=args.input_format
        )

    if args.json:
        print(result.to_json())


if __name__ == "__main__":
    main()
