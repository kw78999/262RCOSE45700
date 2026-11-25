import contextlib
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap

from furiosa.torch import config
from furiosa.torch._C import device, ir

from .runnable import Runnable


logger = logging.getLogger("furiosa_torch")

_TEMP_CRATE_CARGO_TOML = textwrap.dedent(
    """\
    [package]
    name = "furiosa_torch_visa_tmp"
    version = "0.1.0"
    edition = "2024"

    [package.metadata.furiosa-opt]

    [dependencies]
    furiosa-opt-std = "0.5.1"

    [workspace]
    """
)

# Without this, a temp directory outside the repo doesn't inherit its root toolchain
# override and would resolve to the default toolchain instead (too old for `edition = "2024"`).
_TEMP_CRATE_RUST_TOOLCHAIN_TOML = textwrap.dedent(
    """\
    [toolchain]
    channel = "nightly-2026-05-01"
    """
)


def _check_compatible_furiosa_opt(cargo_path: str) -> None:
    if shutil.which("cargo-furiosa-opt") is None:
        raise RuntimeError("cargo-furiosa-opt not found")

    args = [
        cargo_path,
        "furiosa-opt",
        "--version",
    ]

    try:
        result = subprocess.run(
            args=args,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Compiler timed out after 10 seconds: {e}") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr
        err = RuntimeError("Compiler returned non-zero status")
        err.add_note(f"stderr:\n{stderr}")
        raise err from e

    lines = result.stdout.splitlines()
    version_string = lines[0].removeprefix("cargo-furiosa-opt ") if lines else ""

    # Build metadata does not affect precedence. Since both bounds are stable
    # releases, only the presence of a prerelease suffix matters when the core
    # version equals a bound.
    precedence_version = version_string.partition("+")[0]
    core_version, prerelease_separator, _ = precedence_version.partition("-")
    if prerelease_separator:
        raise RuntimeError(
            f"Found pre-release version of furiosa-opt {version_string}, which is not supporeted. "
            f"Set {config.env.FURIOSA_TORCH_SKIP_VERSION_CHECK}=1 to skip the version check"
        )

    try:
        version = tuple(int(component) for component in core_version.split("."))
    except ValueError:
        version = ()

    lower_bound = (0, 5, 0)
    upper_bound = (0, 6, 0)
    is_compatible = (
        len(version) == 3 and version >= lower_bound and version < upper_bound
    )
    if not is_compatible:
        raise RuntimeError(
            f"Incompatible furiosa-opt version {version_string}; "
            "expected it to be compatible with 0.5.0. "
            f"Set {config.env.FURIOSA_TORCH_SKIP_VERSION_CHECK}=1 to skip the version check"
        )


def compile_from_visa(crate_root: os.PathLike, device_function: str) -> Runnable:
    """
    Compile a vISA device function from a Cargo crate directory into a `Runnable`.

    Args:
        crate_root (os.PathLike): Path to the crate containing the device function,
            e.g. a crate with a `#[device(..)]`-annotated `pub fn` in `src/lib.rs`.
        device_function (str): Name of the `#[device(..)]` function to compile.

    Returns:
        Runnable: The compiled device function, callable like a `torch.nn.Module`.
    """
    cargo_path = shutil.which("cargo")
    if cargo_path is None:
        raise RuntimeError("cargo not found")

    if not config.env.is_version_check_skipped():
        _check_compatible_furiosa_opt(cargo_path)

    # Normalize path before hashing
    crate_root = Path(crate_root).resolve()

    cache_config = config.get_cache_config()
    if cache_config.root_dir:
        root_dir = Path(cache_config.root_dir)
        crate_root_hash = hashlib.sha256(bytes(crate_root)).hexdigest()
        # Two-depth directory structure to reduce cache directory clutter
        target_dir = root_dir / "visa" / crate_root_hash[:2] / crate_root_hash[2:]
        target_dir.mkdir(parents=True, exist_ok=True)
        tempdir_handle = contextlib.nullcontext(str(target_dir))
    else:
        tempdir_handle = tempfile.TemporaryDirectory()

    with tempdir_handle as target_dir, tempfile.TemporaryDirectory() as out_dir:
        logger.debug(
            f"Compiling vISA device function `{device_function}` of `{crate_root}`; "
            f"target_dir = {target_dir}"
        )

        args = [
            cargo_path,
            "furiosa-opt",
            "compile",
            "--exact",
            device_function,
        ]
        env = os.environ | {
            "CARGO_TARGET_DIR": target_dir,
            "FURIOSA_OPT_OUT_DIR": out_dir,
        }

        try:
            result = subprocess.run(
                args=args,
                cwd=crate_root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Compiler timed out after 180 seconds: {e}") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            err = RuntimeError("Compiler returned non-zero status")
            err.add_note(f"stderr:\n{stderr}")
            raise err from e

        edfs = list(Path(out_dir).glob("**/*.edf"))
        logger.debug(f"Compiled successfully; edfs = {edfs}")
        if not edfs:
            err = RuntimeError("Compiler didn't produce any binary")
            err.add_note(f"stderr:\n{result.stderr}")
            raise err
        if len(edfs) != 1:
            err = RuntimeError("Compiler produced more than one binary")
            err.add_note(f"stderr:\n{result.stderr}")
            raise err

        edf_path = edfs[0]
        runnable = ir.Runnable.from_serialized_edf(edf_path.read_bytes())

        # NOTE: This won't do anything until furiosa-opt actually produces Runnable and sets
        # `device_meta` properly.
        current_num_fusion = device.get_fusion().num_fusion
        if current_num_fusion != runnable.expected_num_fusion:
            logger.warning(
                "Expected device fusion of compiled Runnable doesn't match current device fusion "
                f"({runnable.expected_num_fusion} != {current_num_fusion})"
            )

        return Runnable(runnable)


def compile_from_visa_str(source: str, device_function: str) -> Runnable:
    """
    Compile a vISA device function given as source code, without an on-disk crate.

    `source` is materialized as `src/lib.rs` of a throwaway crate in a temporary
    directory, then compiled the same way as `compile_from_visa`.

    Args:
        source (str): The crate's `lib.rs` source, containing a
            `#[device(..)]`-annotated `pub fn` named `device_function`.
        device_function (str): Name of the `#[device(..)]` function to compile.

    Returns:
        Runnable: The compiled device function, callable like a `torch.nn.Module`.
    """
    with tempfile.TemporaryDirectory() as crate_root:
        crate_root = Path(crate_root)
        (crate_root / "src").mkdir()
        (crate_root / "src" / "lib.rs").write_text(source)
        # Marker file `cargo furiosa-opt` requires in the crate with `#[device]` functions.
        (crate_root / "src" / "furiosa-opt.tag").touch()
        (crate_root / "Cargo.toml").write_text(_TEMP_CRATE_CARGO_TOML)
        (crate_root / "rust-toolchain.toml").write_text(_TEMP_CRATE_RUST_TOOLCHAIN_TOML)

        return compile_from_visa(crate_root, device_function)
