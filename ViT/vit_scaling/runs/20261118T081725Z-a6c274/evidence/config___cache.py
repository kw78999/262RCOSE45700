from contextlib import contextmanager

from furiosa.torch._C.config import cache, env


def set_eager_cache_size(
    num_exported_program: int = env.eager_num_exported_program(),
    num_compiled: int = env.eager_num_compiled(),
    num_load: int = env.eager_num_load(),
):
    """
    Set the eager cache size. num_load is for the cache of loaded modules, it may cause Furiosa OOM
    if the cached modules are too large, or too many

    Args:
        num_exported_program (int): The maximum number of exported programs to cache.
        num_compiled (int): The maximum number of compiled modules to cache.
        num_load (int): The maximum number of loaded modules to cache.
    """
    from furiosa.torch._backend.aten import (
        set_build_exported_program_cache,
        set_compile_cache,
        set_load_cache,
    )

    set_build_exported_program_cache(num_exported_program)
    set_compile_cache(num_compiled)
    set_load_cache(num_load)


@contextmanager
def cache_context(
    root_dir=cache.CacheConfig.default_cache_dir(),
    summary: bool = False,
    ensure: bool = False,
    failure_link_dir: str | None = None,
):
    """
    A context manager that scopes the compiler cache configuration.

    Args:
        root_dir (Optional[str]): The cache directory to use within this scope.
            Defaults to the default cache directory (see `CacheConfig.default_cache_dir`).
            Pass `None` explicitly to disable caching within this scope.
        summary (bool): Whether to dump summary after compilation.
        ensure (bool): Whether to ensure the cache directory exists before compilation.
        failure_link_dir (Optional[str]): The directory to link failed compilation artifacts for debugging.
            If None, no links will be created. If provided, the directory will be created if it does not exist.
    """
    cache_config = cache.CacheConfig(
        root_dir=root_dir,
        summary=summary,
        ensure=ensure,
        failure_link_dir=failure_link_dir,
    )
    prev = cache.set_cache_config(cache_config)
    try:
        yield cache_config
    finally:
        cache.set_cache_config(prev)
