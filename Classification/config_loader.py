import configparser
import os

_config = None


def get_config() -> configparser.ConfigParser:
    """Returns the singleton ConfigParser instance loaded from config.ini next to this file."""
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        _config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    return _config


def get_schema() -> str:
    """Returns the database schema name from config.ini."""
    return get_config().get('Database', 'schema')


def get_layer_name(key: str) -> str:
    """Returns a QGIS layer name from the [Layers] section of config.ini."""
    return get_config().get('Layers', key)


def get_option(key: str, fallback: str = '') -> str:
    """Returns a value from the [Options] section of config.ini."""
    return get_config().get('Options', key, fallback=fallback)
