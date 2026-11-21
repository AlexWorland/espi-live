"""Per-camera control profiles and named presets, kept in a settings store.

A profile is everything a camera's controls hold: its Settings (camera
values and every processing and display control) plus its LabVIEW-flow state
(on or off, and the user's own values kept while it is on). Each camera's
profile is saved under a key of its own (profile_key), so a change made to
one camera never reaches another, and a camera that comes back starts with
its own last values. A camera with nothing saved starts from
default_profile, exactly the defaults it always had.

A preset is the same snapshot saved under a name the user chose. It belongs
to no camera: applying it changes only the displayed camera's profile, with
the camera values clamped to that camera's limits (clamp_to_limits).

The store is injected, so nothing here imports Qt: anything with
value(key, default=None), setValue(key, value), remove(key) and allKeys()
works (QSettings in the window, a dictionary in the tests). Entries are JSON
text:

    profiles/<namespace>/<camera>  {"format", "settings", "labview_flow",
                                    "labview_saved"}
    presets/<name, case-folded>    the same, plus "name" and "source_model"

Namespaces: "flir/<serial>" for FLIR cameras, "capture/<vid>-<pid>/<name>"
for video-capture devices (their "video N" serial is only the enumeration
order) and "demo/<serial>" for synthetic cameras, so demo values never land
on a real camera.

Loading is tolerant: unknown keys are ignored (older builds stored palette
and clahe; newer ones may add fields), and a wrong-typed or out-of-range
value falls back to the camera's default and is logged. A broken entry never
raises.
"""

import json
import math
import string
from dataclasses import asdict, dataclass, fields, replace
from urllib.parse import unquote

from espi_live.camera import CAMERA_RANGES
from espi_live.engine import LABVIEW_FLOW, Settings, default_settings, is_analog, own_settings
from espi_live.processing import (MAX_DENOISE_PERCENT, MAX_FRAMES_BACK,
                                  MAX_REFERENCE_FRAMES, SMOOTHING_METHODS, VIEW_MODES)

FORMAT = 1
PROFILE_PREFIX = "profiles/"
PRESET_PREFIX = "presets/"
# Never saved: a camera must never come back recording.
TRANSIENT_FIELDS = frozenset({"record"})
PERSISTED_FIELDS = tuple(field.name for field in fields(Settings)
                         if field.name not in TRANSIENT_FIELDS)
FIELD_TYPES = {name: type(getattr(Settings(), name)) for name in PERSISTED_FIELDS}
# Camera values: (limit node name, label, unit). They are clamped to the
# camera's own limits when a preset is applied.
CAMERA_FIELDS = {
    "exposure_us": ("ExposureTime", "Exposure", " µs"),
    "gain_db": ("Gain", "Gain", " dB"),
    "fps": ("AcquisitionFrameRate", "Frame rate", " fps"),
}
# The values the program can produce for each number, inclusive. Each
# control's own range lies inside (a test checks this); the sensitivity
# floor goes to 255 because Calibrate noise can set it that high.
RANGES = {
    "exposure_us": CAMERA_RANGES["ExposureTime"],
    "gain_db": CAMERA_RANGES["Gain"],
    "fps": CAMERA_RANGES["AcquisitionFrameRate"],
    "noise_reduction": (0, MAX_DENOISE_PERCENT),
    "average": (1, 60),
    "ref_refresh_s": (0, 10),
    "ref_frames": (1, MAX_REFERENCE_FRAMES),
    "smoothing": (0, 31),
    "frames_back": (1, MAX_FRAMES_BACK),
    "outlier_filter": (0, 15),
    "difference_gain": (1, 64),
    "average_multiplier": (0.1, 10),
    "clip_percent": (0, 10),
    "min_range": (0, 255),
    "threshold": (0, 64),
    "steady_range_s": (0, 3),
    "peak_hold_s": (0, 10),
    "normalize_window": (0, 255),
    "gamma": (0.1, 3.0),
    "brightness": (0, 128),
    "display_gain": (1, 8),
    "contrast": (0, 100),
}
CHOICES = {"mode": VIEW_MODES, "smoothing_method": SMOOTHING_METHODS}
KEY_SAFE = frozenset(string.ascii_letters + string.digits + "-_")


@dataclass(frozen=True)
class Profile:
    """One camera's controls: its settings and its LabVIEW-flow state.

    settings never records (record is always False here); labview_saved is
    the user's own value of every LABVIEW_FLOW setting while the LabVIEW
    flow is on, and None while it is off.
    """

    settings: Settings
    labview_saved: dict | None = None

    @property
    def labview_flow(self):
        """Report whether the LabVIEW flow is on.

        Args: None.
        Returns: True while the flow is on.
        """
        return self.labview_saved is not None

    def to_data(self):
        """Describe the profile as plain JSON data.

        Args: None.
        Returns: A dict with "format", "settings" (every persisted field),
            "labview_flow" and "labview_saved" (a dict, or None when off).
        """
        values = asdict(self.settings)
        return {
            "format": FORMAT,
            "settings": {name: values[name] for name in PERSISTED_FIELDS},
            "labview_flow": self.labview_flow,
            "labview_saved": (None if self.labview_saved is None
                              else {key: self.labview_saved[key] for key in LABVIEW_FLOW}),
        }

    def dumps(self):
        """Write the profile as JSON text, the same text for the same values.

        Args: None.
        Returns: A JSON string with sorted keys.
        """
        return json.dumps(self.to_data(), sort_keys=True)


@dataclass(frozen=True)
class PresetEntry:
    """A saved preset as the preset list shows it.

    name is the name the user typed; source_model is the camera model it was
    saved from ("" when unknown).
    """

    name: str
    source_model: str = ""


def default_profile(info, demo=False):
    """Choose the profile a camera with nothing saved starts from.

    Args: info is the camera's CameraInfo; demo is True for a synthetic source.
    Returns: A Profile: engine.default_settings for the camera (exposure
        8000 µs, gain 0 dB, 30 fps), and for an analog camera the LabVIEW
        flow on, keeping its own settings to return to.
    """
    source = None if demo else info
    labview_saved = None
    if is_analog(source):
        own = own_settings(source)
        labview_saved = {key: getattr(own, key) for key in LABVIEW_FLOW}
    return Profile(default_settings(source), labview_saved)


def summary(profile):
    """Describe a profile's camera values and view in one line, for the log.

    Args: profile is a Profile.
    Returns: Text such as "exposure 8000 µs, gain 0 dB, 30 fps, Raw view",
        ending in ", LabVIEW flow" while the flow is on.
    """
    s = profile.settings
    return (f"exposure {s.exposure_us:g} µs, gain {s.gain_db:g} dB, {s.fps:g} fps, "
            f"{s.mode} view" + (", LabVIEW flow" if profile.labview_flow else ""))


def _segment(text):
    """Make one part of a store key safe on every platform's settings store.

    Args: text is a serial, model name or preset name.
    Returns: The text with everything except letters, digits, "-" and "_"
        percent-encoded, so "/", "\\" and "." never split a key.
    """
    return "".join(char if char in KEY_SAFE else
                   "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
                   for char in text)


def profile_key(info, demo=False):
    """Build the stable store key for one camera's profile.

    Args: info is the CameraInfo; demo is True for a synthetic source.
    Returns: "profiles/demo/<serial>" for a synthetic camera,
        "profiles/capture/<vid>-<pid>/<name>" for a video-capture device
        (its serial is only the enumeration order; "usb-unknown" when the
        USB IDs were not reported) or "profiles/flir/<serial>" for a FLIR
        camera.
    """
    if demo:
        parts = ("demo", info.serial)
    elif info.kind == "capture":
        usb = (f"{info.vid:04x}-{info.pid:04x}"
               if info.vid is not None and info.pid is not None else "usb-unknown")
        parts = ("capture", usb, info.model or "unnamed")
    else:
        parts = ("flir", info.serial)
    return PROFILE_PREFIX + "/".join(_segment(part) for part in parts)


def preset_key(name):
    """Build the store key for a named preset.

    Names differ only in letter case or surrounding spaces name the same
    preset, as the Windows registry would treat them anyway.

    Args: name is the preset name.
    Returns: "presets/<encoded name>".
    """
    return PRESET_PREFIX + _segment(name.strip().casefold())


def _note(log, message):
    """Log one message when there is a log.

    Args: log is an AppLog-like object or None; message is the text.
    Returns: None.
    """
    if log is not None:
        log.write(message)


def clean_value(name, value):
    """Check one stored setting against its type and allowed values.

    Args: name is a persisted Settings field; value is what the store held.
    Returns: The value converted to the field's type (an integral float for
        a whole-number field becomes an int, an int for a decimal field a
        float). Raises ValueError when the type is wrong, the number is not
        finite or is out of range, or the text is not one of the choices.
    """
    kind = FIELD_TYPES[name]
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError("not true or false")
    elif kind is str:
        if not isinstance(value, str):
            raise ValueError("not text")
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("not a number")
        if not math.isfinite(value):
            raise ValueError("not a finite number")
        if kind is int:
            if value != int(value):
                raise ValueError("not a whole number")
            value = int(value)
        else:
            value = float(value)
    if name in RANGES:
        low, high = RANGES[name]
        if not low <= value <= high:
            raise ValueError(f"outside {low:g}–{high:g}")
    if name in CHOICES and value not in CHOICES[name]:
        raise ValueError("not one of " + ", ".join(CHOICES[name]))
    return value


def _clean_values(data, base, log, where):
    """Read every known field of a stored settings dict.

    Args: data is the stored dict; base is {field: default value} used for
        missing or invalid fields; log is a log or None; where names the
        entry in log lines.
    Returns: {field: value} for every field of base.
    """
    values = dict(base)
    for name in base:
        if name not in data:
            continue  # saved by a build that did not have this setting yet
        try:
            values[name] = clean_value(name, data[name])
        except ValueError as exc:
            _note(log, f"{where}: {name} = {data[name]!r} is {exc}; "
                       f"using the default {base[name]!r}.")
    return values


def parse_profile(text, defaults, log=None, where="Profile"):
    """Read a stored profile, falling back to the defaults wherever it is wrong.

    Args: text is the stored JSON text (anything else counts as unreadable);
        defaults is the camera's default Profile; log is a log or None;
        where names the entry in log lines.
    Returns: A Profile, or None when the entry is not readable at all.
    """
    try:
        data = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        _note(log, f"{where}: the saved entry cannot be read; using the defaults.")
        return None
    stored = data.get("settings")
    if not isinstance(stored, dict):
        _note(log, f"{where}: no saved settings; using the defaults.")
        stored = {}
    base = asdict(defaults.settings)
    values = _clean_values(stored, {name: base[name] for name in PERSISTED_FIELDS},
                           log, where)
    settings = replace(defaults.settings, **values, record=False)
    flow = data.get("labview_flow", defaults.labview_flow)
    if not isinstance(flow, bool):
        _note(log, f"{where}: labview_flow = {flow!r} is not true or false; "
                   "using the default.")
        flow = defaults.labview_flow
    if not flow:
        return Profile(settings, None)
    fallback = (defaults.labview_saved if defaults.labview_saved is not None
                else {key: base[key] for key in LABVIEW_FLOW})
    saved = data.get("labview_saved")
    if not isinstance(saved, dict):
        _note(log, f"{where}: the settings kept for the LabVIEW flow are missing; "
                   "using the defaults.")
        saved = {}
    return Profile(settings, _clean_values(saved, fallback, log, where))


def clamp_to_limits(settings, limits):
    """Fit the camera values of a Settings into one camera's limits.

    Args: settings is a Settings; limits is {node name: (low, high)}, as a
        "camera_limits" event reports them; missing names are left alone.
    Returns: (settings, notes): the clamped Settings and one sentence per
        value that changed, for the log.
    """
    changes = {}
    notes = []
    for name, (node, label, unit) in CAMERA_FIELDS.items():
        if node not in limits:
            continue
        low, high = limits[node]
        value = getattr(settings, name)
        used = min(max(value, low), high)
        if used != value:
            changes[name] = used
            notes.append(f"{label} {value:g}{unit} is outside this camera's range "
                         f"({low:g}–{high:g}{unit}); using {used:g}{unit}.")
    return replace(settings, **changes), notes


class ProfileStore:
    """Read and write camera profiles and named presets in an injected store."""

    def __init__(self, store, log=None):
        """Wrap a settings store.

        Args: store has value, setValue, remove and allKeys (QSettings, or a
            test double); log is an AppLog-like object or None.
        Returns: None.
        """
        self.store = store
        self.log = log
        self._written = {}  # key -> the JSON text the store holds (read or written)
        self._seen = {}  # key -> the JSON text seen by the last save_when_settled

    # ----- camera profiles ---------------------------------------------

    def load_profile(self, key, defaults):
        """Read one camera's saved profile.

        Args: key is its profile_key; defaults is its default Profile, used
            for anything missing or invalid.
        Returns: A Profile, or None when nothing (readable) is saved.
        """
        text = self.store.value(key)
        if text is None:
            return None
        self._written[key] = text
        return parse_profile(text, defaults, self.log, f"Saved settings {key}")

    def track(self, key, profile):
        """Note a profile in force, so it is written only after it changes.

        A profile read with load_profile keeps the text actually stored, so
        an entry with values that were replaced is rewritten clean.

        Args: key is the profile_key; profile is the Profile in force.
        Returns: None.
        """
        self._written.setdefault(key, profile.dumps())

    def save_profile(self, key, profile):
        """Write a camera's profile now, when it differs from what is stored.

        Args: key is the profile_key; profile is the Profile to keep.
        Returns: True when the store was written.
        """
        text = profile.dumps()
        self._seen[key] = text
        if self._written.get(key) == text:
            return False
        self.store.setValue(key, text)
        self._written[key] = text
        return True

    def save_when_settled(self, key, profile):
        """Write a profile once it has stopped changing.

        Called from a timer: a value that changed since the previous call
        (a slider being dragged) waits for the next one, so a drag writes
        once, after it ends.

        Args: key is the profile_key; profile is the Profile now in force.
        Returns: True when the store was written.
        """
        text = profile.dumps()
        settled = self._seen.get(key) == text
        self._seen[key] = text
        return settled and self.save_profile(key, profile)

    def forget_profile(self, key, defaults):
        """Remove a camera's saved profile; it starts from the defaults next time.

        Args: key is the profile_key; defaults is the default Profile now in
            force for it.
        Returns: None.
        """
        self.store.remove(key)
        self._written[key] = self._seen[key] = defaults.dumps()

    def sync(self):
        """Write the store's pending changes to disk now, when it can.

        Args: None.
        Returns: None; QSettings has sync(), and a store without one is
            left alone.
        """
        sync = getattr(self.store, "sync", None)
        if sync is not None:
            sync()

    # ----- named presets ------------------------------------------------

    def _read_preset(self, key):
        """Read a stored preset's JSON object.

        Args: key is a preset key.
        Returns: The dict, or None when nothing readable is stored.
        """
        text = self.store.value(key)
        try:
            data = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            data = None
        return data if isinstance(data, dict) else None

    def presets(self):
        """List the saved presets, sorted by name.

        Args: None.
        Returns: A list of PresetEntry; unreadable entries are left out and
            logged.
        """
        entries = []
        for key in self.store.allKeys():
            if not key.startswith(PRESET_PREFIX):
                continue
            data = self._read_preset(key)
            if data is None:
                _note(self.log, f"Preset {key}: the saved entry cannot be read; "
                                "it is not listed.")
                continue
            name = data.get("name")
            if not isinstance(name, str) or not name.strip():
                name = unquote(key[len(PRESET_PREFIX):])
            source = data.get("source_model")
            entries.append(PresetEntry(name, source if isinstance(source, str) else ""))
        return sorted(entries, key=lambda entry: entry.name.casefold())

    def find_preset(self, name):
        """Look up a preset by name, ignoring case and surrounding spaces.

        Args: name is the name to look for.
        Returns: The PresetEntry of the stored preset, or None.
        """
        key = preset_key(name)
        return next((entry for entry in self.presets()
                     if preset_key(entry.name) == key), None)

    def load_preset(self, name, defaults):
        """Read a preset's profile for one camera.

        Args: name is the preset name; defaults is the target camera's
            default Profile, used for anything missing or invalid.
        Returns: A Profile, or None when the preset is missing or unreadable.
        """
        text = self.store.value(preset_key(name))
        if text is None:
            return None
        return parse_profile(text, defaults, self.log, f"Preset {name}")

    def save_preset(self, name, profile, source_model=""):
        """Save a profile as a named preset, replacing one with the same name.

        Args: name is the preset name (stripped); profile is the Profile to
            keep; source_model names the camera it came from.
        Returns: None.
        """
        data = profile.to_data()
        data.update(name=name.strip(), source_model=source_model)
        self.store.setValue(preset_key(name), json.dumps(data, sort_keys=True))

    def delete_preset(self, name):
        """Remove a named preset.

        Args: name is the preset name.
        Returns: None.
        """
        self.store.remove(preset_key(name))
