"""Checks for espi_live.profiles: per-camera profiles and named presets.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests

The store is a dictionary, so nothing touches the real QSettings.
"""

import json
import unittest
from dataclasses import asdict, replace

from espi_live import profiles
from espi_live.camera import CameraInfo
from espi_live.engine import LABVIEW_FLOW, Settings
from list_log import ListLog

DEMO = CameraInfo("DEMO1", "Synthetic camera")
FLIR = CameraInfo("12345678", "Blackfly S BFS-U3-63S4M")
ANALOG = CameraInfo("video 1", "Hauppauge USB-Live2", kind="capture", vid=0x2040,
                    pid=0xC200)
WEBCAM = CameraInfo("video 2", "Webcam C270", kind="capture", vid=0x046D, pid=0x0825)


class DictStore(dict):
    """A dictionary with the QSettings calls ProfileStore uses."""

    def value(self, key, default=None):
        """Read a stored value.

        Args: key is the setting name; default is returned when unset.
        Returns: The stored value or default.
        """
        return self.get(key, default)

    def setValue(self, key, value):
        """Store a value.

        Args: key is the setting name; value is what to keep.
        Returns: None.
        """
        self[key] = value

    def remove(self, key):
        """Forget a value.

        Args: key is the setting name.
        Returns: None.
        """
        self.pop(key, None)

    def allKeys(self):
        """List every stored key.

        Args: None.
        Returns: A list of setting names.
        """
        return list(self)


class SyncingStore(DictStore):
    """A store that also counts sync() calls, as QSettings has one."""

    syncs = 0

    def sync(self):
        """Count one request to write to disk.

        Args: None.
        Returns: None.
        """
        self.syncs += 1


def stored(store, key):
    """Decode one stored JSON entry.

    Args: store is a DictStore; key is the entry.
    Returns: The decoded object.
    """
    return json.loads(store[key])


class FieldTests(unittest.TestCase):
    """Which Settings fields a profile keeps."""

    def test_every_setting_is_kept_except_recording(self):
        """Recording is transient; every other field is part of the profile.

        Args: None.
        Returns: None.
        """
        names = [field for field in asdict(Settings())]
        self.assertNotIn("record", profiles.PERSISTED_FIELDS)
        self.assertEqual(set(profiles.PERSISTED_FIELDS), set(names) - {"record"})
        for name in ("exposure_us", "gain_db", "fps", "mode", "auto_reference",
                     "single_field", "min_range", "flip_horizontal"):
            self.assertIn(name, profiles.PERSISTED_FIELDS)

    def test_every_number_has_a_range_and_every_text_its_choices(self):
        """Each number is range-checked; each text field is one of its choices.

        Args: None.
        Returns: None.
        """
        for name, kind in profiles.FIELD_TYPES.items():
            if kind in (int, float):
                self.assertIn(name, profiles.RANGES, name)
            elif kind is str:
                self.assertIn(name, profiles.CHOICES, name)
        defaults = Settings()
        for name in profiles.PERSISTED_FIELDS:  # the defaults themselves are valid
            self.assertEqual(profiles.clean_value(name, getattr(defaults, name)),
                             getattr(defaults, name))


class DefaultProfileTests(unittest.TestCase):
    """A camera with nothing saved starts exactly as before."""

    def test_demo_and_flir_cameras_keep_the_documented_defaults(self):
        """Exposure 8000 µs, gain 0 dB, 30 fps, Raw view, LabVIEW flow off.

        Args: None.
        Returns: None.
        """
        for info, demo in ((DEMO, True), (FLIR, False), (ANALOG, True)):
            profile = profiles.default_profile(info, demo)
            s = profile.settings
            self.assertEqual((s.exposure_us, s.gain_db, s.fps), (8000.0, 0.0, 30.0))
            self.assertEqual(s.mode, "Raw")
            self.assertFalse(s.record)
            self.assertFalse(profile.labview_flow)
            self.assertIsNone(profile.labview_saved)

    def test_an_analog_camera_starts_in_the_labview_flow(self):
        """The analog camera's own Fixed-gain settings are kept to return to.

        Args: None.
        Returns: None.
        """
        profile = profiles.default_profile(ANALOG)
        self.assertTrue(profile.labview_flow)
        self.assertEqual(profile.settings.mode, "Reference")
        self.assertEqual(set(profile.labview_saved), set(LABVIEW_FLOW))
        self.assertFalse(profile.labview_saved["scale_range"])
        self.assertEqual(profile.labview_saved["mode"], "Raw")

    def test_another_capture_device_keeps_its_own_settings(self):
        """An unrecognised video device gets the analog floor but no LabVIEW flow.

        Args: None.
        Returns: None.
        """
        profile = profiles.default_profile(WEBCAM)
        self.assertIsNone(profile.labview_saved)
        self.assertEqual(profile.settings.min_range, 30)


class SummaryTests(unittest.TestCase):
    """The one-line description used in log lines."""

    def test_summary_names_the_camera_values_view_and_flow(self):
        """Exposure, gain, rate and view; the LabVIEW flow when it is on.

        Args: None.
        Returns: None.
        """
        self.assertEqual(profiles.summary(profiles.default_profile(FLIR)),
                         "exposure 8000 µs, gain 0 dB, 30 fps, Raw view")
        self.assertEqual(profiles.summary(profiles.default_profile(ANALOG)),
                         "exposure 8000 µs, gain 0 dB, 30 fps, Reference view, LabVIEW flow")


class KeyTests(unittest.TestCase):
    """Stable store keys per camera, with demo cameras kept apart."""

    def test_flir_cameras_are_keyed_by_serial(self):
        """A FLIR camera's serial number names its profile.

        Args: None.
        Returns: None.
        """
        self.assertEqual(profiles.profile_key(FLIR), "profiles/flir/12345678")

    def test_demo_cameras_have_their_own_namespace(self):
        """A synthetic camera never shares a key with a real camera of that serial.

        Args: None.
        Returns: None.
        """
        self.assertEqual(profiles.profile_key(DEMO, demo=True), "profiles/demo/DEMO1")
        same_serial = CameraInfo("DEMO1", "Blackfly S")
        self.assertNotEqual(profiles.profile_key(same_serial),
                            profiles.profile_key(same_serial, demo=True))

    def test_capture_devices_are_keyed_by_usb_id_and_name_not_position(self):
        """The "video N" position changes between runs, so it is not in the key.

        Args: None.
        Returns: None.
        """
        moved = replace(ANALOG, serial="video 3", index=7, path="elsewhere")
        key = profiles.profile_key(ANALOG)
        self.assertEqual(key, "profiles/capture/2040-c200/Hauppauge%20USB-Live2")
        self.assertEqual(profiles.profile_key(moved), key)
        unknown = CameraInfo("video 1", "", kind="capture")
        self.assertEqual(profiles.profile_key(unknown),
                         "profiles/capture/usb-unknown/unnamed")

    def test_iphones_have_their_own_namespace(self):
        """An iPhone is keyed by its UDID-derived serial, apart from FLIR cameras.

        Args: None.
        Returns: None.
        """
        phone = CameraInfo("iphone-4d5e801e", "iPhone (USB)", kind="iphone",
                           path="00008110-001A4D5E801E")
        self.assertEqual(profiles.profile_key(phone), "profiles/iphone/iphone-4d5e801e")
        self.assertNotEqual(profiles.profile_key(phone),
                            profiles.profile_key(replace(phone, kind="flir")))

    def test_separators_in_names_are_encoded(self):
        """Slashes, backslashes and dots cannot split a key into groups.

        Args: None.
        Returns: None.
        """
        info = CameraInfo("a/b\\c.d", "")
        self.assertEqual(profiles.profile_key(info), "profiles/flir/a%2Fb%5Cc%2Ed")
        self.assertEqual(profiles.preset_key(" Fast/Slow é "),
                         "presets/fast%2Fslow%20%C3%A9")

    def test_preset_names_ignore_case_and_surrounding_spaces(self):
        """"Fast" and " fast " are the same preset on every platform.

        Args: None.
        Returns: None.
        """
        self.assertEqual(profiles.preset_key("Fast"), profiles.preset_key(" fast "))


class CleanValueTests(unittest.TestCase):
    """Type and range checks for one stored value."""

    def test_values_are_converted_to_the_field_type(self):
        """A whole float fits an int field, an int fits a float field.

        Args: None.
        Returns: None.
        """
        self.assertEqual(profiles.clean_value("average", 8.0), 8)
        self.assertIsInstance(profiles.clean_value("average", 8.0), int)
        self.assertIsInstance(profiles.clean_value("gamma", 2), float)
        self.assertIs(profiles.clean_value("invert", True), True)
        self.assertEqual(profiles.clean_value("mode", "Reference"), "Reference")

    def test_wrong_values_are_refused(self):
        """Wrong types, non-finite or fractional numbers, ranges and choices.

        Args: None.
        Returns: None.
        """
        for name, value in (("invert", 1), ("invert", "true"), ("mode", 3),
                            ("gamma", True), ("gamma", "1.0"), ("gamma", None),
                            ("gamma", float("nan")), ("fps", float("inf")),
                            ("average", 2.5), ("average", 0), ("exposure_us", 60000),
                            ("gain_db", -1), ("min_range", 256),
                            ("mode", "Fringes"), ("smoothing_method", "Box")):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    profiles.clean_value(name, value)


class ParseProfileTests(unittest.TestCase):
    """Tolerant reading: bad parts fall back to the defaults, never raise."""

    def setUp(self):
        """Prepare a log and the demo camera's defaults.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.defaults = profiles.default_profile(DEMO, demo=True)

    def parse(self, data, defaults=None):
        """Parse a profile written as Python data.

        Args: data is the object to store as JSON; defaults overrides the
            demo camera's defaults.
        Returns: The parsed Profile or None.
        """
        return profiles.parse_profile(json.dumps(data), defaults or self.defaults,
                                      self.log, "Saved settings demo")

    def test_a_profile_survives_a_round_trip(self):
        """Every kept value, and the LabVIEW state, reads back unchanged.

        Args: None.
        Returns: None.
        """
        profile = profiles.Profile(
            replace(Settings(), exposure_us=12000.0, gain_db=3.5, fps=20.0,
                    mode="Previous", average=8, smoothing_method="Median",
                    invert=True, min_range=200, record=True),
            {key: getattr(Settings(), key) for key in LABVIEW_FLOW})
        read = profiles.parse_profile(profile.dumps(), self.defaults, self.log)
        self.assertEqual(read.settings, replace(profile.settings, record=False))
        self.assertEqual(read.labview_saved, profile.labview_saved)
        self.assertNotIn("record", stored({"x": profile.dumps()}, "x")["settings"])
        self.assertEqual(self.log.lines, [])

    def test_unreadable_entries_give_none(self):
        """Not JSON, JSON that is not an object, or not text at all.

        Args: None.
        Returns: None.
        """
        for text in ("{not json", "[1, 2]", b"bytes", 42):
            self.assertIsNone(profiles.parse_profile(text, self.defaults, self.log, "P"))
        self.assertEqual(len(self.log.lines), 4)
        self.assertIn("P: the saved entry cannot be read", self.log.lines[0])

    def test_unknown_keys_are_ignored(self):
        """Older builds' palette and clahe, and future fields, change nothing.

        Args: None.
        Returns: None.
        """
        profile = self.parse({"format": 99, "future": 1, "settings": {
            "palette": "Jet", "clahe": True, "exposure_us": 9000, "new_field": 3}})
        self.assertEqual(profile.settings,
                         replace(self.defaults.settings, exposure_us=9000.0))
        self.assertEqual(self.log.lines, [])

    def test_invalid_values_fall_back_to_the_defaults_and_are_logged(self):
        """Each bad value is replaced by the camera's default and named in the log.

        Args: None.
        Returns: None.
        """
        profile = self.parse({"settings": {
            "gain_db": "loud", "average": 2.5, "mode": "Fringes", "invert": 1,
            "gamma": 9.0, "fps": 25, "record": True}})
        s = profile.settings
        self.assertEqual((s.gain_db, s.average, s.mode, s.invert, s.gamma),
                         (0.0, 1, "Raw", False, 1.0))
        self.assertEqual(s.fps, 25.0)
        self.assertFalse(s.record)  # never restored, even when a store holds it
        self.assertEqual(len(self.log.lines), 5)
        self.assertIn("gain_db = 'loud' is not a number; using the default 0.0",
                      "\n".join(self.log.lines))

    def test_missing_settings_use_the_defaults(self):
        """An entry without a settings object starts from the defaults.

        Args: None.
        Returns: None.
        """
        profile = self.parse({"labview_flow": False, "settings": [1]})
        self.assertEqual(profile.settings, self.defaults.settings)
        self.assertIn("no saved settings", self.log.lines[0])

    def test_labview_state_is_restored(self):
        """A profile saved in the LabVIEW flow comes back in it, values kept.

        Args: None.
        Returns: None.
        """
        saved = {key: getattr(Settings(), key) for key in LABVIEW_FLOW}
        saved["gamma"] = 1.5
        profile = self.parse({"settings": {"mode": "Reference"}, "labview_flow": True,
                              "labview_saved": saved})
        self.assertTrue(profile.labview_flow)
        self.assertEqual(profile.labview_saved, saved)

    def test_bad_labview_state_falls_back(self):
        """A wrong flag uses the default; missing kept values use the settings.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.parse({"settings": {}, "labview_flow": "yes"}).labview_flow)
        analog = profiles.default_profile(ANALOG)
        self.assertEqual(self.parse({"settings": {}}, analog).labview_saved,
                         {**analog.labview_saved})  # nothing said: the default (on)
        profile = self.parse({"settings": {"gamma": 2.0}, "labview_flow": True,
                              "labview_saved": "lost"})
        self.assertEqual(profile.labview_saved,
                         {key: getattr(profile.settings, key) if key != "gamma" else 1.0
                          for key in LABVIEW_FLOW})
        profile = self.parse({"settings": {}, "labview_flow": True,
                              "labview_saved": {"gamma": "x", "invert": True}}, analog)
        self.assertEqual(profile.labview_saved,
                         {**analog.labview_saved, "invert": True})
        lines = "\n".join(self.log.lines)
        self.assertIn("labview_flow = 'yes' is not true or false", lines)
        self.assertIn("the settings kept for the LabVIEW flow are missing", lines)
        self.assertIn("gamma = 'x' is not a number", lines)


class ClampTests(unittest.TestCase):
    """Camera values fitted into one camera's limits."""

    def test_values_inside_the_limits_are_unchanged(self):
        """No change, no note.

        Args: None.
        Returns: None.
        """
        settings = Settings()
        self.assertEqual(profiles.clamp_to_limits(settings, {"ExposureTime": (10, 30000)}),
                         (settings, []))

    def test_values_outside_are_clamped_and_described(self):
        """Each clamped value is named with the range and the value used.

        Args: None.
        Returns: None.
        """
        settings = replace(Settings(), exposure_us=40000.0, gain_db=0.0, fps=60.0)
        clamped, notes = profiles.clamp_to_limits(
            settings, {"ExposureTime": (10, 30000), "Gain": (1, 24),
                       "AcquisitionFrameRate": (1, 59.5)})
        self.assertEqual((clamped.exposure_us, clamped.gain_db, clamped.fps),
                         (30000, 1, 59.5))
        self.assertEqual(notes, [
            "Exposure 40000 µs is outside this camera's range (10–30000 µs); "
            "using 30000 µs.",
            "Gain 0 dB is outside this camera's range (1–24 dB); using 1 dB.",
            "Frame rate 60 fps is outside this camera's range (1–59.5 fps); "
            "using 59.5 fps."])

    def test_missing_limits_leave_values_alone(self):
        """A camera that reports no limits (a video device) changes nothing.

        Args: None.
        Returns: None.
        """
        settings = replace(Settings(), exposure_us=40000.0)
        self.assertEqual(profiles.clamp_to_limits(settings, {}), (settings, []))


class ProfileStoreTests(unittest.TestCase):
    """Saving and restoring camera profiles."""

    def setUp(self):
        """Create an empty store.

        Args: None.
        Returns: None.
        """
        self.store = DictStore()
        self.log = ListLog()
        self.profiles = profiles.ProfileStore(self.store, self.log)
        self.defaults = profiles.default_profile(FLIR)
        self.key = profiles.profile_key(FLIR)

    def changed(self, **values):
        """Build the default profile with some settings changed.

        Args: values are Settings fields.
        Returns: A Profile.
        """
        return replace(self.defaults, settings=replace(self.defaults.settings, **values))

    def test_nothing_saved_gives_none(self):
        """A camera seen for the first time has no profile.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.profiles.load_profile(self.key, self.defaults))

    def test_save_then_load_in_a_new_session(self):
        """What one run saves, the next run reads back.

        Args: None.
        Returns: None.
        """
        profile = self.changed(exposure_us=9000.0, gain_db=6.0)
        self.assertTrue(self.profiles.save_profile(self.key, profile))
        self.assertFalse(self.profiles.save_profile(self.key, profile))  # unchanged
        again = profiles.ProfileStore(self.store, self.log)
        self.assertEqual(again.load_profile(self.key, self.defaults), profile)

    def test_a_tracked_profile_is_written_only_after_a_change(self):
        """The defaults in force are not written; a changed value is.

        Args: None.
        Returns: None.
        """
        self.profiles.track(self.key, self.defaults)
        self.assertFalse(self.profiles.save_profile(self.key, self.defaults))
        self.assertEqual(self.store, {})
        self.assertTrue(self.profiles.save_profile(self.key, self.changed(gain_db=1.0)))

    def test_settled_saves_wait_until_the_value_stops_changing(self):
        """A drag (a new value every tick) writes once, after it ends.

        Args: None.
        Returns: None.
        """
        self.profiles.track(self.key, self.defaults)
        writes = [self.profiles.save_when_settled(self.key, self.changed(gain_db=gain))
                  for gain in (1.0, 2.0, 3.0, 3.0, 3.0)]
        self.assertEqual(writes, [False, False, False, True, False])
        self.assertEqual(stored(self.store, self.key)["settings"]["gain_db"], 3.0)

    def test_a_cleaned_entry_is_rewritten(self):
        """Values replaced on loading are written back clean once settled.

        Args: None.
        Returns: None.
        """
        self.store[self.key] = json.dumps({"settings": {"gain_db": "loud", "fps": 20}})
        profile = self.profiles.load_profile(self.key, self.defaults)
        self.profiles.track(self.key, profile)
        self.assertFalse(self.profiles.save_when_settled(self.key, profile))
        self.assertTrue(self.profiles.save_when_settled(self.key, profile))
        self.assertEqual(stored(self.store, self.key)["settings"]["gain_db"], 0.0)

    def test_forget_removes_the_entry(self):
        """Reset to defaults leaves nothing saved and writes nothing more.

        Args: None.
        Returns: None.
        """
        self.profiles.save_profile(self.key, self.changed(gain_db=2.0))
        self.profiles.forget_profile(self.key, self.defaults)
        self.assertNotIn(self.key, self.store)
        self.assertFalse(self.profiles.save_when_settled(self.key, self.defaults))
        self.assertFalse(self.profiles.save_profile(self.key, self.defaults))
        self.assertNotIn(self.key, self.store)

    def test_cameras_do_not_share_entries(self):
        """Two cameras' profiles live under two keys.

        Args: None.
        Returns: None.
        """
        other = profiles.profile_key(CameraInfo("87654321", "Blackfly S"))
        self.profiles.save_profile(self.key, self.changed(gain_db=2.0))
        self.profiles.save_profile(other, self.changed(gain_db=5.0))
        self.assertEqual(self.profiles.load_profile(self.key, self.defaults).settings.gain_db,
                         2.0)
        self.assertEqual(self.profiles.load_profile(other, self.defaults).settings.gain_db,
                         5.0)

    def test_sync_writes_through_when_the_store_can(self):
        """QSettings' sync is called; a store without one is left alone.

        Args: None.
        Returns: None.
        """
        self.profiles.sync()  # DictStore has no sync()
        syncing = SyncingStore()
        profiles.ProfileStore(syncing).sync()
        self.assertEqual(syncing.syncs, 1)


class PresetTests(unittest.TestCase):
    """Named presets: save, list, find, load and delete."""

    def setUp(self):
        """Create an empty store.

        Args: None.
        Returns: None.
        """
        self.store = DictStore()
        self.log = ListLog()
        self.profiles = profiles.ProfileStore(self.store, self.log)
        self.defaults = profiles.default_profile(DEMO, demo=True)

    def preset(self, **values):
        """Build a profile with some settings changed.

        Args: values are Settings fields.
        Returns: A Profile.
        """
        return replace(self.defaults, settings=replace(self.defaults.settings, **values))

    def test_saved_presets_are_listed_by_name(self):
        """The list is sorted ignoring case and names the source model.

        Args: None.
        Returns: None.
        """
        self.profiles.save_preset("beta", self.preset(), "Blackfly S")
        self.profiles.save_preset(" Alpha ", self.preset(), "Synthetic camera")
        self.store["profiles/demo/DEMO1"] = "{}"  # not a preset
        self.assertEqual(self.profiles.presets(), [
            profiles.PresetEntry("Alpha", "Synthetic camera"),
            profiles.PresetEntry("beta", "Blackfly S")])
        data = stored(self.store, profiles.preset_key("alpha"))
        self.assertEqual((data["name"], data["source_model"], data["format"]),
                         ("Alpha", "Synthetic camera", profiles.FORMAT))

    def test_find_ignores_case_and_load_reads_the_settings(self):
        """"ALPHA" finds "Alpha"; loading gives its saved values.

        Args: None.
        Returns: None.
        """
        self.profiles.save_preset("Alpha", self.preset(gain_db=4.0, mode="Previous"))
        self.assertEqual(self.profiles.find_preset("ALPHA").name, "Alpha")
        self.assertIsNone(self.profiles.find_preset("Gamma"))
        loaded = self.profiles.load_preset("alpha", self.defaults)
        self.assertEqual((loaded.settings.gain_db, loaded.settings.mode), (4.0, "Previous"))
        self.assertIsNone(self.profiles.load_preset("Gamma", self.defaults))

    def test_saving_a_name_again_replaces_the_preset(self):
        """One entry per name; the newer settings and spelling win.

        Args: None.
        Returns: None.
        """
        self.profiles.save_preset("Alpha", self.preset(gain_db=1.0))
        self.profiles.save_preset("ALPHA", self.preset(gain_db=2.0))
        self.assertEqual(self.profiles.presets(), [profiles.PresetEntry("ALPHA", "")])
        self.assertEqual(self.profiles.load_preset("alpha", self.defaults).settings.gain_db,
                         2.0)

    def test_delete_removes_the_preset(self):
        """A deleted preset is no longer listed or loadable.

        Args: None.
        Returns: None.
        """
        self.profiles.save_preset("Alpha", self.preset())
        self.profiles.delete_preset("alpha")
        self.assertEqual(self.profiles.presets(), [])
        self.assertEqual(self.store, {})

    def test_broken_entries_are_skipped_and_odd_ones_repaired(self):
        """Unreadable entries are logged and left out; a lost name comes from the key.

        Args: None.
        Returns: None.
        """
        self.store["presets/broken"] = "{oops"
        self.store["presets/fast%20one"] = json.dumps({"settings": {}, "name": "",
                                                       "source_model": 7})
        self.assertEqual(self.profiles.presets(), [profiles.PresetEntry("fast one", "")])
        self.assertIn("Preset presets/broken: the saved entry cannot be read",
                      self.log.lines[0])

    def test_a_preset_from_another_model_loads_with_the_targets_defaults(self):
        """Missing or invalid values come from the camera it is applied to.

        Args: None.
        Returns: None.
        """
        self.store[profiles.preset_key("Old")] = json.dumps(
            {"name": "Old", "settings": {"exposure_us": 20000, "palette": "Jet",
                                         "gamma": "bright"}})
        analog = profiles.default_profile(ANALOG)
        loaded = self.profiles.load_preset("Old", analog)
        self.assertEqual(loaded.settings.exposure_us, 20000.0)
        self.assertEqual(loaded.settings.gamma, analog.settings.gamma)
        self.assertEqual(loaded.settings.mode, "Reference")  # the analog default
        self.assertTrue(loaded.labview_flow)


if __name__ == "__main__":
    unittest.main()
