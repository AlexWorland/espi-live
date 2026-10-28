import os
import sys
import time

import numpy as np
import PySpin
import cv2

continue_recording = True


def configure_camera(nodemap, exposure_us=20000.0, gain_db=5.0, fps=30.0):
    """
    Put camera into manual mode and set initial exposure, gain, and frame rate.

    Parameters
    ----------
    nodemap : PySpin.INodeMap
        Camera nodemap.
    exposure_us : float
        Desired exposure time in microseconds.
    gain_db : float
        Desired gain in dB.
    fps : float
        Desired frame rate in frames per second.
    """

    # -------- Exposure (manual) --------
    try:
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsReadable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
            entry_off = PySpin.CEnumEntryPtr(node_exposure_auto.GetEntryByName("Off"))
            node_exposure_auto.SetIntValue(entry_off.GetValue())

        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsReadable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
            exp_min = node_exposure_time.GetMin()
            exp_max = node_exposure_time.GetMax()
            exposure_us = max(exp_min, min(exp_max, exposure_us))
            node_exposure_time.SetValue(exposure_us)
            print(f"Initial exposure set to {node_exposure_time.GetValue():.1f} us "
                  f"(range {exp_min:.1f}–{exp_max:.1f})")
    except PySpin.SpinnakerException as ex:
        print("Exposure configuration error:", ex)

    # -------- Gain (manual) --------
    try:
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsReadable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
            entry_off = PySpin.CEnumEntryPtr(node_gain_auto.GetEntryByName("Off"))
            node_gain_auto.SetIntValue(entry_off.GetValue())

        node_gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsReadable(node_gain) and PySpin.IsWritable(node_gain):
            g_min = node_gain.GetMin()
            g_max = node_gain.GetMax()
            gain_db = max(g_min, min(g_max, gain_db))
            node_gain.SetValue(gain_db)
            print(f"Initial gain set to {node_gain.GetValue():.2f} dB "
                  f"(range {g_min:.2f}–{g_max:.2f})")
    except PySpin.SpinnakerException as ex:
        print("Gain configuration error:", ex)

    # -------- Frame rate (manual) --------
    try:
        # Some models need this boolean to enable manual frame rate
        node_fps_enable = PySpin.CBooleanPtr(
            nodemap.GetNode("AcquisitionFrameRateEnabled")
        )
        if PySpin.IsReadable(node_fps_enable) and PySpin.IsWritable(node_fps_enable):
            node_fps_enable.SetValue(True)

        # Turn off frame-rate auto, if present
        node_fps_auto = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionFrameRateAuto")
        )
        if node_fps_auto is not None and PySpin.IsReadable(node_fps_auto) and PySpin.IsWritable(node_fps_auto):
            entr
