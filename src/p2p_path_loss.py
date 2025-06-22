from astropy import units as u
from astropy import constants as const
from geopy.distance import geodesic
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
from pycraf import pathprof, conversions as cnv
import math

import warnings

warnings.filterwarnings("ignore")

pathprof.SrtmConf.set(download='missing', server='viewpano', srtm_dir='./srtmdir')


def calculate_fspl(distance, frequency):
    """
    Calculate Free Space Path Loss (FSPL) in dB

    Parameters:
    distance: Distance with units (e.g., 1 * u.km)
    frequency: Frequency with units (e.g., 5 * u.GHz)

    Returns:
    float: Path loss in dB
    """
    # Convert inputs to standard units for the formula
    distance_km = distance.to(u.km).value
    frequency_mhz = frequency.to(u.MHz).value

    # Calculate path loss using the formula
    path_loss = 20 * math.log10(distance_km) + 20 * math.log10(frequency_mhz) + 32.44
    return path_loss


def tx_rx_path_loss(lon_tx, lat_tx,
                    lon_rx, lat_rx,
                    h_tg, h_rg,
                    time_percent=1 * u.percent,
                    hprof_step=90,
                    frequency=5 * u.GHz,
                    temperature=290. * u.K,
                    pressure=1013. * u.hPa,
                    zone_r=pathprof.CLUTTER.SPARSE,
                    zone_t=pathprof.CLUTTER.SPARSE,
                    G_t=0 * cnv.dBi,
                    G_r=10 * cnv.dBi):
    hprof_step_default = hprof_step
    distance_km = geodesic((lon_tx, lat_tx),
                           (lon_rx, lat_rx)).kilometers

    if distance_km * 1000 / hprof_step_default >= 5:
        hprof_step_this = hprof_step_default * u.m
    else:
        min_steps = 5
        hprof_step_this = max(1, int(distance_km * 1000 / (min_steps + 0.2))) * u.m
        # print(f"distance: {distance_km}; Step Size: {hprof_step_this}")
        # failed_points.append((geometry.x, geometry.y, distance_km))

    if distance_km > 0.1:
        results = pathprof.losses_complete(
            frequency,
            temperature,
            pressure,
            lon_tx * u.deg, lat_tx * u.deg,
            lon_rx * u.deg, lat_rx * u.deg,
            h_tg, h_rg,
            hprof_step_this,
            time_percent * u.percent,
            zone_t=zone_t, zone_r=zone_r,
            G_t=G_t, G_r=G_r,
        )
        loss_db = results['L_b_corr'].value[0]
    else:
        height_diff = abs(h_tg - h_rg)
        loss_db = calculate_fspl(distance=height_diff, frequency=frequency)

    return loss_db, distance_km
