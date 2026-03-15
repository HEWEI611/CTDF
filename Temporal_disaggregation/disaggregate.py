import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def temporal_disaggregation(daily_01deg_file, daily_01deg_coarse_file,
                             hourly_coarse_file, output_file):
    """
    Disaggregate spatially downscaled daily precipitation (0.01°) to hourly
    by interpolating temporal ratios from coarse-resolution (0.1°) hourly data.
    """
    ds_daily_fine   = xr.open_dataset(daily_01deg_file)
    ds_daily_coarse = xr.open_dataset(daily_01deg_coarse_file)
    ds_hourly_coarse = xr.open_dataset(hourly_coarse_file)

    ref_dims = ds_daily_fine.precipitation.dims  # reference dimension order

    lon_fine   = ds_daily_fine.lon.values
    lat_fine   = ds_daily_fine.lat.values
    lon_coarse = ds_daily_coarse.lon.values
    lat_coarse = ds_daily_coarse.lat.values

    daily_times  = pd.to_datetime(ds_daily_fine.time.values)
    hourly_times = pd.to_datetime(ds_hourly_coarse.time.values)

    common_dates = sorted(np.intersect1d(daily_times.date, hourly_times.date))
    print(f"Processing {len(common_dates)} days: {common_dates[0]} – {common_dates[-1]}")

    all_data, all_times = [], []

    for date in common_dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        daily_fine   = ds_daily_fine.sel(time=date_str)["precipitation"].values
        daily_coarse = ds_daily_coarse.sel(time=date_str)["precipitation"].values

        hourly_mask  = hourly_times.date == date
        day_hours    = hourly_times[hourly_mask]
        day_hourly   = ds_hourly_coarse["precipitation"].values[hourly_mask]

        for h, t in enumerate(day_hours):
            hour_coarse = day_hourly[h]

            # Temporal ratio at coarse resolution
            ratio_coarse = np.where(daily_coarse > 1e-6,
                                    hour_coarse / daily_coarse, 0.0)

            # Interpolate ratio to fine resolution
            if ref_dims == ("time", "lon", "lat"):
                interp = RegularGridInterpolator(
                    (lon_coarse, lat_coarse), ratio_coarse,
                    method="linear", bounds_error=False, fill_value=0)
                lon_g, lat_g = np.meshgrid(lon_fine, lat_fine, indexing="ij")
            else:  # (time, lat, lon)
                interp = RegularGridInterpolator(
                    (lat_coarse, lon_coarse), ratio_coarse,
                    method="linear", bounds_error=False, fill_value=0)
                lat_g, lon_g = np.meshgrid(lat_fine, lon_fine, indexing="ij")

            points = np.column_stack([lon_g.ravel(), lat_g.ravel()]) \
                if ref_dims == ("time", "lon", "lat") \
                else np.column_stack([lat_g.ravel(), lon_g.ravel()])

            ratio_fine = interp(points).reshape(lon_g.shape)
            ratio_fine = np.clip(ratio_fine, 0, 2)

            hourly_fine = np.where(np.isfinite(daily_fine * ratio_fine),
                                   np.maximum(daily_fine * ratio_fine, 0), 0)

            all_data.append(hourly_fine)
            all_times.append(t)

    hourly_array = np.stack(all_data, axis=0)

    # Build output dataset preserving original dimension order
    if ref_dims == ("time", "lon", "lat"):
        coords = {"time": pd.to_datetime(all_times), "lon": lon_fine, "lat": lat_fine}
        dims   = ["time", "lon", "lat"]
    else:
        coords = {"time": pd.to_datetime(all_times), "lat": lat_fine, "lon": lon_fine}
        dims   = ["time", "lat", "lon"]

    ds_out = xr.Dataset({"precipitation": (dims, hourly_array)}, coords=coords)

    ds_out.attrs = {
        "title": "Hourly downscaled GPM precipitation at 0.01°",
        "method": "Temporal disaggregation with bilinearly interpolated ratios",
        "spatial_resolution": "0.01 degree",
        "temporal_resolution": "hourly",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ds_out["precipitation"].attrs = {"units": "mm/hour", "valid_range": [0.0, 200.0]}

    ds_out.to_netcdf(output_file, encoding={
        "precipitation": {"zlib": True, "complevel": 6,
                          "dtype": "float32", "_FillValue": np.nan},
        "time": {"units": "hours since 1900-01-01"},
    })
    print(f"Saved to: {output_file}")

    ds_daily_fine.close()
    ds_daily_coarse.close()
    ds_hourly_coarse.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTDF: Temporal disaggregation of downscaled daily GPM to hourly")
    parser.add_argument("--daily-fine",   required=True,
                        help="Downscaled daily precipitation at 0.01° (NetCDF)")
    parser.add_argument("--daily-coarse", required=True,
                        help="Original daily GPM at 0.1° (NetCDF)")
    parser.add_argument("--hourly-coarse", required=True,
                        help="Original hourly GPM at 0.1° (NetCDF)")
    parser.add_argument("--output",       required=True,
                        help="Output hourly 0.01° NetCDF file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    temporal_disaggregation(
        daily_01deg_file=args.daily_fine,
        daily_01deg_coarse_file=args.daily_coarse,
        hourly_coarse_file=args.hourly_coarse,
        output_file=args.output,
    )
```
