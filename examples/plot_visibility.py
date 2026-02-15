"""Plot visibility rings for a ground site at three elevation angles."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from thistle.ground_sites import visibility_circle

LAT, LON = 40.0, -105.0
SAT_ALT = 500_000

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

ax.set_global()
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

for el, color in [(0, "C0"), (5, "C1"), (10, "C2")]:
    lats, lons = visibility_circle(LAT, LON, sat_alt=SAT_ALT, min_el=el)
    ax.plot([*lons, lons[0]], [*lats, lats[0]], color=color, label=f"{el}°",
            transform=ccrs.Geodetic())

ax.plot(LON, LAT, "k+", markersize=10, transform=ccrs.PlateCarree())
ax.legend(title="Min elevation", loc="lower left")
ax.set_title(f"Visibility circles \u2013 {SAT_ALT/1000:.0f} km altitude")
plt.tight_layout()
plt.show()
