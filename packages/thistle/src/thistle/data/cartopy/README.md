# Vendored cartopy data

Natural Earth 110m shapefiles (public domain, naturalearthdata.com) for the
two features `thistle groundtrack` renders by default — coastlines and
country borders — so the map works offline on a fresh machine (~0.65 MB).

The directory layout mirrors cartopy's cache
(`shapefiles/natural_earth/<category>/<name>.*`) because `_map.py` points
`cartopy.config["pre_existing_data_dir"]` here: cartopy checks this
read-only directory before its download cache, so these exact
(feature, scale) pairs never trigger a download, while any other feature
falls through to cartopy's normal download-and-cache path (and degrades
gracefully offline — see `_map._add_base_features`).

These files are a pinned snapshot; coastlines and land borders at 110m
change rarely enough that staleness is not a practical concern. If
`make_map` ever adds a feature or changes resolution, vendor the matching
files here or it will download at first use.
