SELECT
    r.park_code AS park_code,
    p.park_name,
    p.state,
    b.scientific_name,
    b.common_names,
    b.raptor_group,
    r.occurrence,
    r.nativeness,
    r.abundance,
    r.seasonality,
    r.conservation_status,
    ST_AsGeoJSON(ps.geometry) AS geometry_geojson
FROM records AS r
LEFT JOIN birds AS b ON r.species_code = b.species_code
LEFT JOIN parks AS p ON r.park_code = p.park_code
LEFT JOIN park_shapes AS ps ON r.park_code = ps.park_code
WHERE b.common_names = 'California Condor'
ORDER BY r.occurrence DESC, r.abundance DESC;


--- Count of birds of prey by park
SELECT
    p.park_name,
    p.state,
    COUNT(*) AS raptor_count
FROM records AS r
LEFT JOIN birds AS b ON r.species_code = b.species_code
LEFT JOIN parks AS p ON r.park_code = p.park_code
WHERE b.is_raptor = True
GROUP BY p.park_name, p.state
ORDER BY raptor_count DESC;


-- Count of species by park
SELECT
    p.park_name,
    p.state,
    COUNT(*) AS total_species
FROM records AS r
LEFT JOIN birds AS b ON r.species_code = b.species_code
LEFT JOIN parks AS p ON r.park_code = p.park_code
GROUP BY p.park_name, p.state
ORDER BY total_species DESC;