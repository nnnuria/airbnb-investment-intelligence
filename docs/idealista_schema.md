# Idealista scraper — output schema

This document describes what each scrape job produces. It is a companion to
[`docs/schema.md`](schema.md), which is the ABT contract; this file is the
*upstream raw* contract for the Idealista source.

## Files

For every (city, operation) pair, one JSON Lines file is written to:

```
data/raw/idealista/<city>/<operation>_<YYYY-MM-DD>.jsonl
```

- `city` ∈ `{madrid, barcelona, malaga}`
- `operation` ∈ `{sale, rent}` (Idealista calls these *venta* / *alquiler*)
- One JSON object per line, UTF-8 encoded
- Files are append-once per snapshot — re-run to refresh

> The raw directory is git-ignored. Commit only aggregated / processed
> outputs.

## Raw fields (from the actor)

The fields below are produced by
[`igolaizola/idealista-scraper`](https://apify.com/igolaizola/idealista-scraper).
Names and presence depend on the actor version and on whether
`fetchDetails` / `fetchStats` were enabled. Treat the list as best-effort
documentation — verify against the first real run.

| Category    | Likely keys (raw)                                                                  |
| ----------- | ---------------------------------------------------------------------------------- |
| identifiers | `propertyCode` · `url`                                                             |
| pricing     | `price` (EUR) · `pricePerM2`                                                       |
| size/rooms  | `size` (m²) · `rooms` · `bathrooms`                                                |
| type/state  | `propertyType` · `homeType` · `status` · `floor`                                   |
| location    | `address` · `district` · `neighborhood` · `municipality` · `province` · `country`  |
| geo         | `latitude` · `longitude`                                                           |
| narrative   | `title` · `description`                                                            |
| media       | `images` (list of URLs) · `virtualTour`                                            |
| features    | `features` / `characteristics` (list of amenity tags)                              |
| advertiser  | `agency` · `phone`                                                                 |
| timestamps  | `publicationDate`                                                                  |
| optional    | `_details` (when `fetchDetails=true`) · `_stats` (when `fetchStats=true`)          |

The scraper additionally tags every record with three provenance fields it
adds itself:

| Field         | Type   | Meaning                                            |
| ------------- | ------ | -------------------------------------------------- |
| `_city`       | string | One of `madrid`, `barcelona`, `malaga`             |
| `_operation`  | string | `sale` or `rent`                                   |
| `_scraped_at` | string | ISO date (YYYY-MM-DD) the scrape ran               |

## Canonical (normalised) schema

`airbnb_iip.data.scrapers.idealista.normalize_listing` projects a raw record
onto the table below. Use this when joining with the Inside Airbnb ABT or
the external €/m² and rent-index tables.

| Column              | Type        | Notes                                            |
| ------------------- | ----------- | ------------------------------------------------ |
| `listing_id`        | string      | Idealista property code; primary key            |
| `url`               | string      | Full listing URL                                |
| `title`             | string      |                                                 |
| `price_eur`         | float       | EUR. Sale = asking price, rent = €/month         |
| `price_per_m2_eur`  | float       | EUR / m²                                         |
| `size_m2`           | float       | Useful area, m²                                 |
| `rooms`             | int         | Bedrooms                                        |
| `bathrooms`         | int         |                                                 |
| `property_type`     | string      | `homes` / `newDevelopments` / …                 |
| `home_type`         | string      | `flat` / `penthouse` / `duplex` / …             |
| `floor`             | string      | e.g. `"3"` / `"bajo"` / `"ático"`               |
| `address`           | string      | Street-level if Idealista exposes it            |
| `district`          | string      | Administrative district                         |
| `neighborhood`      | string      |                                                 |
| `municipality`      | string      |                                                 |
| `province`          | string      |                                                 |
| `latitude`          | float       |                                                 |
| `longitude`         | float       |                                                 |
| `description`       | string      |                                                 |
| `features`          | list[str]   | Amenity tags                                    |
| `images`            | list[str]   | Image URLs                                      |
| `n_images`          | int         | Derived: `len(images)`                          |
| `virtual_tour`      | string/null | URL of 3D tour if present                       |
| `has_virtual_tour`  | bool        | Derived: `virtual_tour is not None`             |
| `agency`            | string      | Advertiser name                                 |
| `phone`             | string      | Contact phone (when shown)                      |
| `publication_date`  | string      | ISO date of original listing publication        |
| `status`            | string      | `bareOwnership` / `tenanted` / `free` / …       |
| `city`              | string      | From `_city` provenance tag                     |
| `operation`         | string      | From `_operation` provenance tag                |
| `scraped_at`        | string      | From `_scraped_at` provenance tag (ISO date)    |

## Sample

See [`Data/sample/idealista_sample.jsonl`](../Data/sample/idealista_sample.jsonl)
for three illustrative records (one Madrid sale, one Barcelona rent, one
Málaga sale). The sample is synthetic and committed for tests / offline
development; field names are best-effort and should be reconciled against
the first real run.
