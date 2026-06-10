# Short-Term Rental Regulatory Sources

**UC2 Regulatory Agent — corpus index**
Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026

---

**Snapshot date:** 10 June 2026
**Maintainer:** _(add your name)_
**Status:** living document — Spanish STR rules changed repeatedly through 2025; **re-verify every source against the official site before the demo.**

> This file is the index for the regulatory RAG corpus. The agent's answers are grounded **only** in the official documents listed here, must **cite source + snapshot date**, and must carry the disclaimer: _"Indicative only — not legal advice. Verify against current official municipal/regional sources."_

---

## How to use this file

1. Download each **official primary document** below into `data/regulatory/<city>/`.
2. Record the **source URL + download date** alongside each file (that metadata is the citation layer and the governance story).
3. Prefer official sources (city council, regional government, BOE/BOCM/BOJA) over third-party summaries.
4. The agent returns a **regulatory-risk flag** + the cited rule, which feeds the UC2 recommendation.

---

## 0. National layer (Spain) — applies to all three cities

A national registry now sits **on top of** the regional tourist licence.

- **Rule:** every short-term rental needs a national registration number (NRA), under **Royal Decree 1312/2024** (23 Dec 2024) and **EU Regulation 2024/1028**.
- **Timeline:** in force 2 Jan 2025, with a grace period to **1 July 2025**; after that, booking platforms (Airbnb, Booking, Vrbo) must remove listings without an NRA.
- **Key distinction:** the NRA is separate from, and additional to, the regional licence (VUT / HUT / VFT below).
- **Note:** classified-ad sites that don't process bookings (e.g. Idealista) are currently **exempt** from this registry requirement.

**Official sources**
- Ministerio de Vivienda — Ventanilla Única Digital portal: https://www.mivau.gob.es/vivienda/ventanilla-unica/alojamiento-de-uso-turistico
- Royal Decree 1312/2024 — full text on the BOE (boe.es)
- EU Regulation 2024/1028 — EUR-Lex

---

## 1. Madrid — licence: VUT (Vivienda de Uso Turístico)

- **Current rule:** the **Plan RESIDE** replaced the 2019 *Plan Especial de Hospedaje* (the old "three rings"). Approved definitively **27 Aug 2025**, published in the **BOCM on 22 Sept 2025**.
- **Zoning:** two areas — the **historic centre** (area APE 00.01, roughly inside the M-30) and the rest of the city, with much stricter rules in the centre.
- **Centre restriction:** no new licences for "dispersed" tourist flats inside residential buildings — **not even on the ground floor**; tourist use is only allowed in whole, exclusively-tourist buildings.
- **Also required:** CIVUT (Certificado de Idoneidad) technical certificate + express approval from the community of owners.

**Official sources**
- Ayuntamiento de Madrid: https://www.madrid.es _(search "Plan RESIDE")_
- Plan RESIDE — BOCM publication, 22 Sept 2025

---

## 2. Barcelona — licence: HUT (Habitatge d'Ús Turístic) · registry: NIRTC (Catalonia)

- **Current rule:** **no new HUT licences** have been issued since the 2014 moratorium and the **2017 PEUAT** (Pla Especial Urbanístic d'Allotjaments Turístics), which divides the city into four zones.
- **2028 phase-out:** all ~10,101 existing HUT licences are set to **expire by Oct/Nov 2028 with no renewal**; Spain's Constitutional Court upheld this in **March 2025**.
- **Governing law:** Catalan **Decree Law 3/2023**.
- **Practical effect for UC2:** for almost any Barcelona flat, a *new* legal STR is not possible.

**Official sources**
- Ajuntament de Barcelona — Decree Law 3/2023 / HUTs: https://ajuntament.barcelona.cat/turisme/en/application-decree-law-32023-tourist-use-flats-huts
- PEUAT — Ajuntament de Barcelona urban planning pages
- Registry of Tourism of Catalonia (NIRTC) — Generalitat de Catalunya

---

## 3. Málaga — licence: VFT (Vivienda con Fines Turísticos) · registry: RTA (Andalusia)

- **Governing law:** Andalusian **Decree 28/2016** establishes the VFT regime.
- **Staged restriction:** licences frozen in **43 saturated neighbourhoods** (tourist flats > 8% of housing) from **Jan 2025**, followed by a **full city-wide moratorium** on new tourist dwellings from **late Aug 2025**, for up to three years (or until the revised PGOU), under Andalusian **Decree-Law 1/2025**.
- **Existing licences:** continue to operate; no new ones granted anywhere in the city while the moratorium is in force.

**Official sources**
- Junta de Andalucía — Registro de Turismo de Andalucía (RTA): https://www.juntadeandalucia.es
- Ayuntamiento de Málaga: https://www.malaga.eu
- Primary laws: Decree 28/2016 and Decree-Law 1/2025 (published in the BOJA)

---

## How this maps to UC2

In **all three cities**, a typical central flat now mostly **cannot obtain a new short-term-rental licence** (Madrid centre: no dispersed VUT; Barcelona: no new HUT + 2028 phase-out; Málaga: city-wide moratorium). This makes the regulatory layer **decision-changing**: the financial engine may show Airbnb earning more, while the regulatory agent flags that a *new* legal STR isn't permitted — which is exactly the responsible, governed output the project is built to deliver.

---

## Disclaimer

This corpus supports an academic capstone and provides **decision support only**. It is **not legal advice**. Regulations are indicative and must be verified against current official municipal and regional sources at the time of use.
