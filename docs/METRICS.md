# Metrics & Parameters Reference

This is the complete reference for every number WattWise collects and every number it
computes. For each one you get: what it is, the exact formula or capture rule, what feeds
it and when it is unavailable, honest typical ranges with their source, how it moves your
training state, what a good trend looks like, and where it stops being trustworthy.

Two ground rules run through everything here:

- **No fabricated numbers.** A metric that cannot be computed correctly is reported as
  *Unavailable* with a typed reason — never a zero, a default, or a plausible guess. If you
  see a value, it was actually computed from your data.
- **Honest ranges, cited.** Typical-value ranges are orientation, not diagnosis. Each one
  names its origin (a published convention or the math itself). Where no defensible
  reference range exists, this document says so rather than inventing one.

Canonical keys are shown in `code font` — these are the exact field names the API and the
coaching agent use, so a value you read here matches a value you see there.

A note on **fidelity**. Many readings carry a fidelity tag describing how they were
obtained. `raw_stream` and `device_computed` mean the value came from (or was reproduced
from) the per-second recording. `summary_only` means a source supplied only a summary
number with no underlying stream, so it is treated as lower-fidelity. `SUBSTITUTED` means a
value was recomputed from a lower-fidelity stand-in (for example, an HR-based load after a
power source went offline); it is always flagged, never presented as the original.

A note on **sport**. Power-based metrics (Normalized Power, Intensity Factor, power TSS,
W'balance, critical power, the power curve) are defined only for sports that produce true
mechanical power — cycling first. Requested for a sport without power, they return
*Not applicable for sport* and are omitted from that sport's view rather than faked. The
heart-rate family (TRIMP, HRV, HR-based decoupling) works for any sport that records heart
rate.

---

## Index

### Collected parameters

Activity scalars: [elapsed_time_s](#elapsed-time-elapsed_time_s) ·
[moving_time_s](#moving-time-moving_time_s) · [start_time](#start-time-start_time) ·
[sport](#sport-sport) · [sub_sport](#sub-sport-sub_sport) · [is_indoor](#indoor-flag-is_indoor) ·
[distance_m](#distance-distance_m) · [total_work_j](#total-mechanical-work-total_work_j) ·
[energy_kj](#session-energy-energy_kj) · [avg_power_w](#average-power-avg_power_w) ·
[max_power_w](#maximum-power-max_power_w) · [avg_hr_bpm](#average-heart-rate-avg_hr_bpm) ·
[max_hr_bpm](#maximum-heart-rate-max_hr_bpm) · [avg_cadence_rpm](#average-cadence-avg_cadence_rpm) ·
[avg_speed_mps](#average-speed-avg_speed_mps) · [elevation_gain_m](#elevation-gain-elevation_gain_m) ·
[avg_temp_c](#average-temperature-avg_temp_c) · [training_effect_aerobic](#aerobic-training-effect-training_effect_aerobic) ·
[anaerobic_effect](#anaerobic-training-effect-anaerobic_effect) ·
[vo2max_estimate](#per-activity-vo2max-estimate-vo2max_estimate) ·
[training_load_source](#source-reported-session-load-training_load_source) ·
[device_class](#device-class-device_class) · [has_power](#channel-presence-flags-has_power--has_hr--has_gps--has_cadence) ·
[has_hr](#channel-presence-flags-has_power--has_hr--has_gps--has_cadence) ·
[has_gps](#channel-presence-flags-has_power--has_hr--has_gps--has_cadence) ·
[has_cadence](#channel-presence-flags-has_power--has_hr--has_gps--has_cadence)

Daily wellness: [resting_hr_bpm](#resting-heart-rate-resting_hr_bpm) ·
[min_hr_bpm](#daily-minimum-heart-rate-min_hr_bpm) · [max_hr_bpm](#daily-maximum-heart-rate-max_hr_bpm) ·
[stress_avg](#average-stress-stress_avg) · [stress_max](#peak-stress-stress_max) ·
[body_battery_high](#body-battery-high-body_battery_high) · [body_battery_low](#body-battery-low-body_battery_low) ·
[steps](#steps-steps) · [active_s](#active-seconds-active_s) · [highly_active_s](#highly-active-seconds-highly_active_s) ·
[sedentary_s](#sedentary-seconds-sedentary_s) · [active_kcal](#active-calories-active_kcal) ·
[bmr_kcal](#resting-calories-bmr_kcal) · [total_kcal](#total-calories-total_kcal) ·
[distance_m (daily)](#daily-distance-distance_m) ·
[intensity_minutes_moderate](#moderate-intensity-minutes-intensity_minutes_moderate) ·
[intensity_minutes_vigorous](#vigorous-intensity-minutes-intensity_minutes_vigorous) ·
[intensity_minutes_goal](#intensity-minutes-goal-intensity_minutes_goal) ·
[floors_ascended](#floors-ascended-floors_ascended) · [floors_descended](#floors-descended-floors_descended) ·
[floors_ascended_m](#floors-ascended-metres-floors_ascended_m) ·
[floors_descended_m](#floors-descended-metres-floors_descended_m) ·
[respiration_avg_rpm](#average-respiration-respiration_avg_rpm) ·
[respiration_latest_rpm](#latest-respiration-respiration_latest_rpm) ·
[respiration_lowest_rpm](#lowest-respiration-respiration_lowest_rpm) ·
[respiration_highest_rpm](#highest-respiration-respiration_highest_rpm) ·
[spo2_avg_pct](#average-spo2-spo2_avg_pct) · [spo2_latest_pct](#latest-spo2-spo2_latest_pct) ·
[spo2_lowest_pct](#lowest-spo2-spo2_lowest_pct) · [sleep_score](#sleep-score-sleep_score) ·
[sleep_duration_s](#sleep-duration-sleep_duration_s) · [sleep_start](#sleep-start-sleep_start) ·
[sleep_end](#sleep-end-sleep_end) · [sleep_deep_s](#deep-sleep-sleep_deep_s) ·
[sleep_light_s](#light-sleep-sleep_light_s) · [sleep_rem_s](#rem-sleep-sleep_rem_s) ·
[sleep_awake_s](#awake-time-sleep_awake_s) · [hrv_rmssd_ms](#hrv-rmssd-hrv_rmssd_ms) ·
[hrv_sdnn_ms](#hrv-sdnn-hrv_sdnn_ms) · [hrv_pnn50_pct](#hrv-pnn50-hrv_pnn50_pct) ·
[hrv_weekly_avg_ms](#hrv-weekly-average-hrv_weekly_avg_ms) ·
[hrv_baseline_low_ms](#hrv-baseline-low-hrv_baseline_low_ms) ·
[hrv_baseline_high_ms](#hrv-baseline-high-hrv_baseline_high_ms) · [hrv_status](#hrv-status-hrv_status) ·
[hrv_method](#hrv-method-hrv_method) · [vo2max (daily)](#vo2max-snapshot-vo2max) ·
[fitness_age_years](#fitness-age-fitness_age_years) · [body_mass_kg](#body-mass-body_mass_kg) ·
[height_cm](#height-height_cm) · [ftp_watts (snapshot)](#ftp-snapshot-ftp_watts) ·
[lactate_threshold_hr_bpm](#lactate-threshold-heart-rate-lactate_threshold_hr_bpm) ·
[training_status](#training-status-training_status) · [training_load_balance](#training-load-balance-training_load_balance) ·
[acute_load](#source-acute-load-acute_load) · [chronic_load](#source-chronic-load-chronic_load) ·
[acwr](#acutechronic-workload-ratio-acwr) · [acwr_status](#acwr-status-acwr_status) ·
[load_aerobic_low](#low-aerobic-load-load_aerobic_low) · [load_aerobic_high](#high-aerobic-load-load_aerobic_high) ·
[load_anaerobic](#anaerobic-load-load_anaerobic) · [endurance_score (source)](#source-endurance-score-endurance_score) ·
[readiness_external](#external-readiness-readiness_external)

Thresholds (fitness signature): [ftp_w](#functional-threshold-power-ftp_w) ·
[cp_w](#critical-power-threshold-cp_w) · [w_prime_j](#anaerobic-work-capacity-w_prime_j) ·
[threshold_hr_bpm](#threshold-heart-rate-threshold_hr_bpm) · [max_hr_bpm (signature)](#maximum-heart-rate-threshold-max_hr_bpm) ·
[resting_hr_bpm (signature)](#resting-heart-rate-threshold-resting_hr_bpm) ·
[vo2max (signature)](#vo2max-threshold-vo2max) · [signature_type](#signature-sport-scope-signature_type) ·
[effective_date](#threshold-effective-date-effective_date) · [effective_to](#threshold-effective-to-effective_to) ·
[origin](#threshold-origin-origin)

Streams: [power_w](#power-stream-power_w) · [hr_bpm](#heart-rate-stream-hr_bpm) ·
[cadence_rpm](#cadence-stream-cadence_rpm) · [speed_mps](#speed-stream-speed_mps) ·
[altitude_m](#altitude-stream-altitude_m) · [distance_m (stream)](#distance-stream-distance_m) ·
[latlng](#position-stream-latlng) · [temp_c](#temperature-stream-temp_c) ·
[left_right_balance](#leftright-balance-stream-left_right_balance) · [smo2](#muscle-oxygen-stream-smo2) ·
[core_temp_c](#core-temperature-stream-core_temp_c) · [respiration_rpm](#respiration-stream-respiration_rpm) ·
[rr_intervals_ms](#rr-interval-stream-rr_intervals_ms)

### Computed metrics

Load family: [tss](#training-stress-score-tss) · [hr_load](#hr-based-load-hr_load) ·
[hr_load_zonal](#zone-weighted-hr-load-hr_load_zonal) ·
[load_model](#load-model-label-load_model) · [tss_per_hour](#load-density-tss_per_hour)

Performance Management Chart: [ctl](#chronic-training-load-ctl) · [atl](#acute-training-load-atl) ·
[tsb](#training-stress-balance-tsb) · [form](#form-form) ·
[weekly_load_target](#weekly-load-target-weekly_load_target) ·
[monthly_load_target](#monthly-load-target-monthly_load_target)

Power family: [np](#normalized-power-np) · [if_](#intensity-factor-if_) ·
[critical_power_w](#critical-power-critical_power_w) · [w_prime_j (computed)](#anaerobic-work-capacity-fit-w_prime_j) ·
[power_curve](#power-curve-power_curve) · [wbal](#wbalance-wbal)

Efficiency family: [efficiency_factor](#efficiency-factor-efficiency_factor) ·
[variability_index](#variability-index-variability_index) · [intensity_class](#intensity-class-intensity_class) ·
[decoupling](#aerobic-decoupling-decoupling)

Heart-rate family: [trimp](#trimp-trimp) · [hrv_rmssd_ms (computed)](#computed-rmssd-hrv_rmssd_ms)

Composites: [endurance_score](#endurance-score-endurance_score)

### Arriving in an upcoming release

These keys are part of the canonical model but are **not collected or computed by current
builds** — they land with an upcoming release. They are listed separately so nothing here is
mistaken for a value you can read today: [perceived_exertion](#session-rpe-perceived_exertion) ·
[feel](#session-feel-feel) · [srpe_load](#session-rpe-load-srpe_load)

---

# Collected parameters

These are the values WattWise stores from your devices, your connected platforms, and your
own self-reports. They are standardized into source-neutral fields with explicit units. A
collected value is what was observed; the computed metrics below are derived from it.

## Activity scalars

One activity is one continuous training session, in canonical form, no matter how many
sources reported it. Summary scalars are reproduced from the per-second stream when a
stream exists; when a source supplied only a summary, the scalar is tagged `summary_only`.

### Elapsed time (`elapsed_time_s`)

**Units:** seconds.

Wall-clock duration of the session from start to finish, including stops and pauses.

**Capture rule.** Standardized directly from the source's session record. It is not
recomputed from the stream.

**Inputs & when unavailable.** Present whenever the source reports a session duration.
Absent (typed null) when no duration was reported.

**Typical values.** Spans from a few minutes to many hours, by session. No reference range
applies — it is a duration, not a physiological measure.

**How it moves state.** Context for reading other numbers; not itself a load input. The
exercise duration that drives load is computed separately (see `tss`).

**Reading trends.** Compare against `moving_time_s`: a large gap means a lot of stopped
time (cafe stops, traffic, long descents).

**Caveats.** Includes non-moving time, so it overstates training duration. Do not use it as
the denominator for intensity.

### Moving time (`moving_time_s`)

**Units:** seconds.

Duration during which you were actually moving, with stopped time removed.

**Capture rule.** Standardized from the source's moving-time summary. Distinct from the
engine-derived valid-moving duration used for load (see `tss`), which is computed from the
stream.

**Inputs & when unavailable.** Present when the source reports moving time; otherwise a
typed null.

**Typical values.** Less than or equal to `elapsed_time_s`. No physiological reference
range applies.

**How it moves state.** Context only; the canonical load math uses its own stream-derived
valid-moving duration, not this summary.

**Reading trends.** A stable ratio of moving to elapsed time across similar rides indicates
consistent conditions.

**Caveats.** Source definitions of "moving" vary (speed threshold, auto-pause behaviour),
so treat cross-source comparisons cautiously.

### Start time (`start_time`)

**Units:** UTC timestamp.

The instant the session began, stored in UTC.

**Capture rule.** Standardized from the source's session start. Your local wall-clock and
local calendar day are derived from it using your reference timezone.

**Inputs & when unavailable.** Always present — it anchors the activity in time.

**Typical values.** Not applicable (a timestamp).

**How it moves state.** Determines which local calendar day the activity belongs to, which
in turn places its load on the correct day of the Performance Management Chart.

**Reading trends.** Not applicable.

**Caveats.** Day attribution uses your local date, not the UTC date, so a late-night
session lands on the correct local day.

### Sport (`sport`)

**Units:** registry code (for example `cycling`, `running`, `swimming`, `rowing`,
`xc_ski`, `strength`, `other`).

The canonical sport of the session, from a pluggable registry that can grow without a
schema change.

**Capture rule.** Mapped from the source's activity type to a canonical sport code. An
unmappable value maps to the registered `other` member.

**Inputs & when unavailable.** Always present (required).

**Typical values.** Not applicable (a category).

**How it moves state.** Decides which metrics apply. Power metrics run only for sports with
true mechanical power; the heart-rate family runs for any sport with heart rate. Aggregates
(such as the power curve and daily load) are partitioned by sport, never pooled across
incommensurable sports.

**Reading trends.** Not applicable.

**Caveats.** A wrong sport tag can suppress applicable metrics or hide a session from a
sport-scoped view. The sport drives applicability, never the computation itself.

### Sub-sport (`sub_sport`)

**Units:** registry code (for example `virtual_ride`, `track`, `trail_run`, `gravel`,
`indoor`).

A finer classification within a sport.

**Capture rule.** Mapped to the sub-sport registry; unmappable values map to `other`; null
when none is reported.

**Inputs & when unavailable.** Optional; null when the source gives no sub-classification.

**Typical values.** Not applicable (a category).

**How it moves state.** Context for interpretation (for example, distinguishing an indoor
trainer ride from an outdoor ride); it does not change a metric's formula.

**Reading trends.** Useful for filtering like-for-like sessions.

**Caveats.** Coverage varies widely by source.

### Indoor flag (`is_indoor`)

**Units:** boolean.

Whether the session was indoors.

**Capture rule.** Standardized from the source flag; null when unknown.

**Inputs & when unavailable.** Optional; null when not reported.

**Typical values.** Not applicable.

**How it moves state.** Context only.

**Reading trends.** Indoor efforts often lack GPS and may run hotter, which can affect
heart rate and decoupling — useful when comparing.

**Caveats.** Some sources never set it.

### Distance (`distance_m`)

**Units:** metres.

Total distance covered in the session.

**Capture rule.** Standardized from the source summary; reproducible from the distance
stream where present.

**Inputs & when unavailable.** Present for GPS or wheel-/foot-pod sessions; null for
fixed-position indoor sessions without a distance signal.

**Typical values.** Sport- and duration-dependent; no reference range applies.

**How it moves state.** Not a load input; context and a denominator for average speed.

**Reading trends.** Compare against time and elevation to characterize a session.

**Caveats.** Indoor trainers may report virtual distance that depends on the simulation.

### Total mechanical work (`total_work_j`)

**Units:** joules.

The mechanical work done, integrated from the power stream.

**Capture rule.** Standardized from a source summary, ideally reproduced as the integral of
power over time.

**Inputs & when unavailable.** Requires power; null without a power signal.

**Typical values.** A one-hour ride at 200 W is about 720,000 J (0.72 MJ); ranges scale
with power and duration. The relation is exact, not a reference band.

**How it moves state.** Context for energy expenditure; not a direct load input.

**Reading trends.** Tracks total physical output; rises with both intensity and duration.

**Caveats.** Distinct from `energy_kj`, which is a source-reported energy summary that may
include metabolic assumptions.

### Session energy (`energy_kj`)

**Units:** kilojoules.

The source's reported energy expenditure for the session.

**Capture rule.** Standardized from a source's calories/energy summary. A session-level
summary, distinct from the per-sample-derived `total_work_j`.

**Inputs & when unavailable.** Present when the source reports it; otherwise null.

**Typical values.** For cycling, kilojoules of work roughly equal kilocalories burned
because human efficiency on a bike is near 24% and the unit conversion (4.184) nearly
cancels it — a widely used field convention, not an exact identity.

**How it moves state.** Context only; not a canonical load input.

**Reading trends.** Useful for fuelling discussions over long sessions.

**Caveats.** Source energy models differ; treat it as approximate.

### Average power (`avg_power_w`)

**Units:** watts.

Mean mechanical power over the session.

**Capture rule.** Standardized from the source summary; reproduced as the arithmetic mean
of the valid power stream when a stream exists. When only a summary is present it is tagged
`summary_only`.

**Inputs & when unavailable.** Requires a power meter or smart trainer; null otherwise.

**Typical values.** Highly individual. For trained cyclists, hour-long efforts commonly sit
in the 150-300 W range; this is orientation only and depends entirely on the athlete.

**How it moves state.** Feeds the Variability Index denominator and supports interpreting
Normalized Power. It is not itself the load.

**Reading trends.** At equal heart rate, rising average power over weeks suggests improving
fitness.

**Caveats.** Average power understates the cost of variable efforts — that is exactly what
Normalized Power corrects for.

### Maximum power (`max_power_w`)

**Units:** watts.

The highest one-second power in the session.

**Capture rule.** Standardized from the source; reproducible as the stream maximum. The
chart decimation that thins streams for display always preserves the true peak, so a
displayed chart never contradicts this value.

**Inputs & when unavailable.** Requires power; null otherwise.

**Typical values.** Sprint peaks vary enormously by rider and effort; no reference range
applies.

**How it moves state.** Context; not a load input.

**Reading trends.** Track sprint peaks over a season as a neuromuscular marker.

**Caveats.** Sensitive to a single spike; a dropout or spike can distort it.

### Average heart rate (`avg_hr_bpm`)

**Units:** beats per minute.

Mean heart rate over the session.

**Capture rule.** Standardized from the source; reproduced as the mean of the valid heart-
rate stream over the valid-moving window. This is the same denominator used by the
Efficiency Factor.

**Inputs & when unavailable.** Requires a heart-rate signal; null otherwise.

**Typical values.** Individual; commonly 120-160 bpm for steady endurance work, but this is
orientation and depends on the athlete's heart-rate range.

**How it moves state.** Denominator for the Efficiency Factor; input to interpreting
intensity.

**Reading trends.** At equal power, a lower average heart rate over time suggests improving
aerobic fitness.

**Caveats.** Affected by heat, caffeine, stress, dehydration, and cardiac drift on long
efforts.

### Maximum heart rate (`max_hr_bpm`)

**Units:** beats per minute.

The highest heart rate recorded in the session.

**Capture rule.** Standardized from the source; reproducible as the stream maximum.

**Inputs & when unavailable.** Requires a heart-rate signal; null otherwise.

**Typical values.** Bounded by your true maximum heart rate; orientation only.

**How it moves state.** Context; not a load input.

**Reading trends.** A session max near your known maximum confirms a genuinely hard effort.

**Caveats.** Strap dropouts and electrical interference produce spurious spikes. This is the
session maximum, not your physiological maximum (that lives in the fitness signature).

### Average cadence (`avg_cadence_rpm`)

**Units:** revolutions per minute (or steps per minute, by sport).

Mean cadence over the session.

**Capture rule.** Standardized from the source; reproducible from the cadence stream.

**Inputs & when unavailable.** Requires a cadence sensor; null otherwise.

**Typical values.** Cycling endurance cadence is commonly 80-95 rpm; running cadence
commonly 160-180 spm. Orientation only; strongly individual.

**How it moves state.** Context only.

**Reading trends.** Stable cadence at a given effort indicates settled pacing.

**Caveats.** Coasting periods can depress the average if included.

### Average speed (`avg_speed_mps`)

**Units:** metres per second.

Mean speed over the session.

**Capture rule.** Standardized from the source; reproducible from the speed stream.

**Inputs & when unavailable.** Requires a speed or GPS signal; null otherwise.

**Typical values.** Conditions-dependent; no reference range applies.

**How it moves state.** For sports without power, speed is the output channel used by
aerobic decoupling.

**Reading trends.** Useful only alongside terrain, wind, and surface — raw speed is a poor
fitness signal on its own.

**Caveats.** Indoor virtual speed depends on the trainer simulation.

### Elevation gain (`elevation_gain_m`)

**Units:** metres.

Total ascent over the session.

**Capture rule.** Standardized from the source; reproducible from the altitude stream.

**Inputs & when unavailable.** Requires a barometric or GPS altitude signal; null
otherwise.

**Typical values.** Route-dependent; no reference range applies.

**How it moves state.** Context only.

**Reading trends.** Helps explain why a ride felt hard at modest average power.

**Caveats.** Barometric drift and GPS noise inflate gain; source smoothing differs.

### Average temperature (`avg_temp_c`)

**Units:** degrees Celsius.

Mean ambient (or device) temperature during the session.

**Capture rule.** Standardized from the source's temperature summary.

**Inputs & when unavailable.** Requires a temperature sensor; null otherwise.

**Typical values.** Ambient range; no reference range applies.

**How it moves state.** Context only, but an important confounder for heart-rate-based
readings.

**Reading trends.** Hot sessions push heart rate up at equal power, which can look like a
fitness loss if temperature is ignored.

**Caveats.** Device-mounted sensors read warmer than true ambient when in sunlight.

### Aerobic training effect (`training_effect_aerobic`)

**Units:** unitless score (typically 0-5).

A source-reported score for the aerobic stimulus of the session.

**Capture rule.** Retained as a typed source summary. WattWise does not recompute it and
does not treat it as a canonical metric.

**Inputs & when unavailable.** Present only when the source provides it; otherwise null.

**Typical values.** 0-5 on the common device scale, where the source defines the bands.

**How it moves state.** Advisory context only; it does not feed the Performance Management
Chart.

**Reading trends.** Use the source's own interpretation of its bands.

**Caveats.** Vendor-specific and not comparable across sources. Not a substitute for the
canonical load.

### Anaerobic training effect (`anaerobic_effect`)

**Units:** unitless score (typically 0-5).

A source-reported score for the anaerobic stimulus of the session.

**Capture rule.** Retained as a typed source summary; never recomputed or treated as
canonical.

**Inputs & when unavailable.** Present only when the source provides it.

**Typical values.** 0-5 on the common device scale.

**How it moves state.** Advisory context only.

**Reading trends.** Use the source's own band interpretation.

**Caveats.** Vendor-specific; not cross-comparable.

### Per-activity VO2max estimate (`vo2max_estimate`)

**Units:** millilitres per kilogram per minute.

A source's per-activity estimate of maximal oxygen uptake.

**Capture rule.** Retained as a source summary. The authoritative effective-dated VO2max
lives in the fitness signature, not here.

**Inputs & when unavailable.** Present only when the source estimates it.

**Typical values.** Recreational athletes commonly 35-50; well-trained endurance athletes
can exceed 60-70 (population conventions; individual).

**How it moves state.** Advisory; not a canonical input.

**Reading trends.** Estimates are noisy session to session; trust the trend, not a single
value.

**Caveats.** Algorithm differs by vendor; treat as approximate.

### Source-reported session load (`training_load_source`)

**Units:** unitless load score.

The source's own pre-computed session load (its TSS, XSS, or equivalent).

**Capture rule.** Retained as a typed summary for audit and conflict use only. The
canonical training load is computed by WattWise and is never read from this field — this
field exists precisely so a source's number cannot masquerade as canonical truth.

**Inputs & when unavailable.** Present only when the source reports a load.

**Typical values.** Depends on the source's scale; not directly comparable to canonical
TSS.

**How it moves state.** None. It does not feed the Performance Management Chart.

**Reading trends.** Useful only for reconciling with a source's own dashboards.

**Caveats.** Different sources use different load models; never mix this with canonical
load.

### Device class (`device_class`)

**Units:** category (`powermeter`, `trainer`, `gps_watch`, `phone`, `estimated`,
`unknown`).

The kind of device that produced the measurement — provenance of measurement, not the
name of the source platform.

**Capture rule.** Standardized to a fixed set of measurement-provenance classes.

**Inputs & when unavailable.** Present when the device kind is known; `unknown` otherwise.

**Typical values.** Not applicable (a category).

**How it moves state.** Helps explain fidelity; an `estimated` class flags a value as
lower-confidence.

**Reading trends.** Not applicable.

**Caveats.** It is a measurement class, never a source/brand identity.

### Channel-presence flags (`has_power` / `has_hr` / `has_gps` / `has_cadence`)

**Units:** booleans.

Whether the session carries each of the power, heart-rate, GPS, and cadence channels.

**Capture rule.** Set from the resolved stream set: true when the corresponding canonical
channel is present.

**Inputs & when unavailable.** Always present (default false).

**Typical values.** Not applicable.

**How it moves state.** Determine which metrics can be computed. `has_power` gates the power
family; `has_hr` gates the heart-rate family.

**Reading trends.** Not applicable.

**Caveats.** A flag reflects channel presence, not channel quality — a sparse or
gap-riddled channel still sets the flag.

## Daily wellness

Exactly one reconciled wellness row exists per athlete per local day. These are
source-reported physiology and lifestyle fields. They are never canonical fitness state —
the canonical Performance Management Chart is computed separately and lives elsewhere.

### Resting heart rate (`resting_hr_bpm`)

**Units:** beats per minute.

Your resting heart rate for the day, as reported by a wearable.

**Capture rule.** Standardized from the source's daily resting heart rate.

**Inputs & when unavailable.** Requires a wearable that measures it; null otherwise.

**Typical values.** Commonly 40-60 bpm in trained endurance athletes, 60-80 in the general
population (population conventions; individual).

**How it moves state.** A recovery and readiness signal; an elevated resting heart rate can
indicate fatigue or illness.

**Reading trends.** Track against your own baseline; an upward deviation of several beats is
more informative than the absolute value.

**Caveats.** Affected by alcohol, late meals, heat, and measurement timing. Device methods
differ.

### Daily minimum heart rate (`min_hr_bpm`)

**Units:** beats per minute.

The lowest heart rate recorded during the day.

**Capture rule.** Standardized from the source daily summary.

**Inputs & when unavailable.** Requires continuous heart-rate monitoring; null otherwise.

**Typical values.** Often near or below resting heart rate; individual.

**How it moves state.** Context only.

**Reading trends.** Usually tracks resting heart rate.

**Caveats.** Sensitive to brief artefacts during sleep.

### Daily maximum heart rate (`max_hr_bpm`)

**Units:** beats per minute.

The highest heart rate recorded during the day (across all activity).

**Capture rule.** Standardized from the source daily summary.

**Inputs & when unavailable.** Requires continuous monitoring; null otherwise.

**Typical values.** Reflects the hardest moment of the day; individual.

**How it moves state.** Context only.

**Reading trends.** Not a fitness signal on its own.

**Caveats.** This is a daily-wellness field, distinct from an activity's maximum and from
your physiological maximum in the signature.

### Average stress (`stress_avg`)

**Units:** unitless score (typically 0-100).

A wearable's average daily "stress" score, usually derived from heart-rate variability.

**Capture rule.** Standardized from the source; retained as a vendor score.

**Inputs & when unavailable.** Vendor-specific; null when not reported.

**Typical values.** 0-100 on the common vendor scale, where higher means more
physiological stress.

**How it moves state.** Advisory recovery context only.

**Reading trends.** Sustained high daily stress can accompany under-recovery.

**Caveats.** Proprietary and not comparable across vendors.

### Peak stress (`stress_max`)

**Units:** unitless score (typically 0-100).

The day's highest stress score.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** 0-100 vendor scale.

**How it moves state.** Advisory context only.

**Reading trends.** Usually elevated on hard training or high-stress days.

**Caveats.** Proprietary; not cross-comparable.

### Body Battery high (`body_battery_high`)

**Units:** unitless score (0-100).

The day's peak of a wearable's energy-reserve estimate.

**Capture rule.** Standardized from the source; retained as a vendor score.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** 0-100, where 100 is fully charged (vendor convention).

**How it moves state.** Advisory recovery context only.

**Reading trends.** A high morning peak suggests good overnight recovery.

**Caveats.** Proprietary model; orientation only.

### Body Battery low (`body_battery_low`)

**Units:** unitless score (0-100).

The day's trough of the energy-reserve estimate.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** 0-100 (vendor convention).

**How it moves state.** Advisory context only.

**Reading trends.** A very low trough indicates a draining day.

**Caveats.** Proprietary; orientation only.

### Steps (`steps`)

**Units:** count.

Total steps for the day.

**Capture rule.** Standardized from the source daily summary.

**Inputs & when unavailable.** Requires a step-counting device; null otherwise.

**Typical values.** General activity guidance often cites several thousand steps a day; no
training reference range applies.

**How it moves state.** Non-training activity context; not a load input.

**Reading trends.** A proxy for daily movement outside structured training.

**Caveats.** Cycling and swimming under-count steps badly.

### Active seconds (`active_s`)

**Units:** seconds.

Time the wearable classified as active during the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Lifestyle-activity context only.

**Reading trends.** Useful for spotting low-movement days.

**Caveats.** Vendor activity thresholds differ.

### Highly active seconds (`highly_active_s`)

**Units:** seconds.

Time classified as highly active.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Context only.

**Reading trends.** Tracks vigorous non-training movement.

**Caveats.** Vendor-defined threshold.

### Sedentary seconds (`sedentary_s`)

**Units:** seconds.

Time classified as sedentary.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Context only.

**Reading trends.** High sedentary time can matter for general health, not training load.

**Caveats.** Vendor-defined.

### Active calories (`active_kcal`)

**Units:** kilocalories.

Estimated calories burned through activity during the day.

**Capture rule.** Standardized from the source's estimate.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Depends on body size and activity; no reference range applies.

**How it moves state.** Fuelling context only.

**Reading trends.** Useful for energy-balance discussions.

**Caveats.** All wearable calorie estimates are approximate.

### Resting calories (`bmr_kcal`)

**Units:** kilocalories.

Estimated basal/resting energy expenditure for the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Depends on body size, age, and sex; no single reference applies.

**How it moves state.** Fuelling context only.

**Reading trends.** Roughly stable day to day.

**Caveats.** Modelled, not measured.

### Total calories (`total_kcal`)

**Units:** kilocalories.

Estimated total daily energy expenditure (resting plus active).

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** No single reference applies.

**How it moves state.** Fuelling context only.

**Reading trends.** Higher on heavy training days.

**Caveats.** Approximate.

### Daily distance (`distance_m`)

**Units:** metres.

Total distance covered across the whole day (all movement).

**Capture rule.** Standardized from the source daily summary. Distinct from an activity's
distance and from the distance stream channel.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Lifestyle-activity context only.

**Reading trends.** A movement proxy.

**Caveats.** Aggregates all-day movement, not just training.

### Moderate intensity minutes (`intensity_minutes_moderate`)

**Units:** minutes.

Minutes of moderate-intensity activity counted by the wearable.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Public-health guidance often targets weekly moderate-activity minutes;
no training reference range applies.

**How it moves state.** Lifestyle context only.

**Reading trends.** Tracks general activity guidelines.

**Caveats.** Vendor-defined thresholds.

### Vigorous intensity minutes (`intensity_minutes_vigorous`)

**Units:** minutes.

Minutes of vigorous-intensity activity.

**Capture rule.** Standardized from the source. Many vendors weight vigorous minutes double
toward a goal.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** No training reference range applies.

**How it moves state.** Lifestyle context only.

**Reading trends.** Tracks vigorous activity.

**Caveats.** Vendor-defined.

### Intensity minutes goal (`intensity_minutes_goal`)

**Units:** minutes.

The wearable's target for weekly intensity minutes.

**Capture rule.** Standardized from the source's goal setting.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** A user/vendor setting, not a measurement.

**How it moves state.** Context only.

**Reading trends.** Not applicable.

**Caveats.** A goal, not an observation.

### Floors ascended (`floors_ascended`)

**Units:** count of floors.

Floors climbed during the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a barometric altimeter; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Lifestyle context only.

**Reading trends.** A movement proxy.

**Caveats.** Barometric noise affects the count.

### Floors descended (`floors_descended`)

**Units:** count of floors.

Floors descended during the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a barometric altimeter; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Context only.

**Reading trends.** A movement proxy.

**Caveats.** Barometric noise affects the count.

### Floors ascended (metres) (`floors_ascended_m`)

**Units:** metres.

Vertical metres climbed during the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a barometric altimeter; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Context only.

**Reading trends.** A movement proxy.

**Caveats.** Barometric noise.

### Floors descended (metres) (`floors_descended_m`)

**Units:** metres.

Vertical metres descended during the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a barometric altimeter; null otherwise.

**Typical values.** No reference range applies.

**How it moves state.** Context only.

**Reading trends.** A movement proxy.

**Caveats.** Barometric noise.

### Average respiration (`respiration_avg_rpm`)

**Units:** breaths per minute.

Average respiration rate for the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a device that measures respiration; null otherwise.

**Typical values.** Resting adult respiration is commonly 12-20 breaths per minute
(general physiology convention).

**How it moves state.** Advisory recovery context only.

**Reading trends.** An elevated resting respiration can accompany stress or illness.

**Caveats.** Estimated from other signals on most wearables.

### Latest respiration (`respiration_latest_rpm`)

**Units:** breaths per minute.

The most recent respiration reading.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires respiration measurement; null otherwise.

**Typical values.** 12-20 breaths per minute at rest (general physiology convention).

**How it moves state.** Context only.

**Reading trends.** A point reading, less useful than the average.

**Caveats.** Estimated on most devices.

### Lowest respiration (`respiration_lowest_rpm`)

**Units:** breaths per minute.

The lowest respiration reading of the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires respiration measurement; null otherwise.

**Typical values.** Typically during deep sleep; individual.

**How it moves state.** Context only.

**Reading trends.** Tracks sleep depth loosely.

**Caveats.** Estimated.

### Highest respiration (`respiration_highest_rpm`)

**Units:** breaths per minute.

The highest respiration reading of the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires respiration measurement; null otherwise.

**Typical values.** Elevated during exertion; individual.

**How it moves state.** Context only.

**Reading trends.** Not a fitness signal on its own.

**Caveats.** Estimated.

### Average SpO2 (`spo2_avg_pct`)

**Units:** percent.

Average blood-oxygen saturation for the day.

**Capture rule.** Standardized from the source's pulse-oximetry summary.

**Inputs & when unavailable.** Requires a SpO2-capable device; null otherwise.

**Typical values.** 95-100% at sea level is the usual healthy range; lower at altitude
(general physiology convention; not a medical assessment).

**How it moves state.** Advisory context only.

**Reading trends.** Drops are expected at altitude; a persistent low reading at sea level
warrants attention from a clinician, not this tool.

**Caveats.** Wrist pulse oximetry is noisy; not a medical device.

### Latest SpO2 (`spo2_latest_pct`)

**Units:** percent.

The most recent blood-oxygen reading.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a SpO2-capable device; null otherwise.

**Typical values.** 95-100% at sea level (general physiology convention).

**How it moves state.** Context only.

**Reading trends.** A point reading.

**Caveats.** Noisy; not a medical device.

### Lowest SpO2 (`spo2_lowest_pct`)

**Units:** percent.

The lowest blood-oxygen reading of the day.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires a SpO2-capable device; null otherwise.

**Typical values.** Often during sleep; below 90% at sea level is unusual (general
physiology convention; not a diagnosis).

**How it moves state.** Context only.

**Reading trends.** Useful at altitude.

**Caveats.** Wrist measurement is unreliable at the low end.

### Sleep score (`sleep_score`)

**Units:** unitless score (typically 0-100).

A wearable's composite sleep-quality score.

**Capture rule.** Standardized from the source; retained as a vendor score.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** 0-100, higher is better (vendor convention).

**How it moves state.** Advisory recovery context only.

**Reading trends.** A run of poor scores can accompany under-recovery.

**Caveats.** Proprietary and not comparable across vendors.

### Sleep duration (`sleep_duration_s`)

**Units:** seconds.

Total time asleep.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires sleep tracking; null otherwise.

**Typical values.** Adult sleep guidance commonly cites 7-9 hours (general sleep-health
convention).

**How it moves state.** Recovery context only.

**Reading trends.** Chronic short sleep undermines adaptation to training.

**Caveats.** Wearable sleep staging is approximate.

### Sleep start (`sleep_start`)

**Units:** timestamp.

When sleep began.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires sleep tracking; null otherwise.

**Typical values.** Not applicable (a timestamp).

**How it moves state.** Context for sleep timing only.

**Reading trends.** Consistent timing supports recovery.

**Caveats.** Onset detection is approximate.

### Sleep end (`sleep_end`)

**Units:** timestamp.

When sleep ended.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Requires sleep tracking; null otherwise.

**Typical values.** Not applicable (a timestamp).

**How it moves state.** Context only.

**Reading trends.** Consistent wake time supports recovery.

**Caveats.** Approximate.

### Deep sleep (`sleep_deep_s`)

**Units:** seconds.

Time in deep (slow-wave) sleep.

**Capture rule.** Standardized from the source's sleep staging.

**Inputs & when unavailable.** Requires sleep staging; null otherwise.

**Typical values.** Often cited near 13-23% of total sleep (general sleep-staging
convention); individual.

**How it moves state.** Recovery context only.

**Reading trends.** Deep sleep is associated with physical recovery.

**Caveats.** Wearable staging is approximate; not polysomnography.

### Light sleep (`sleep_light_s`)

**Units:** seconds.

Time in light sleep.

**Capture rule.** Standardized from the source's staging.

**Inputs & when unavailable.** Requires staging; null otherwise.

**Typical values.** Usually the largest share of the night (general convention).

**How it moves state.** Recovery context only.

**Reading trends.** Less informative than deep or REM share.

**Caveats.** Approximate.

### REM sleep (`sleep_rem_s`)

**Units:** seconds.

Time in REM sleep.

**Capture rule.** Standardized from the source's staging.

**Inputs & when unavailable.** Requires staging; null otherwise.

**Typical values.** Often cited near 20-25% of total sleep (general convention);
individual.

**How it moves state.** Recovery context only.

**Reading trends.** REM is associated with cognitive recovery.

**Caveats.** Approximate.

### Awake time (`sleep_awake_s`)

**Units:** seconds.

Time awake during the sleep window.

**Capture rule.** Standardized from the source's staging.

**Inputs & when unavailable.** Requires staging; null otherwise.

**Typical values.** Brief awakenings are normal; individual.

**How it moves state.** Recovery context only.

**Reading trends.** Rising awake time signals fragmented sleep.

**Caveats.** Approximate.

### HRV RMSSD (`hrv_rmssd_ms`)

**Units:** milliseconds.

A source-reported heart-rate-variability summary, in the RMSSD statistic, for the day.

**Capture rule.** Standardized from the source. One field carries one statistic in one
unit; the `hrv_method` field records which variant a source provided. When only this
summary exists (no beat-to-beat intervals), it is surfaced at `summary_only` fidelity and
is not used to back-fill series-only metrics.

**Inputs & when unavailable.** Requires a source HRV reading; null otherwise.

**Typical values.** Strongly individual and method-dependent; commonly tens of
milliseconds. No universal reference range applies — read it against your own baseline.

**How it moves state.** A recovery signal. When a baseline band is available, a suppressed
reading can inform a readiness nudge; without a baseline the engine reads readiness from
training form alone rather than against a fabricated baseline.

**Reading trends.** Trends against your personal baseline matter far more than absolute
values; a sustained drop can indicate fatigue or illness.

**Caveats.** Highly sensitive to measurement conditions (posture, time of day, breathing).
This is the source's summary; the computed RMSSD entry below describes the value WattWise
derives from beat-to-beat data.

### HRV SDNN (`hrv_sdnn_ms`)

**Units:** milliseconds.

A source-reported HRV summary in the SDNN statistic.

**Capture rule.** Standardized from the source; one statistic, one unit.

**Inputs & when unavailable.** Requires a source SDNN reading; null otherwise.

**Typical values.** Individual and method-dependent; no universal reference range applies.

**How it moves state.** Recovery context.

**Reading trends.** Read against your own baseline.

**Caveats.** Window length strongly affects SDNN; comparisons need matched windows.

### HRV pNN50 (`hrv_pnn50_pct`)

**Units:** percent.

A source-reported HRV summary as pNN50 (the percentage of successive interval differences
greater than 50 ms).

**Capture rule.** Standardized from the source; one statistic, one unit.

**Inputs & when unavailable.** Requires a source pNN50 reading; null otherwise.

**Typical values.** Individual; no universal reference range applies.

**How it moves state.** Recovery context.

**Reading trends.** Read against your own baseline.

**Caveats.** Sensitive to artefacts and recording length.

### HRV weekly average (`hrv_weekly_avg_ms`)

**Units:** milliseconds.

A source-reported rolling weekly average of HRV.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Individual; no universal reference range applies.

**How it moves state.** Recovery context; a smoother signal than a single day.

**Reading trends.** The weekly average is more stable than daily readings for spotting a
genuine shift.

**Caveats.** Vendor-defined averaging window.

### HRV baseline low (`hrv_baseline_low_ms`)

**Units:** milliseconds.

The lower bound of the source's personal HRV baseline band.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Individual; defines the bottom of your normal range.

**How it moves state.** With the high bound, defines the baseline band the readiness HRV
nudge needs; a reading below the band can inform the verdict.

**Reading trends.** The band itself shifts slowly as fitness changes.

**Caveats.** Vendor-defined; meaningful only against the same vendor's readings.

### HRV baseline high (`hrv_baseline_high_ms`)

**Units:** milliseconds.

The upper bound of the source's personal HRV baseline band.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Individual; defines the top of your normal range.

**How it moves state.** With the low bound, defines the baseline band used by the readiness
nudge. The engine uses the midpoint when both bounds are present.

**Reading trends.** Shifts slowly with fitness.

**Caveats.** Vendor-defined.

### HRV status (`hrv_status`)

**Units:** category.

A source's qualitative HRV state (for example balanced, unbalanced, low).

**Capture rule.** Standardized to a typed status enum.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Not applicable (a category).

**How it moves state.** Advisory recovery context only.

**Reading trends.** Use the vendor's own interpretation.

**Caveats.** Proprietary classification.

### HRV method (`hrv_method`)

**Units:** category (`rmssd`, `sdnn`, `pnn50`).

A pointer recording which time-domain HRV statistic a source provided.

**Capture rule.** Standardized to a typed variant tag. It is deliberately distinct from the
spectral-method provenance recorded on computed frequency-domain HRV; the two never share a
vocabulary.

**Inputs & when unavailable.** Present when an HRV summary exists; null otherwise.

**Typical values.** Not applicable (a category).

**How it moves state.** Tells a reader which statistic the day's HRV summary is, so it is
never misread as a different statistic.

**Reading trends.** Not applicable.

**Caveats.** It labels the variant, not the value.

### VO2max snapshot (`vo2max`)

**Units:** millilitres per kilogram per minute.

A source-reported daily VO2max snapshot.

**Capture rule.** Standardized from the source. The authoritative effective-dated VO2max
lives in the fitness signature.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Recreational athletes commonly 35-50; well-trained endurance athletes
can exceed 60-70 (population conventions; individual).

**How it moves state.** Advisory context; not a canonical input.

**Reading trends.** Noisy day to day; trust the trend.

**Caveats.** Vendor estimate, not a lab test.

### Fitness age (`fitness_age_years`)

**Units:** years.

A wearable's "fitness age" estimate.

**Capture rule.** Standardized from the source's proprietary model.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** A vendor-derived figure relative to your chronological age.

**How it moves state.** Advisory context only.

**Reading trends.** Improves as fitness improves, per the vendor's model.

**Caveats.** Proprietary; orientation only.

### Body mass (`body_mass_kg`)

**Units:** kilograms.

Your body mass snapshot for the day.

**Capture rule.** Standardized from a connected scale or manual entry.

**Inputs & when unavailable.** Requires a weigh-in source; null otherwise.

**Typical values.** Individual; no reference range applies.

**How it moves state.** Context for power-to-weight discussion; not a canonical metric
input.

**Reading trends.** Track the multi-day trend, not single readings.

**Caveats.** Day-to-day weight is noisy (hydration, food).

### Height (`height_cm`)

**Units:** centimetres.

Your height snapshot.

**Capture rule.** Standardized from the source or manual entry.

**Inputs & when unavailable.** Null when never recorded.

**Typical values.** Individual; no reference range applies.

**How it moves state.** Context only.

**Reading trends.** Not applicable.

**Caveats.** Essentially static.

### FTP snapshot (`ftp_watts`)

**Units:** watts.

A source-reported functional threshold power snapshot for the day.

**Capture rule.** Standardized from the source as a daily snapshot. This is **not** the
authoritative threshold — analytics use the effective-dated `ftp_w` in the fitness
signature, never this snapshot.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Individual; orientation only.

**How it moves state.** None for canonical computation. It is retained for audit and
comparison only.

**Reading trends.** Compare against your authoritative signature value to spot drift.

**Caveats.** Easy to confuse with the canonical threshold — it is the source's snapshot,
not the value the engine computes against.

### Lactate threshold heart rate (`lactate_threshold_hr_bpm`)

**Units:** beats per minute.

A source-reported lactate-threshold heart rate.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Individual; often a high percentage of maximum heart rate.

**How it moves state.** Advisory context; the canonical threshold heart rate lives in the
signature.

**Reading trends.** Rises modestly as fitness improves.

**Caveats.** Estimated by the vendor, not measured in a lab.

### Training status (`training_status`)

**Units:** category.

A source's qualitative training-state label (for example productive, maintaining,
overreaching).

**Capture rule.** Standardized to a typed status enum. Source-reported; never the canonical
fitness state.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Not applicable (a category).

**How it moves state.** Advisory context only; the canonical state comes from the
Performance Management Chart.

**Reading trends.** Use the vendor's own interpretation.

**Caveats.** Proprietary; can disagree with the canonical chart.

### Training load balance (`training_load_balance`)

**Units:** unitless (vendor scale).

A source's acute-to-chronic load balance indicator.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined; not comparable to canonical form.

**How it moves state.** Advisory context only.

**Reading trends.** Use the vendor's interpretation.

**Caveats.** Proprietary; do not equate with canonical TSB.

### Source acute load (`acute_load`)

**Units:** unitless (vendor scale).

A source's own acute (short-term) training-load figure.

**Capture rule.** Standardized from the source; retained as a vendor figure.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined scale.

**How it moves state.** None for canonical computation; the canonical ATL is computed
separately.

**Reading trends.** Compare only within the same vendor.

**Caveats.** Not the canonical acute load.

### Source chronic load (`chronic_load`)

**Units:** unitless (vendor scale).

A source's own chronic (long-term) training-load figure.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined scale.

**How it moves state.** None for canonical computation; the canonical CTL is computed
separately.

**Reading trends.** Compare only within the same vendor.

**Caveats.** Not the canonical chronic load.

### Acute/chronic workload ratio (`acwr`)

**Units:** ratio.

A source's acute-to-chronic workload ratio.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** A commonly cited "sweet spot" sits near 0.8-1.3, with higher ratios
flagged as elevated risk (a widely used but debated sports-science convention).

**How it moves state.** Advisory context only.

**Reading trends.** Sharp spikes are the usual concern.

**Caveats.** The ratio's predictive value is contested in the literature; treat as
orientation.

### ACWR status (`acwr_status`)

**Units:** category.

A source's qualitative banding of the workload ratio.

**Capture rule.** Standardized to a typed status enum.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Not applicable (a category).

**How it moves state.** Advisory context only.

**Reading trends.** Use the vendor banding.

**Caveats.** Inherits the ratio's contested status.

### Low aerobic load (`load_aerobic_low`)

**Units:** unitless (vendor scale).

A source's low-aerobic share of training load.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined scale.

**How it moves state.** Advisory context only.

**Reading trends.** Tracks low-intensity volume per the vendor.

**Caveats.** Proprietary breakdown.

### High aerobic load (`load_aerobic_high`)

**Units:** unitless (vendor scale).

A source's high-aerobic share of training load.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined scale.

**How it moves state.** Advisory context only.

**Reading trends.** Tracks tempo/threshold volume per the vendor.

**Caveats.** Proprietary breakdown.

### Anaerobic load (`load_anaerobic`)

**Units:** unitless (vendor scale).

A source's anaerobic share of training load.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined scale.

**How it moves state.** Advisory context only.

**Reading trends.** Tracks high-intensity volume per the vendor.

**Caveats.** Proprietary breakdown.

### Source endurance score (`endurance_score`)

**Units:** unitless (vendor scale).

A source's own endurance-capacity score.

**Capture rule.** Standardized from the source and retained as a vendor figure. Distinct
from the WattWise computed endurance score described later.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Vendor-defined scale.

**How it moves state.** None for canonical computation.

**Reading trends.** Compare only within the same vendor.

**Caveats.** Do not confuse with the computed `endurance_score` metric.

### External readiness (`readiness_external`)

**Units:** unitless (vendor scale).

A source's own readiness or recovery score.

**Capture rule.** Standardized from the source.

**Inputs & when unavailable.** Vendor-specific; null otherwise.

**Typical values.** Often a 0-100 vendor scale.

**How it moves state.** Advisory context; the canonical readiness verdict is reasoned from
form and, when available, an HRV baseline.

**Reading trends.** Use the vendor interpretation.

**Caveats.** Proprietary; can disagree with the canonical readiness view.

## Thresholds (fitness signature)

The fitness signature holds your effective-dated reference values — the thresholds metrics
compute against. Each version carries an effective date range, so a metric for a past
activity uses the thresholds that were in force then, not today's.

### Functional Threshold Power (`ftp_w`)

**Units:** watts.

The power you can sustain at threshold, used to scale intensity and stress.

**Capture rule.** Stored as an effective-dated threshold (tested, modelled, or entered).
This authoritative value — not the daily `ftp_watts` snapshot — is what metrics use.

**Inputs & when unavailable.** Must be present and in-effect for the activity date;
absence makes Intensity Factor and power TSS unavailable.

**Typical values.** Highly individual. As watts per kilogram, recreational riders often
sit near 2-3, competitive riders 3.5-4.5, and elites above 5 (common cycling convention;
orientation only).

**How it moves state.** Scales Intensity Factor and therefore power TSS, and so flows into
the whole Performance Management Chart.

**Reading trends.** A rising FTP at stable training load is the clearest fitness gain.

**Caveats.** A stale or wrong FTP distorts every downstream power metric. Keep it current.

### Critical power threshold (`cp_w`)

**Units:** watts.

The asymptote of your power-duration curve — the highest power you can theoretically
sustain aerobically.

**Capture rule.** Stored as an effective-dated threshold, either entered or fit from your
power-duration data (see the computed critical-power entry).

**Inputs & when unavailable.** Required (with W') for W'balance; absence makes W'balance
unavailable. The engine never silently substitutes FTP for CP.

**Typical values.** Close to but usually a little below FTP for most riders; strongly
individual.

**How it moves state.** With W', it drives the W'balance model that tracks anaerobic
reserve through an effort.

**Reading trends.** Rises with aerobic development.

**Caveats.** Sensitive to the durations used in any fit; a poor fit is refused rather than
stored as a confident number.

### Anaerobic work capacity (`w_prime_j`)

**Units:** joules.

The fixed amount of work you can do above critical power before exhaustion — your
anaerobic "battery".

**Capture rule.** Stored as an effective-dated threshold, entered or fit alongside CP.

**Inputs & when unavailable.** Required (with CP) for W'balance.

**Typical values.** Commonly on the order of 10,000-30,000 J for trained cyclists
(orientation; strongly individual).

**How it moves state.** Sets the capacity of the W'balance tank: how much you can spend
above CP before the model shows depletion.

**Reading trends.** Reflects anaerobic capacity; changes with the right training.

**Caveats.** Harder to pin down than CP; depends on the quality of short maximal efforts in
the fit.

### Threshold heart rate (`threshold_hr_bpm`)

**Units:** beats per minute.

Your heart rate at lactate/functional threshold.

**Capture rule.** Stored as an effective-dated threshold.

**Inputs & when unavailable.** Optional reference; used for heart-rate zone context.

**Typical values.** Individual; often a high fraction of maximum heart rate.

**How it moves state.** Reference context for interpreting heart-rate intensity.

**Reading trends.** Rises modestly with aerobic fitness.

**Caveats.** Determined by testing protocol; keep it current.

### Maximum heart rate (threshold) (`max_hr_bpm`)

**Units:** beats per minute.

Your physiological maximum heart rate.

**Capture rule.** Stored as an effective-dated threshold (tested or entered).

**Inputs & when unavailable.** Required (with resting heart rate) for TRIMP; absence makes
TRIMP unavailable.

**Typical values.** Strongly individual and age-related; population formulas are only rough
estimates.

**How it moves state.** With resting heart rate, defines the heart-rate reserve used by
TRIMP; must satisfy maximum greater than resting or TRIMP reports an out-of-domain failure.

**Reading trends.** Declines slowly with age; not a fitness signal.

**Caveats.** Age-prediction formulas are inaccurate for individuals; prefer a tested value.

### Resting heart rate (threshold) (`resting_hr_bpm`)

**Units:** beats per minute.

Your reference resting heart rate, as a threshold parameter.

**Capture rule.** Stored as an effective-dated threshold. Distinct from the daily wellness
resting heart rate, which is an observation.

**Inputs & when unavailable.** Required (with maximum heart rate) for TRIMP.

**Typical values.** Commonly 40-60 bpm in trained endurance athletes (population
convention; individual).

**How it moves state.** The floor of the heart-rate reserve in TRIMP.

**Reading trends.** Falls as aerobic fitness improves.

**Caveats.** Use a stable reference, not a single noisy morning reading.

### VO2max (threshold) (`vo2max`)

**Units:** millilitres per kilogram per minute.

Your authoritative effective-dated maximal oxygen uptake.

**Capture rule.** Stored as an effective-dated threshold — the authoritative VO2max, unlike
the per-activity and daily snapshots.

**Inputs & when unavailable.** Optional reference value.

**Typical values.** Recreational athletes commonly 35-50; well-trained endurance athletes
can exceed 60-70 (population conventions; individual).

**How it moves state.** Reference context; not a direct input to the computed metrics here.

**Reading trends.** Rises with aerobic development.

**Caveats.** A lab test is the gold standard; estimates carry error.

### Signature sport scope (`signature_type`)

**Units:** sport-registry code.

The sport a signature version applies to.

**Capture rule.** Stored as the canonical sport code that scopes the threshold set.

**Inputs & when unavailable.** Always present on a signature row.

**Typical values.** Not applicable (a category).

**How it moves state.** Ensures a metric resolves thresholds for the activity's sport — a
running threshold never feeds a cycling metric.

**Reading trends.** Not applicable.

**Caveats.** Thresholds are sport-specific; the scope keeps them from crossing.

### Threshold effective date (`effective_date`)

**Units:** date.

The date a signature version takes effect.

**Capture rule.** Stored as the start of the version's effective interval.

**Inputs & when unavailable.** Always present.

**Typical values.** Not applicable (a date).

**How it moves state.** Selects which thresholds are in force for a given activity date, so
historical metrics use historical thresholds.

**Reading trends.** Not applicable.

**Caveats.** A version is in force from its effective date until the next version supersedes
it.

### Threshold effective-to (`effective_to`)

**Units:** timestamp (nullable).

When a signature version stops being in force.

**Capture rule.** Set when a newer version supersedes this one; null while the version is
current.

**Inputs & when unavailable.** Null for the currently effective version.

**Typical values.** Not applicable (a timestamp).

**How it moves state.** Closes a version's interval so a superseded threshold can never
shadow its successor.

**Reading trends.** Not applicable.

**Caveats.** A closed interval is exclusive of its end instant.

### Threshold origin (`origin`)

**Units:** category.

How a signature's values were obtained (for example tested, modelled, entered).

**Capture rule.** Stored as a typed origin enum. A modelled signature also carries its fit
quality; a poor modelled fit is refused rather than used.

**Inputs & when unavailable.** Always present.

**Typical values.** Not applicable (a category).

**How it moves state.** A modelled origin with a weak fit makes dependent metrics fail
closed instead of computing against an untrustworthy threshold.

**Reading trends.** Not applicable.

**Caveats.** Tested values are generally more reliable than modelled or entered ones.

## Streams

Streams are the per-second canonical channels decoded from your original recording files.
Each channel is named with its unit, gaps are kept explicit (never zero-filled), and
analytics resample to one sample per second before computing.

### Power stream (`power_w`)

**Units:** watts (per sample).

Mechanical power, second by second.

**Capture rule.** Decoded from the original recording into a canonical one-value-per-second
channel; gaps are explicit nulls, never zeros.

**Inputs & when unavailable.** Requires a power meter or smart trainer.

**Typical values.** Per-sample power spans from zero (coasting) to sprint peaks; no single
reference applies.

**How it moves state.** The foundation of the whole power family: Normalized Power, TSS,
W'balance, the power curve, critical power, and power-based decoupling all read this
channel.

**Reading trends.** The per-second detail is what separates a steady effort from a spiky
one.

**Caveats.** Dropouts appear as gaps; a window straddling a long gap is excluded from
metrics rather than interpolated across.

### Heart-rate stream (`hr_bpm`)

**Units:** beats per minute (per sample).

Heart rate, second by second.

**Capture rule.** Decoded into a canonical per-second channel with explicit gaps.

**Inputs & when unavailable.** Requires a heart-rate monitor.

**Typical values.** Per-sample heart rate spans your working range; individual.

**How it moves state.** Drives TRIMP, the Efficiency Factor denominator, and heart-rate-
based decoupling.

**Reading trends.** Cardiac drift over a long steady effort is what decoupling measures.

**Caveats.** Strap dropouts and interference create spikes and gaps; optical sensors lag on
hard intervals.

### Cadence stream (`cadence_rpm`)

**Units:** revolutions per minute (per sample).

Cadence, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a cadence sensor.

**Typical values.** Per-sample cadence ranges from zero (coasting) to high rpm; individual.

**How it moves state.** Context for interpreting power and effort; not a direct metric
input here.

**Reading trends.** Useful for pacing and pedalling analysis.

**Caveats.** Coasting reads as zero cadence.

### Speed stream (`speed_mps`)

**Units:** metres per second (per sample).

Speed, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a speed or GPS signal.

**Typical values.** Conditions-dependent; no reference applies.

**How it moves state.** For sports without power, speed is the output channel for aerobic
decoupling.

**Reading trends.** Only meaningful alongside terrain and conditions.

**Caveats.** Indoor virtual speed depends on the trainer simulation.

### Altitude stream (`altitude_m`)

**Units:** metres (per sample).

Elevation, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires barometric or GPS altitude.

**Typical values.** Route-dependent; no reference applies.

**How it moves state.** Context for interpreting effort on climbs.

**Reading trends.** Explains power and heart-rate patterns over terrain.

**Caveats.** Barometric drift; GPS altitude is noisier than barometric.

### Distance stream (`distance_m`)

**Units:** metres (per sample, cumulative).

Cumulative distance, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a distance or GPS signal.

**Typical values.** Monotonically increasing; no reference applies.

**How it moves state.** Supports speed and pace derivations.

**Reading trends.** Not a fitness signal on its own.

**Caveats.** Distinct from the activity and daily distance summaries.

### Position stream (`latlng`)

**Units:** latitude/longitude pairs (per sample).

Your geographic track, second by second.

**Capture rule.** Decoded into a canonical per-second channel. For map display, the track
is simplified with a shape-preserving algorithm that keeps turns and corners.

**Inputs & when unavailable.** Requires GPS.

**Typical values.** Not applicable (coordinates).

**How it moves state.** Context and mapping; not a metric input.

**Reading trends.** Not applicable.

**Caveats.** GPS noise; absent for indoor sessions.

### Temperature stream (`temp_c`)

**Units:** degrees Celsius (per sample).

Temperature, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a temperature sensor.

**Typical values.** Ambient range; no reference applies.

**How it moves state.** Context; an important confounder for heart-rate readings.

**Reading trends.** Heat pushes heart rate up at equal power.

**Caveats.** Device sensors read warm in direct sun.

### Left/right balance stream (`left_right_balance`)

**Units:** percent split (per sample).

The left/right pedalling power balance, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a dual-sided or balance-capable power meter.

**Typical values.** Near a 50/50 split for most riders; individual.

**How it moves state.** Context for pedalling analysis; not a metric input here.

**Reading trends.** A persistent asymmetry can prompt a bike-fit or injury review.

**Caveats.** Single-sided meters estimate rather than measure balance.

### Muscle oxygen stream (`smo2`)

**Units:** percent (per sample).

Muscle oxygen saturation, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a muscle-oxygen sensor.

**Typical values.** Drops under hard effort and recovers in rest; strongly individual and
sensor-dependent.

**How it moves state.** Advanced effort context; not a metric input here.

**Reading trends.** Used in advanced threshold and interval analysis.

**Caveats.** Placement-sensitive; specialist hardware.

### Core temperature stream (`core_temp_c`)

**Units:** degrees Celsius (per sample).

Estimated core body temperature, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a core-temperature sensor.

**Typical values.** Rises with sustained effort and heat; individual.

**How it moves state.** Heat-strain context; not a metric input here.

**Reading trends.** Used for heat-acclimation work.

**Caveats.** Usually modelled/estimated rather than directly measured.

### Respiration stream (`respiration_rpm`)

**Units:** breaths per minute (per sample).

Respiration rate, second by second.

**Capture rule.** Decoded into a canonical per-second channel.

**Inputs & when unavailable.** Requires a respiration-capable sensor.

**Typical values.** Rises with intensity; individual.

**How it moves state.** Effort context; not a metric input here.

**Reading trends.** Tracks ventilatory effort.

**Caveats.** Estimated on most consumer devices.

### RR-interval stream (`rr_intervals_ms`)

**Units:** milliseconds (per beat).

The beat-to-beat interval series — the time between heartbeats.

**Capture rule.** Decoded into a canonical channel. This is the raw material for computed
HRV: it is artifact-corrected into normal-to-normal intervals before any HRV statistic is
derived.

**Inputs & when unavailable.** Requires a device that records beat-to-beat data; without
it, HRV falls back to a source summary or reports unavailable.

**Typical values.** Intervals cluster around the inverse of heart rate (about 1000 ms at 60
bpm); individual.

**How it moves state.** Feeds the full computed HRV pipeline — time-domain statistics and,
where the signal-processing stack is present, frequency-domain band powers.

**Reading trends.** The variability between intervals, not the intervals themselves, is the
recovery signal.

**Caveats.** Artifact-laden recordings (above a correction ceiling) are reported as
insufficient rather than as a clean-looking number.

---

# Computed metrics

These are the numbers WattWise derives. Every one is computed by a pure function from
canonical inputs and returns a typed result: either a value with a quality report and
lineage, or a typed *Unavailable* reason. None of them is ever read from a source's
pre-computed field.

## Load family

Training load turns a session into a single comparable stress number that feeds the
Performance Management Chart. WattWise picks the load model automatically by what data
exists, in a fidelity order, and always labels which model produced a value.

### Training Stress Score (`tss`)

**Units:** load points (100 = one hour at threshold).

The power-based stress of a session, scaled so that one hour exactly at threshold equals
100 points.

**Formula.** `TSS = (duration_s * NP * IF) / (FTP * 3600) * 100`, equivalently
`(duration_s * NP^2) / (FTP^2 * 3600) * 100`, where `duration_s` is the valid moving
(exercise) duration — the seconds you were actually exercising, computed from the stream,
not wall-clock time. (Computed by the analytics engine, per the Coggan TSS definition.)

**Inputs & when unavailable.** Needs a power stream, an in-effect FTP, and a computable
Normalized Power. Unavailable if any is missing (for example no FTP), or not applicable for
sports without mechanical power.

**Typical values.** A recovery ride is well under 50; an endurance ride 50-100; a hard
threshold hour around 100; long hard days can exceed 200-300 (Coggan TSS convention;
orientation).

**How it moves state.** It is the default daily-load input to the Performance Management
Chart — the currency CTL, ATL, and form are built from.

**Reading trends.** Weekly TSS totals describe training volume-times-intensity; ramping
them gradually builds fitness, ramping them sharply builds fatigue.

**Caveats.** Only as good as your FTP. Genuine non-moving seconds (stops, long gaps) reduce
the valid duration and so reduce TSS, which is intended.

### HR-based load (`hr_load`)

**Units:** load points (commensurate with TSS).

The heart-rate-based stress of a session, used when there is no power. It is the canonical
TRIMP value (see the TRIMP entry) surfaced as a load.

**Formula.** The Banister heart-rate-reserve TRIMP (see `trimp`), reported as the session's
load on the heart-rate path. (Computed by the analytics engine, the Banister TRIMP surfaced as a load.)

**Inputs & when unavailable.** Needs a heart-rate stream and in-effect maximum and resting
heart rates. Unavailable without them.

**Typical values.** Scaled to be comparable with TSS for similar sessions; orientation
only.

**How it moves state.** Substitutes for power TSS in the daily load when power is absent, so
the Performance Management Chart stays continuous across sports and devices.

**Reading trends.** Read like TSS for volume-times-intensity, with the caveat that
heart-rate load is less precise on short, sharp efforts.

**Caveats.** It is always labelled as an HR-path load and never relabelled as power TSS. It
lags fast intensity changes because heart rate does.

### Zone-weighted HR load (`hr_load_zonal`)

**Units:** load points.

A heart-rate load variant that sums time in each heart-rate zone times a per-zone weight.

**Formula.** Time-in-zone times declared per-zone weights, with declared zone boundaries.
(Computed by the analytics engine.) It is a distinctly labelled alternative to the Banister
variant, never relabelled as it.

**Inputs & when unavailable.** Needs a heart-rate stream and declared heart-rate zone
boundaries. Unavailable without them.

**Typical values.** Depends on the zone scheme; orientation only.

**How it moves state.** Used as the day's heart-rate load only when you have chosen the
zonal model as your default; otherwise the Banister `hr_load` is used.

**Reading trends.** Emphasizes time at intensity according to the zone weights.

**Caveats.** Comparable only within the same zone definitions. It is produced only on
request or as your chosen default, never auto-selected ahead of `hr_load`.

### Load-model label (`load_model`)

**Units:** label (`power_tss`, `hr_load`, `hr_load_zonal`; `srpe_load` is added with the
upcoming session-RPE release described at the end of this reference).

The honest record of which load model produced a given load value.

**Formula.** Not a number — it names the selected member of the load-model set. Selection is
automatic and fidelity-ordered: power TSS first, then Banister HR load, then session-RPE
load, with the zonal HR variant chosen only as your default. (Computed by the analytics
engine.)

**Inputs & when unavailable.** Always populated whenever a load is reported, so the two load
families are never silently mixed.

**Typical values.** Not applicable (a label).

**How it moves state.** Lets every consumer see at what fidelity a day's load was produced,
and prevents an HR or RPE load from being read as power TSS.

**Reading trends.** A day labelled with a lower-fidelity model is a day to read with more
caution.

**Caveats.** A label outside the defined set is never valid.

### Load density (`tss_per_hour`)

**Units:** load points per hour.

How concentrated a session's stress was — load accrued per valid hour.

**Formula.** `tss_per_hour = tss / (duration_valid_s / 3600)`. Computed only when TSS and a
positive valid duration are present. (Computed by the analytics engine.)

**Inputs & when unavailable.** Needs a computed TSS and a positive valid duration;
unavailable otherwise.

**Typical values.** Roughly tracks the square of Intensity Factor: an easy hour is well
under 100 per hour, a threshold hour near 100, a hard short effort higher (by the formula).

**How it moves state.** A descriptor of intensity density on the activity detail; it is not
itself a Performance Management Chart input.

**Reading trends.** High load density means a short, intense session; low density a long,
easy one.

**Caveats.** Inherits TSS's dependence on a correct FTP.

## Performance Management Chart

The Performance Management Chart tracks fitness, fatigue, and form over time from your
daily load. It is the canonical training-state view — the source-reported training-state
fields above are never used here.

### Chronic Training Load (`ctl`)

**Units:** load points (an exponentially weighted average of daily load).

Your "fitness": a slow, 42-day exponentially weighted average of daily load.

**Formula.** `CTL(d) = CTL(d-1) + alpha * (L(d) - CTL(d-1))` with `alpha = 1 -
exp(-1/42)`, over a contiguous daily grid where a true rest day contributes zero load.
(Computed by the analytics engine, the Banister/Coggan Performance Management Chart model.)

**Inputs & when unavailable.** Built from the daily-load series, seeded from your true
training origin. Unavailable if a mid-history window cannot be seeded honestly (it will not
zero-seed and report a wrong-but-plausible fitness).

**Typical values.** Scales with sustained weekly load; an individual figure with no
universal reference range.

**How it moves state.** The backbone of fitness tracking and the basis of the maintenance
load targets; form is built from it.

**Reading trends.** Slowly rising CTL is building fitness. How fast you can raise it safely
is individual.

**Caveats.** Only as meaningful as the completeness of your daily load. Days overlapping an
open data gap are marked provisional and may revise.

### Acute Training Load (`atl`)

**Units:** load points (an exponentially weighted average of daily load).

Your "fatigue": a fast, 7-day exponentially weighted average of daily load.

**Formula.** `ATL(d) = ATL(d-1) + alpha * (L(d) - ATL(d-1))` with `alpha = 1 -
exp(-1/7)`. (Computed by the analytics engine, the Banister/Coggan Performance Management Chart model.)

**Inputs & when unavailable.** Same daily-load series and seeding as CTL.

**Typical values.** Rises and falls faster than CTL; individual, no universal reference.

**How it moves state.** With CTL, it determines form.

**Reading trends.** A spike after hard days that decays within about a week is normal
fatigue dynamics.

**Caveats.** Same completeness and provisional-day caveats as CTL.

### Training Stress Balance (`tsb`)

**Units:** load points (a difference of two averages).

Your "form": yesterday's fitness minus yesterday's fatigue.

**Formula.** `TSB(d) = CTL(d-1) - ATL(d-1)` — the previous-day balance, by definition.
(Computed by the analytics engine.)

**Inputs & when unavailable.** Derived from CTL and ATL; unavailable when they are.

**Typical values.** Negative during heavy training blocks, positive when rested and tapering
(a widely used interpretation; the exact bands are individual, so no fixed reference range
is asserted).

**How it moves state.** Informs readiness reasoning: positive form generally means rested,
deeply negative means heavily fatigued.

**Reading trends.** Form dips during a build and rises into a taper. The "right" race-day
form is individual.

**Caveats.** A blunt single number; pair it with how you actually feel and with HRV when
available.

### Form (`form`)

**Units:** load points.

An athlete-facing name for Training Stress Balance — the same value, not a second
computation.

**Formula.** Identical to `tsb`: `CTL(d-1) - ATL(d-1)`. (Computed by the analytics engine.) It is a
pure alias.

**Inputs & when unavailable.** Exactly as for `tsb`.

**Typical values.** As for `tsb`.

**How it moves state.** As for `tsb` — it is the same value under a friendlier name.

**Reading trends.** As for `tsb`.

**Caveats.** Reading "form" and "TSB" as two different numbers is the only trap; they are
one.

### Weekly load target (`weekly_load_target`)

**Units:** load points per week.

The weekly load that would hold your fitness steady.

**Formula.** `weekly_load_target = 7 * CTL` — holding CTL steady needs an average daily load
equal to CTL, so a week needs seven times it. (Computed by the analytics engine, derived from CTL.)

**Inputs & when unavailable.** Needs a computable CTL; unavailable otherwise.

**Typical values.** Seven times your current CTL; individual.

**How it moves state.** Grounds maintenance plan targets in your real canonical fitness
rather than an invented number.

**Reading trends.** Train above it to build, below it to detrain, at it to maintain.

**Caveats.** A maintenance reference, not a prescription; it does not account for intensity
distribution.

### Monthly load target (`monthly_load_target`)

**Units:** load points per four weeks.

The four-week load that would hold your fitness steady.

**Formula.** `monthly_load_target = 28 * CTL` — twenty-eight days at an average daily load
equal to CTL. (Computed by the analytics engine, derived from CTL.)

**Inputs & when unavailable.** Needs a computable CTL; unavailable otherwise.

**Typical values.** Twenty-eight times your current CTL; individual.

**How it moves state.** Grounds month-horizon maintenance plan targets in canonical
fitness.

**Reading trends.** As for the weekly target, over a four-week horizon.

**Caveats.** A maintenance reference, not a prescription.

## Power family

The power family quantifies the demand of an effort from the power stream, scaled by your
thresholds. All of it requires true mechanical power and is omitted for sports without it.

### Normalized Power (`np`)

**Units:** watts.

An adjusted average power that reflects the true physiological cost of a variable effort —
higher than plain average power when the effort is spiky.

**Formula.** Take a 30-second rolling average of power, raise it to the fourth power,
average that over the analysis window, then take the fourth root:
`NP = (mean(R(t)^4))^(1/4)` where `R(t)` is the 30-second rolling mean. (Computed by the
analytics engine, the Coggan Normalized Power definition.)

**Inputs & when unavailable.** Needs at least 30 contiguous valid seconds of power.
Unavailable with no power channel, or with fewer than 30 contiguous valid seconds.

**Typical values.** Equal to average power for perfectly steady efforts and progressively
higher as variability rises; individual.

**How it moves state.** Feeds Intensity Factor and therefore TSS, and is the smoothness
numerator for the Variability Index and the Efficiency Factor.

**Reading trends.** A large gap between Normalized and average power marks a stochastic
effort (a crit, a hilly group ride).

**Caveats.** The fourth-power weighting is sensitive to spikes; gaps longer than the
interpolation limit reset the rolling window rather than bridging it.

### Intensity Factor (`if_`)

**Units:** ratio (Normalized Power relative to threshold).

How hard a session was relative to your threshold.

**Formula.** `IF = NP / FTP`. (Computed by the analytics engine, the Coggan Intensity Factor.) It is never computed from average
power as a fallback.

**Inputs & when unavailable.** Needs a computed Normalized Power and an in-effect FTP.
Unavailable if either is missing.

**Typical values.** Coggan's conventional bands: under about 0.75 endurance, around 0.75-
0.85 tempo, around 0.85-0.95 threshold, around 0.95-1.05 around-threshold, above 1.05
toward maximal (Coggan IF convention; orientation).

**How it moves state.** Scales TSS quadratically and is banded into the intensity class.

**Reading trends.** A sustained Intensity Factor near or above 1.0 over a long effort is
exceptionally hard.

**Caveats.** Entirely dependent on a correct FTP; a stale FTP makes every Intensity Factor
wrong.

### Critical power (`critical_power_w`)

**Units:** watts.

The aerobic asymptote of your power-duration curve, fit from your real best efforts.

**Formula.** A two-parameter fit of `P(t) = W'/t + CP` (equivalently the linear work-time
form `W(t) = W' + CP * t`) to your maximal mean-power points over a valid duration range.
(Computed by the analytics engine, the Monod-Scherrer critical-power model.) The fit reports goodness-of-fit and is accepted only past
declared gates.

**Inputs & when unavailable.** Needs enough well-spread maximal efforts (by default at
least three distinct durations spanning a wide enough range, with a good fit). Unavailable
as insufficient data or poor fit otherwise; never fabricated.

**Typical values.** Close to but usually a little below FTP for most riders; strongly
individual.

**How it moves state.** With W', it parameterizes the W'balance model and can populate the
critical-power threshold.

**Reading trends.** Rises with aerobic development.

**Caveats.** Including efforts longer than the model's valid range biases critical power
high; such fits are flagged. A poor fit is reported unavailable, not clamped.

### Anaerobic work capacity (fit) (`w_prime_j`)

**Units:** joules.

The work-above-critical-power capacity, fit alongside critical power.

**Formula.** The `W'` parameter of the same two-parameter critical-power fit
(`W(t) = W' + CP * t`). (Computed by the analytics engine, the same critical-power fit.)

**Inputs & when unavailable.** Same fit and gates as critical power; unavailable when the
fit fails.

**Typical values.** Commonly on the order of 10,000-30,000 J for trained cyclists
(orientation; individual).

**How it moves state.** Sets the W'balance tank capacity.

**Reading trends.** Reflects anaerobic capacity; changes with targeted training.

**Caveats.** Less stable than critical power; depends on the quality of short maximal
efforts in the fit. (This is the computed fit value; the stored threshold of the same name
is described under thresholds.)

### Power curve (`power_curve`)

**Units:** watts by duration.

Your best sustainable average power for every duration — the power-duration envelope, and
the single source of your best efforts.

**Formula.** For each duration `d`, the maximum mean power over any valid window of at least
`d` seconds; over a date range, the per-duration maximum across activities, with lineage to
the activity that set each peak. (Computed by the analytics engine, the mean-maximal-power curve.) Best efforts are read
directly off this curve.

**Inputs & when unavailable.** Needs a power channel. A duration with no valid window is
reported unavailable for that duration while shorter durations remain computed; the whole
curve does not collapse.

**Typical values.** Non-increasing with duration (your best 5-second power exceeds your best
5-minute power); the actual values are individual.

**How it moves state.** Feeds the critical-power fit and the durability ratio inside the
endurance score, and answers "your best 5-minute power came from this ride".

**Reading trends.** The short end reflects sprint and anaerobic power; the long end reflects
aerobic endurance. Watch the curve lift over a season.

**Caveats.** Built only from recorded efforts — if you never went deep at a duration, that
point reflects pacing, not your ceiling. Windows across long gaps are excluded.

### W'balance (`wbal`)

**Units:** joules (the remaining anaerobic reserve through an effort).

Your anaerobic "battery" level, second by second, as you spend it above critical power and
recover it below.

**Formula.** The Skiba (2012) differential model on the per-second power stream, seeded at
W'. Above critical power it depletes by `(P - CP)` per second; below, it recovers toward W'
with a power-dependent time constant `tau = 546 * exp(-0.01 * (CP - P)) + 316`. (Computed by
the analytics engine, the Skiba W'balance model.)

**Inputs & when unavailable.** Needs a power stream plus critical power and W'. Unavailable
without them; not applicable for sports without power. The engine never substitutes FTP for
critical power silently.

**Typical values.** Starts full at W' and never exceeds it; it can go negative on sustained
over-exhaustion (a permitted modelled state, not an error) unless an optional floor-at-zero
policy is enabled.

**How it moves state.** Shows how deep into the anaerobic reserve an effort or interval set
took you, and how recovery between efforts replenishes it.

**Reading trends.** Repeated dips toward zero with incomplete recovery mark a session that
exhausted the reserve.

**Caveats.** Sensitive to correct critical power and W'. The negative-allowed default and the
optional floor are explicit, reported policies.

## Efficiency family

These metrics describe how smooth, how efficient, and how drift-resistant an effort was.

### Efficiency Factor (`efficiency_factor`)

**Units:** watts per beat (Normalized Power per average heart rate).

Aerobic efficiency: how much normalized power you produced per heartbeat.

**Formula.** `EF = NP / avg_hr_bpm`, where average heart rate is the mean over the full
valid-moving window. Computed only when Normalized Power is computed and average heart rate
is positive. (Computed by the analytics engine.)

**Inputs & when unavailable.** Needs both a power and a heart-rate channel over the effort;
unavailable otherwise.

**Typical values.** Strongly individual; meaningful only against your own history at similar
intensity, so no universal reference range applies.

**How it moves state.** A descriptor on the activity detail; a key aerobic-fitness trend.

**Reading trends.** Rising Efficiency Factor for similar sessions over weeks is a classic
sign of improving aerobic fitness.

**Caveats.** Confounded by heat, fatigue, and cardiac drift. Compare like with like
(similar duration and intensity).

### Variability Index (`variability_index`)

**Units:** ratio (Normalized Power relative to average power).

How evenly an effort was paced — near 1.0 is steady, higher is surgy.

**Formula.** `VI = NP / avg_power`, where average power is the mean over the full
valid-moving window (the Coggan Variability Index). Computed only when Normalized Power is
computed and average power is positive. (Computed by the analytics engine, the Coggan Variability Index.)

**Inputs & when unavailable.** Needs a power channel; unavailable otherwise.

**Typical values.** A steady time trial sits near 1.0-1.05; a stochastic group ride or crit
is well above (Coggan Variability Index convention; orientation).

**How it moves state.** A pacing descriptor on the activity detail.

**Reading trends.** For time-trial-style efforts, lower is better pacing; for a crit, a high
value is expected.

**Caveats.** Reconstructable from the exposed average power only when those scalars were
reproduced from the stream; for a summary-only average it may diverge, which is reported.

### Intensity class (`intensity_class`)

**Units:** ordered label, one of the exact lowercase tokens `recovery`, `endurance`,
`tempo`, `threshold`, `vo2`. A user interface may display these with friendlier casing
(for example "VO2"), but the canonical stored value is always lowercase.

A plain-language band for how hard a session was, derived from Intensity Factor.

**Formula.** A monotone banding of Intensity Factor against default cut-points: `recovery`
below 0.55, `endurance` 0.55-0.75, `tempo` 0.75-0.90, `threshold` 0.90-1.05, `vo2` at or
above 1.05. Computed only when Intensity Factor is computed. (Computed by the analytics engine.)

**Inputs & when unavailable.** Needs a computed Intensity Factor; unavailable otherwise.

**Typical values.** One of the five lowercase labels (the boundaries are configurable defaults).

**How it moves state.** A readable summary of session intensity on the activity detail.

**Reading trends.** A spread of classes across a week reflects a polarized or pyramidal
distribution.

**Caveats.** Inherits Intensity Factor's dependence on a correct FTP. The cut-points are
defaults and can be overridden.

### Aerobic decoupling (`decoupling`)

**Units:** percent.

How much your efficiency drifted from the first to the second half of a steady effort — the
classic cardiac-drift measure.

**Formula.** `decoupling% = ((eff_first_half - eff_second_half) / eff_first_half) * 100`,
where each half's efficiency is mean output over mean heart rate, split at the midpoint of
elapsed time, with coasting samples excluded and smoothed power used. (Computed by the
analytics engine, the aerobic-decoupling / cardiac-drift method.)

**Inputs & when unavailable.** Needs synchronized output (power, or speed where power is
absent) and heart rate over a long, steady effort (by default at least 20 minutes).
Unavailable if too short, too variable, or missing a channel.

**Typical values.** A commonly cited guideline treats under about 5% as good aerobic
durability for a steady effort (a widely used endurance-coaching convention; orientation,
not a strict cutoff).

**How it moves state.** Lower drift contributes to a higher endurance score.

**Reading trends.** Lower is better; rising decoupling at equal intensity suggests fatigue,
heat stress, under-fuelling, or insufficient aerobic base.

**Caveats.** Only meaningful for steady efforts; intervals and variable rides are not valid
inputs. Heat and dehydration inflate it independently of fitness.

## Heart-rate family

The heart-rate family works for any sport that records heart rate, and is the load path
when power is absent.

### TRIMP (`trimp`)

**Units:** load points (training impulse).

A heart-rate-based training load that weights time by intensity using your heart-rate
reserve.

**Formula.** `TRIMP = sum over valid seconds of dt_min * HRR * a * exp(b * HRR)`, where
`HRR = (HR - HR_rest) / (HR_max - HR_rest)` and `(a, b)` are the sex-specific Banister
constants (0.64 and 1.92 for men, 0.86 and 1.67 for women). (Computed by the analytics engine, the Banister TRIMP.)

**Inputs & when unavailable.** Needs a heart-rate stream and in-effect maximum and resting
heart rates. Missing input reports missing-required-input; present-but-inverted values
(maximum not above resting) report out-of-domain — distinct, honest reasons.

**Typical values.** Comparable to TSS for similar sessions; orientation only.

**How it moves state.** It is the canonical heart-rate load (`hr_load`) that substitutes for
power TSS in the daily load when power is absent.

**Reading trends.** Read like TSS for volume-times-intensity.

**Caveats.** Needs your sex for the correct constants; without it, a documented sex-neutral
default is used at reduced confidence or the metric is unavailable, per policy. Heart-rate
samples outside the reserve are clamped for weighting only, and the clamped fraction is
reported.

### Computed RMSSD (`hrv_rmssd_ms`)

**Units:** milliseconds.

Your heart-rate variability in the RMSSD statistic, computed from artifact-corrected
beat-to-beat intervals.

**Formula.** `RMSSD = sqrt(mean((NN[i+1] - NN[i])^2))` over the normal-to-normal intervals,
after mandatory artifact correction. (Computed by the analytics engine.) When only a source
summary exists, that summary is surfaced at summary-only fidelity instead.

**Inputs & when unavailable.** Needs a beat-to-beat interval stream and enough usable data
(by default at least two minutes). Unavailable if too short, too artifact-laden (above the
correction ceiling), or with neither intervals nor a summary — never returned as zero.

**Typical values.** Strongly individual and method-dependent, commonly tens of
milliseconds; no universal reference range applies — read it against your own baseline.

**How it moves state.** Feeds readiness reasoning when a personal baseline band is
available; without a baseline, readiness is read from form alone.

**Reading trends.** Trends against your own baseline matter most; a sustained drop can
indicate fatigue or illness.

**Caveats.** Very sensitive to measurement conditions. Frequency-domain HRV (band powers)
additionally requires a signal-processing stack; when it is unavailable the engine reports a
dependency gap, never fake zeros. (This computed value shares the `hrv_rmssd_ms` key with
the source summary above; the computed pipeline produces it from beat-to-beat data, the
summary is the source's own number.)

## Composites

### Endurance score (`endurance_score`)

**Units:** 0-100.

A single bounded summary of your current aerobic endurance capacity, composed only from
metrics already defined.

**Formula.** A documented, weighted, normalized blend of chronic training load (CTL), a
long-duration power durability ratio from the power curve, and aerobic decoupling (lower
drift scores higher), each normalized to 0-1 and combined to a 0-100 score over the present
components. (Computed by the analytics engine.) It introduces no new physiological model.

**Inputs & when unavailable.** CTL is required; the power-family components may be absent.
Unavailable when a non-substitutable component (CTL) is missing; otherwise it composes on
the available components with reduced confidence recorded in the quality report. A missing
component is never scored as zero.

**Typical values.** 0-100 by construction; higher means greater aerobic endurance capacity.
The mapping is a declared normalization, so no external reference range applies.

**How it moves state.** A high-level fitness summary for the athlete view; it composes
existing metrics rather than feeding new computation.

**Reading trends.** Rises with higher chronic load, better long-duration durability, and
lower decoupling. It moves monotonically in each component in the documented direction.

**Caveats.** Distinct from any source-reported endurance score. As a composite it inherits
the caveats of its inputs; when computed on a partial component set, its confidence is
reduced and reported.

---

# Arriving in an upcoming release

The entries below describe fields that are part of the canonical model and are documented
here in full because the description is accurate for the incoming feature. **Current builds
do not yet collect or compute them** — they land with an upcoming session-RPE release. They
are kept out of the implemented index and sections above so that nothing here is mistaken
for a value you can read today.

### Session RPE (`perceived_exertion`)

**Units:** CR-10 scale, 0-10 (fractional allowed).

Your rating of how hard the whole session felt, on the Borg CR-10 scale.

**Capture rule.** Captured at ingest from a device session self-evaluation, a connected
platform's RPE field, or first-party entry. Malformed, out-of-range, or
encoding-ambiguous values resolve to a typed gap — never clamped, never guessed.

**Inputs & when unavailable.** Present only when you (or a source) recorded it; a gap
otherwise.

**Typical values.** 0 = rest, around 3-4 = easy, 5-6 = moderate, 7-8 = hard, 10 = maximal
(Borg CR-10 convention, Foster's session-RPE literature).

**How it moves state.** When present with a valid duration, it produces `srpe_load`, the
lowest-fidelity training-load member, which lets sensor-less sessions (strength, many
swims) register on the Performance Management Chart instead of reading as rest.

**Reading trends.** Consistent RPE at a given workout type is normal; a creeping RPE at the
same power or pace can signal accumulating fatigue.

**Caveats.** Subjective and timing-sensitive (rate it shortly after the session). It is the
least precise load input and always carries a substituted, reduced-confidence flag when it
wins the load selection.

### Session feel (`feel`)

**Units:** ordinal 1-5.

Your subjective sense of how good you felt, on a 1-5 scale.

**Capture rule.** Captured as an ordinal where 1 = strong and 5 = weak (the common
connected-platform convention). A self-report, advisory only.

**Inputs & when unavailable.** Present only when recorded.

**Typical values.** 1-5 on the stated convention.

**How it moves state.** None directly — it is advisory context for the coach. It is never
an input to a canonical derived metric.

**Reading trends.** A run of "weak" ratings alongside flat or declining performance can
prompt a closer look at recovery.

**Caveats.** Note the scale direction (low is good). Purely subjective.

### Session-RPE load (`srpe_load`)

**Units:** load points (commensurate with TSS).

A load derived from your session RPE, so sensor-less sessions still register instead of
reading as rest.

**Formula.** `srpe_load = (RPE/10)^2 * hours * 100`, the intensity-squared,
TSS-commensurate mapping, so a maximal (CR-10 of 10) hour equals 100 points. Foster's
classical linear figure (`RPE * minutes`) is carried alongside as provenance only.
(Computed by the analytics engine, after Foster's session-RPE method.)

**Inputs & when unavailable.** Needs a valid session RPE (greater than 0, up to 10) and a
positive duration. Unavailable otherwise; a malformed RPE is a gap, never a guessed value.

**Typical values.** A CR-10 of 7 for one hour is about 49 points; a maximal hour is 100
(by the formula). Orientation only.

**How it moves state.** It is the lowest-fidelity training-load member. It wins the load
selection only when no power or heart-rate load can be computed, and it always carries a
substituted, reduced-confidence flag.

**Reading trends.** Lets strength and many swim sessions appear on the Performance
Management Chart; read its contribution as approximate.

**Caveats.** Subjective input, lowest precision. One activity contributes exactly one load
member, so logging RPE alongside power never double-counts.
