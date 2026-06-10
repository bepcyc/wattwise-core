# WattWise content & voice style guide

The single source of truth for every string a person reads — API error and status
copy, and the coaching agent's athlete-facing output. The runtime persona/voice
configuration is the runtime authority; this guide is the human-review rubric, and
the two must stay consistent. It does not govern internal logs, machine codes, or
developer-facing strings.

## Voice

- Plain, calm, warm, human. A coach talking to an athlete — never a system
  reporting its internals, never a dashboard reading numbers aloud.
- Lead with a plain-language read of the athlete's state; keep numbers in the
  background.
- Address the athlete directly ("you"); contractions are welcome.

## Error and validation messages

Structure every message as:

1. **What happened** — specific: name the offending value, bound, field, or format.
2. **Why** — only when it adds something.
3. **How to fix it** — an imperative next step.

Distinct failures get distinct messages. Never a bare "an error occurred" or
"required". Every message pairs with a stable machine code clients branch on —
clients must never keyword-match the human sentence. Unrecoverable failures fall
back to one generic-but-safe message with a support reference.

## Banned vocabulary

- **Blame words** (machine code names only, never user copy): invalid, illegal,
  forbidden, prohibited, incorrect, "you forgot", "you failed". The system or the
  data state is the subject, never the user.
- **Edgy / robotic tokens**: oops, whoops, uh-oh, yikes, gotcha, pwned; leetspeak,
  memes, emoji-as-status, exclamation-heavy hype.
- **Over-apology**: no "please"/"sorry" in routine errors. One apology only for a
  genuine our-side outage.
- `!` is reserved for genuine positive milestones — never in errors, validation,
  or empty states.

## No leakage, no jargon

User copy must never contain stack traces, exception names, SQL, HTTP or provider
internals, tokens, hex codes, or developer vocabulary (adapter, endpoint, schema,
sync engine, fidelity enums, source descriptors). Data states are surfaced in human
terms: "We couldn't reach Garmin right now — your numbers still work with what we
have", never an enum value.

Give the fix unless the fix is a security risk: a login failure never discloses
whether the username or the password was wrong.

## Plain language

- Target an 8th-grade reading level or below for body copy.
- Error bodies: at most ~2 sentences / ~160 characters.
- Prefer plain words and contractions; jargon only from the approved glossary.

## Internationalization

- Every user-facing string resolves through the keyed copy catalog — never an
  inline literal in logic. Code references a stable key; the catalog holds the
  wording.
- Localized copy is assembled from whole, translatable entries — never concatenated
  fragments or interpolation into a hardcoded word order.
- Pluralization, gender, number, date, and unit formatting are locale-correct.
- Every key exists in every supported locale, or falls back to English with a
  human-readable notice. No surface ever mixes languages or shows an untranslated
  internal string.

## Validation UX (for clients rendering this API's errors)

- Preserve the user's input on failure — never clear the field.
- Render the message next to the offending field and in a summary, with identical
  wording.
- Validate on blur/submit, not on the first keystroke.

## Review

The mechanical rules above are enforced by the content lint and the per-locale copy
snapshot tests. The non-mechanical dimensions — read-aloud naturalness, whether the
fix is genuinely helpful, coaching warmth — are a human-review checklist item on
every change that touches user-facing copy.
