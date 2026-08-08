"""VIN validation and the US-vs-Euro parts check. All offline, no network.

The VIN checksum matters more here than it would elsewhere. Armenia's fleet is
mostly rebuilt North American cars, and North America is exactly where the
position-9 check digit is mandated (49 CFR 565 / ISO 3779). So a failed
checksum is almost always a misread character rather than an unusual VIN --
which is what lets the system refuse instead of guessing, and a wrong VIN
sends someone to buy an expensive part for a car they do not own.

Known limit, worth saying out loud: the transliteration table maps several
characters to the same number (1/A/J, 2/B/K/S, ...), so a substitution inside
one of those classes is arithmetically invisible. What rescues it in practice
is that I, O and Q are illegal in a VIN -- killing the 0/O and 1/I confusions
outright -- and every remaining pair a camera actually confuses (5/S, 8/B,
2/Z, 6/G, 1/7) falls in a different class. It catches every realistic OCR
error, not every conceivable one.
"""

from __future__ import annotations

import re

from carmed.models import FitmentWarning, PartListing, SpecRegion, Vehicle

# Letter -> number, as specified by 49 CFR 565. Digits map to themselves.
# Note the deliberate collisions (1/A/J all become 1) -- they are in the
# standard, and they are why a within-class typo is undetectable.
_TRANSLIT = {
    **{str(d): d for d in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}  # fmt: skip

# Positional weights. Position 9 weighs 0 because that is the check digit
# itself -- it must not contribute to the sum that produces it.
_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)

# 17 characters, and I/O/Q excluded -- they were dropped from the alphabet
# precisely because they look like 1 and 0.
_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

# First character of the VIN = where the car was built for sale. This is the
# whole basis of the US-vs-Euro parts check, and it needs no network call.
_REGION_BY_FIRST = {
    "1": SpecRegion.US, "4": SpecRegion.US, "5": SpecRegion.US,
    "2": SpecRegion.CANADA, "3": SpecRegion.MEXICO,
    "J": SpecRegion.JAPAN, "K": SpecRegion.KOREA,
    "S": SpecRegion.EU, "T": SpecRegion.EU, "V": SpecRegion.EU,
    "W": SpecRegion.EU, "X": SpecRegion.EU, "Y": SpecRegion.EU,
    "Z": SpecRegion.EU,
}  # fmt: skip

#: Characters an OCR pass most often confuses. Used only to make a refusal
#: actionable -- "check these characters" beats "invalid VIN" when the user is
#: squinting at a dirty door jamb plate.
_CONFUSABLE = {"0": "O/D/Q", "1": "I/L/7", "5": "S", "8": "B",
               "2": "Z", "6": "G", "B": "8", "S": "5", "Z": "2", "G": "6"}  # fmt: skip


def normalize_vin(raw: str) -> str:
    """Uppercase and drop anything that cannot appear in a VIN.

    People paste VINs with spaces, dashes and stray quotes. Strip first, judge
    second, so formatting noise is never reported as a checksum failure.
    """
    return re.sub(r"[^A-Z0-9]", "", (raw or "").upper())


def is_well_formed(vin: str) -> bool:
    """Right length, legal alphabet. Says nothing about the checksum."""
    return bool(_VIN_RE.match(vin))


def check_digit(vin: str) -> str:
    """The character that *should* sit at position 9.

    Transliterate each character, multiply by its positional weight, sum, then
    take modulo 11. A remainder of 10 is written as 'X' -- the reason a letter
    can legally appear in an otherwise numeric slot.
    """
    total = sum(_TRANSLIT[c] * w for c, w in zip(vin, _WEIGHTS, strict=True))
    return "X" if total % 11 == 10 else str(total % 11)


def is_valid(vin: str) -> bool:
    """Well-formed AND the checksum agrees with position 9."""
    return is_well_formed(vin) and vin[8] == check_digit(vin)


def region_of(vin: str) -> SpecRegion:
    """Market the car was built for, from the first character alone."""
    return _REGION_BY_FIRST.get(vin[:1], SpecRegion.UNKNOWN) if vin else SpecRegion.UNKNOWN


def inspect(raw: str) -> tuple[bool, SpecRegion, str]:
    """Return ``(valid, region, problem)``. ``problem`` is empty when valid.

    Two failure modes, reported differently because the user's next action
    differs: a malformed VIN means they mistyped the length or used an illegal
    letter, while a checksum failure means the shape is right but one
    character is wrong -- so we point at the likely culprits.
    """
    vin = normalize_vin(raw)

    if not is_well_formed(vin):
        why = (
            f"A VIN is 17 characters and never contains I, O or Q -- got {len(vin)}."
            if len(vin) != 17
            else "Contains a character that cannot appear in a VIN."
        )
        return False, SpecRegion.UNKNOWN, why

    if vin[8] != check_digit(vin):
        suspects = sorted({c for c in vin if c in _CONFUSABLE})
        hint = (
            " Check these first: "
            + ", ".join(f"{c} (vs {_CONFUSABLE[c]})" for c in suspects[:4])
            if suspects
            else ""
        )
        return (
            False,
            region_of(vin),
            f"Checksum failed: position 9 is '{vin[8]}' but should be "
            f"'{check_digit(vin)}'. One character is probably misread.{hint}",
        )

    return True, region_of(vin), ""


def resolve(vehicle: Vehicle) -> tuple[Vehicle, str]:
    """Fill in ``vin_valid`` and ``region``. Returns the reason if a VIN failed."""
    if not vehicle.vin:
        return vehicle, ""
    valid, region, problem = inspect(vehicle.vin)
    updated = vehicle.model_copy(
        update={
            "vin": normalize_vin(vehicle.vin),
            "vin_valid": valid,
            "region": region if region is not SpecRegion.UNKNOWN else vehicle.region,
        }
    )
    return updated, problem


# ---------------------------------------------------------------------------
# US-spec car vs Euro-market part
# ---------------------------------------------------------------------------

_EURO_MARKERS = ("euro", "european", "eu spec", "e-code", "ecode",
                 "европ", "евро", "եվրո", "եվրոպական")  # fmt: skip
_US_MARKERS = ("usa", "us spec", "american", "us market", "sae", "dot",
               "сша", "американ", "ամերիկ")  # fmt: skip

_US_LIKE = {SpecRegion.US, SpecRegion.CANADA, SpecRegion.MEXICO}


def _listing_region(listing: PartListing) -> SpecRegion:
    if listing.region is not SpecRegion.UNKNOWN:
        return listing.region
    title = listing.title.casefold()
    if any(m in title for m in _EURO_MARKERS):
        return SpecRegion.EU
    if any(m in title for m in _US_MARKERS):
        return SpecRegion.US
    return SpecRegion.UNKNOWN


def fitment_warnings(
    vehicle: Vehicle, listings: list[PartListing]
) -> list[FitmentWarning]:
    """Flag listings whose market does not match the car's build.

    Deliberately dumb -- VIN region plus listing text. It is not a fitment
    database and does not pretend to be one; it raises a flag for a human to
    check, which is the honest version of this feature. Says nothing at all
    when the car's region is unknown, rather than guessing.
    """
    if vehicle.region is SpecRegion.UNKNOWN:
        return []

    # Group by which market the mismatched parts came from, so the user gets
    # one clear warning per market rather than one per listing.
    mismatched: dict[SpecRegion, list[str]] = {}
    for listing in listings:
        guess = _listing_region(listing)
        if guess is SpecRegion.UNKNOWN:
            continue  # cannot tell -> say nothing, rather than guess
        # Only cross-Atlantic mismatches matter. US/Canada/Mexico parts are
        # interchangeable in practice; US vs EU is where money gets wasted.
        us_car, euro_part = vehicle.region in _US_LIKE, guess is SpecRegion.EU
        euro_car, us_part = vehicle.region is SpecRegion.EU, guess in _US_LIKE
        if (us_car and euro_part) or (euro_car and us_part):
            mismatched.setdefault(guess, []).append(listing.id)

    return [
        FitmentWarning(
            severity="blocking",
            message=(
                f"Your car is a {vehicle.region.value}-market build, but "
                f"{len(ids)} listing(s) look like {region.value}-market parts. "
                "The same model name does not mean the same part -- check the "
                "part number against your VIN before paying."
            ),
            listing_ids=ids,
        )
        for region, ids in mismatched.items()
    ]
